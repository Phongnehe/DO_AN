"""Tiền xử lý văn bản và hình ảnh (mục 3.2, Bảng 3.2) và tách từ tiếng Việt cho PhoBERT."""
import re
import unicodedata
from pathlib import Path

import numpy as np
from PIL import Image

from mabsa.utils import get_logger

log = get_logger()

MIN_WORDS = 10          # tiêu chí 3.1.3: tối thiểu 10 từ
MIN_IMAGE_SIDE = 224    # tiêu chí 3.1.3 / 3.2.2: ảnh tối thiểu 224×224

# Emoji mang cảm xúc rõ ràng được giữ lại (mục 3.2.1)
EMOJI_WHITELIST = set("😊😍😡😢😭😂🥰😋👍👎❤😞😠🤩😒")
_EMOJI_RE = re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF\U00002B00-\U00002BFF️‍]"
)
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_REPEAT_PUNCT = re.compile(r"([!?.,;:~\-])\1+")
_REPEAT_CHAR = re.compile(r"(\w)\1{3,}")           # "ngonnnnn" -> "ngonn"
_ALLOWED = re.compile(r"[^\w\s.,!?;:()\-/%&+'\"“”]", flags=re.UNICODE)


def normalize_text(text: str) -> str:
    """Chuẩn hóa NFC, bỏ URL, emoji không mang cảm xúc, ký tự đặc biệt, dấu câu lặp."""
    text = unicodedata.normalize("NFC", str(text))
    text = _URL_RE.sub(" ", text)
    text = "".join(ch if (ch in EMOJI_WHITELIST or not _EMOJI_RE.match(ch)) else " " for ch in text)
    kept = []
    for ch in text:
        if ch in EMOJI_WHITELIST:
            kept.append(f" {ch} ")
        elif _ALLOWED.match(ch):
            kept.append(" ")
        else:
            kept.append(ch)
    text = "".join(kept)
    text = _REPEAT_PUNCT.sub(r"\1", text)
    text = _REPEAT_CHAR.sub(r"\1\1", text)
    return re.sub(r"\s+", " ", text).strip()


def count_words(text: str) -> int:
    return len([w for w in text.split() if any(c.isalnum() for c in w)])


def looks_like_spam(text: str) -> bool:
    """Phát hiện đơn giản đánh giá quảng cáo/spam (mục 3.2.1)."""
    t = text.lower()
    if re.search(r"(\d[\s.]?){9,}", t):          # số điện thoại
        return True
    keywords = ["inbox", "liên hệ zalo", "khuyến mãi hot", "giảm giá sốc", "click vào link", "mã giảm giá"]
    return any(k in t for k in keywords)


class WordSegmenter:
    """Tách từ tiếng Việt: VnCoreNLP (khuyến nghị cho PhoBERT) → underthesea → không tách.

    VnCoreNLP cần Java (JDK 8+). Lần đầu chạy, mô hình được tải về `save_dir`.
    """

    def __init__(self, backend: str = "vncorenlp", save_dir: str = "vncorenlp"):
        self.backend = "none"
        self._impl = None
        if backend == "vncorenlp":
            try:
                import py_vncorenlp
                save_dir = str(Path(save_dir).resolve())
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                if not (Path(save_dir) / "VnCoreNLP-1.2.jar").exists():
                    py_vncorenlp.download_model(save_dir=save_dir)
                self._impl = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=save_dir)
                self.backend = "vncorenlp"
            except Exception as e:  # thiếu Java hoặc chưa cài py_vncorenlp
                log.warning("Không khởi tạo được VnCoreNLP (%s). Chuyển sang underthesea.", e)
                backend = "underthesea"
        if backend == "underthesea" and self._impl is None:
            try:
                from underthesea import word_tokenize
                self._impl = word_tokenize
                self.backend = "underthesea"
            except Exception as e:
                log.warning("Không có underthesea (%s). Văn bản sẽ không được tách từ.", e)

    def __call__(self, text: str) -> str:
        if not text.strip():
            return text
        if self.backend == "vncorenlp":
            return " ".join(self._impl.word_segment(text))
        if self.backend == "underthesea":
            return self._impl(text, format="text")
        return text


# ------------------------------------------------------------------ ảnh
IMAGE_STATS = {
    # mean/std dùng khi tiền huấn luyện từng bộ mã hóa ảnh
    "vit": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    "clip": ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    "resnet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "tiny": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
}


def load_image(path) -> Image.Image:
    return Image.open(path).convert("RGB")


def image_is_valid(path, min_side: int = MIN_IMAGE_SIDE) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            return min(im.size) >= min_side
    except Exception:
        return False


def image_to_tensor(image: Image.Image, size: int = 224, encoder: str = "vit") -> np.ndarray:
    """Resize về size×size (Bảng 3.2) và chuẩn hóa theo mean/std của bộ mã hóa ảnh → C×H×W float32."""
    mean, std = IMAGE_STATS.get(encoder, IMAGE_STATS["vit"])
    image = image.convert("RGB").resize((size, size), Image.BICUBIC)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
    return arr.transpose(2, 0, 1).copy()
