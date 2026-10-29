"""Tập khía cạnh (Bảng 3.3) và nhãn cảm xúc (mục 1.2.4)."""

# 10 khía cạnh của Bảng 3.3: (mã ngắn, tên tiếng Việt)
ASPECTS = [
    ("FOOD", "Chất lượng món ăn"),
    ("SPACE", "Không gian"),
    ("PRICE", "Giá cả"),
    ("SERVICE", "Phục vụ"),
    ("LOCATION", "Vị trí"),
    ("ROOM", "Phòng ốc"),
    ("FACILITY", "Tiện nghi"),
    ("SCENERY", "Cảnh quan"),
    ("CLEANLINESS", "Vệ sinh"),
    ("GENERAL", "Tổng thể"),
]
ASPECT_CODES = [c for c, _ in ASPECTS]
ASPECT_NAMES = [n for _, n in ASPECTS]
CODE2NAME = dict(ASPECTS)
NAME2CODE = {n: c for c, n in ASPECTS}
ASPECT2ID = {c: i for i, c in enumerate(ASPECT_CODES)}

# 3 nhãn cảm xúc
SENTIMENTS = ["Tích cực", "Tiêu cực", "Trung lập"]
SENT2ID = {s: i for i, s in enumerate(SENTIMENTS)}
SENT_ALIASES = {
    "positive": "Tích cực", "pos": "Tích cực", "tich cuc": "Tích cực",
    "negative": "Tiêu cực", "neg": "Tiêu cực", "tieu cuc": "Tiêu cực",
    "neutral": "Trung lập", "neu": "Trung lập", "trung lap": "Trung lập",
}

# Loại hình dịch vụ (mục 3.1.3)
SERVICE_TYPES = ["hotel", "homestay", "restaurant", "attraction"]

# Từ khóa gợi ý khía cạnh: dùng cho gán nhãn bán tự động (mục 3.3.4)
# và làm phương án dự phòng khi chưa huấn luyện bộ trích xuất khía cạnh (mục 4.4.1).
ASPECT_LEXICON = {
    "FOOD": ["món ăn", "đồ ăn", "thức ăn", "món", "cà phê", "đồ uống", "nước uống", "hương vị", "bữa sáng",
             "buffet", "phở", "bún", "lẩu", "hải sản", "espresso", "trà", "bánh"],
    "SPACE": ["không gian", "bầu không khí", "không khí", "bài trí", "trang trí", "quán"],
    "PRICE": ["giá", "giá cả", "chi phí", "giá tiền", "hóa đơn", "đắt", "rẻ"],
    "SERVICE": ["nhân viên", "phục vụ", "lễ tân", "thái độ", "chủ nhà", "dịch vụ"],
    "LOCATION": ["vị trí", "địa điểm", "gần trung tâm", "đường đi", "chỗ đậu xe"],
    "ROOM": ["phòng", "phòng ngủ", "giường", "phòng tắm", "ban công"],
    "FACILITY": ["tiện nghi", "wifi", "điều hòa", "hồ bơi", "bể bơi", "thang máy", "tivi", "máy lạnh"],
    "SCENERY": ["view", "cảnh", "phong cảnh", "cảnh quan", "cảnh đẹp", "hoàng hôn", "biển", "núi"],
    "CLEANLINESS": ["vệ sinh", "sạch sẽ", "sạch", "bẩn", "nhà vệ sinh"],
    "GENERAL": ["tổng thể", "nói chung", "nhìn chung", "trải nghiệm", "sẽ quay lại"],
}


def normalize_sentiment(label: str) -> str:
    if label in SENT2ID:
        return label
    key = str(label).strip().lower()
    if key in SENT_ALIASES:
        return SENT_ALIASES[key]
    raise ValueError(f"Nhãn cảm xúc không hợp lệ: {label!r}")


def normalize_aspect(category: str) -> str:
    """Nhận mã (FOOD) hoặc tên tiếng Việt (Chất lượng món ăn), trả về mã."""
    if category in ASPECT2ID:
        return category
    if category in NAME2CODE:
        return NAME2CODE[category]
    up = str(category).strip().upper()
    if up in ASPECT2ID:
        return up
    raise ValueError(f"Khía cạnh không hợp lệ: {category!r}")
