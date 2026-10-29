"""Download required Hugging Face assets once for the Windows app."""
import gc
import os
from pathlib import Path

os.environ["USE_TF"] = "0"
os.environ["USE_FLAX"] = "0"
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
from transformers import AutoTokenizer, AutoModel, ViTModel

root = Path(__file__).resolve().parents[1]
for name in ["mabsa_best.pt", "mabsa_before_retrain_20261012.pt"]:
    if not (root / "checkpoints" / name).is_file():
        raise FileNotFoundError(f"Missing checkpoint: {name}. Extract the complete ZIP first.")
print("Downloading PhoBERT tokenizer and model cache...", flush=True)
AutoTokenizer.from_pretrained("vinai/phobert-base")
model = AutoModel.from_pretrained("vinai/phobert-base", attn_implementation="eager")
del model
gc.collect()
print("Downloading ViT model cache...", flush=True)
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k",
                                add_pooling_layer=False, attn_implementation="eager")
del model
gc.collect()
(root / ".windows_models_ready").write_text("Model cache downloaded for this Windows user.\n")
print("Model cache ready. No training was performed.", flush=True)
