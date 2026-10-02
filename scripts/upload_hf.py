"""Upload Maya to Hugging Face.

    python scripts/upload_hf.py --model checkpoints/maya     # -> VishalMysore/maya (PyTorch weights + config)
    python scripts/upload_hf.py --web build/web              # -> VishalMysore/mayaWasm (int8 ONNX, browser)

The model cards are hf/maya.md and hf/mayaWasm.md.
"""
import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent


def upload(folder, repo, card, message):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for f in Path(folder).iterdir():
            if f.is_file() and f.name != "train_log.json":
                shutil.copy(f, tmp / f.name)
        shutil.copy(card, tmp / "README.md")
        for f in ("LICENSE", "NOTICE.md"):
            shutil.copy(ROOT / f, tmp / f)
        HfApi().upload_folder(folder_path=str(tmp), repo_id=repo, commit_message=message)
    print(f"uploaded {folder} -> https://huggingface.co/{repo}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--web")
    ap.add_argument("--message", default="Upload Maya")
    args = ap.parse_args()
    if args.model:
        upload(args.model, "VishalMysore/maya", ROOT / "hf" / "maya.md", args.message)
    if args.web:
        upload(args.web, "VishalMysore/mayaWasm", ROOT / "hf" / "mayaWasm.md", args.message)


if __name__ == "__main__":
    main()
