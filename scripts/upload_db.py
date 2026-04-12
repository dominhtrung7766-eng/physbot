"""
scripts/upload_db.py
────────────────────
Upload ChromaDB snapshot lên Hugging Face Hub.
Chạy 1 lần trên máy dev khi DB đã sẵn sàng và không còn update.

Dùng:
    python scripts/upload_db.py
    python scripts/upload_db.py --version v1.1  # đặt tên version
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def upload(version: str = "v1.0"):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("Thiếu thư viện. Chạy: pip install huggingface_hub")
        sys.exit(1)

    hf_token   = os.getenv("HF_TOKEN")
    hf_repo_id = os.getenv("HF_REPO_ID")

    if not hf_token:
        print("Thiếu HF_TOKEN trong .env")
        sys.exit(1)
    if not hf_repo_id:
        print("Thiếu HF_REPO_ID trong .env  (ví dụ: your-username/physbot-chromadb)")
        sys.exit(1)

    db_path = Path("data/chroma_db")
    if not db_path.exists() or not any(db_path.iterdir()):
        print(f"ChromaDB không tồn tại tại: {db_path}")
        sys.exit(1)

    # Tính size
    total_bytes = sum(f.stat().st_size for f in db_path.rglob("*") if f.is_file())
    total_mb    = total_bytes / 1024 / 1024
    print(f"ChromaDB size: {total_mb:.1f} MB")

    api = HfApi(token=hf_token)

    # Tạo repo nếu chưa có (bỏ qua nếu đã tồn tại)
    try:
        api.create_repo(
            repo_id=hf_repo_id,
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )
        print(f"Repo: {hf_repo_id} (private)")
    except Exception as e:
        print(f"Tạo repo lỗi (có thể đã tồn tại): {e}")

    print(f"Đang upload... (version={version})")
    api.upload_folder(
        folder_path=str(db_path),
        repo_id=hf_repo_id,
        repo_type="dataset",
        path_in_repo="chroma_db",        # lưu trong subfolder chroma_db/
        commit_message=f"ChromaDB snapshot {version}",
    )

    print(f"\nDone! DB đã up lên: https://huggingface.co/datasets/{hf_repo_id}")
    print("Người dùng download bằng: python scripts/download_db.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1.0", help="Tên version (ví dụ: v1.1)")
    args = parser.parse_args()
    upload(args.version)