"""
scripts/download_db.py
──────────────────────
Download ChromaDB từ Hugging Face Hub về local.

Dùng:
    python scripts/download_db.py            # auto-skip nếu đã có
    python scripts/download_db.py --force    # download lại dù đã có
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def download(force: bool = False):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Thiếu thư viện. Chạy: pip install huggingface_hub")
        sys.exit(1)

    hf_token   = os.getenv("HF_TOKEN")
    hf_repo_id = os.getenv("HF_REPO_ID")

    if not hf_repo_id:
        print("Thiếu HF_REPO_ID trong .env  (ví dụ: your-username/physbot-chromadb)")
        sys.exit(1)

    db_path = Path("data/chroma_db")

    # Kiểm tra đã có chưa
    if not force and db_path.exists() and any(db_path.iterdir()):
        print(f"ChromaDB đã có tại {db_path}, bỏ qua. Dùng --force để download lại.")
        return

    print(f"Đang download ChromaDB từ {hf_repo_id}...")
    db_path.mkdir(parents=True, exist_ok=True)

    # Download toàn bộ repo vào thư mục tạm
    tmp_path = Path("data/_hf_download_tmp")
    try:
        snapshot_download(
            repo_id=hf_repo_id,
            repo_type="dataset",
            local_dir=str(tmp_path),
            token=hf_token,          # None nếu repo public
        )

        # Di chuyển subfolder chroma_db/ vào đúng vị trí
        src = tmp_path / "chroma_db"
        if src.exists():
            import shutil
            if db_path.exists():
                shutil.rmtree(db_path)
            shutil.move(str(src), str(db_path))
        else:
            # Repo không có subfolder, dùng thẳng
            import shutil
            if db_path.exists():
                shutil.rmtree(db_path)
            shutil.move(str(tmp_path), str(db_path))

        print(f"Download xong! ChromaDB tại: {db_path}")

        # Kiểm tra size
        total_bytes = sum(f.stat().st_size for f in db_path.rglob("*") if f.is_file())
        print(f"Size: {total_bytes/1024/1024:.1f} MB")

    except Exception as e:
        print(f"Lỗi download: {e}")
        sys.exit(1)
    finally:
        # Dọn tmp
        if tmp_path.exists():
            import shutil
            shutil.rmtree(tmp_path, ignore_errors=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Download lại dù đã có")
    args = parser.parse_args()
    download(args.force)