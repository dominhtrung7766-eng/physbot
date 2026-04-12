"""
scripts/ingest_images.py

Trích hình ảnh từ PDF trong data/raw → mô tả bằng Groq Vision
→ lưu mô tả vào ChromaDB (cùng collection với text, không xóa chunks cũ)

Chạy 1 lần sau khi đã ingest text xong:
    python scripts/ingest_images.py

Yêu cầu thêm:
    pip install pymupdf  (import là fitz)
"""

import os
import re
import base64
import glob
from pathlib import Path

import fitz  # pymupdf
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Cấu hình ──────────────────────────────────────────────
PDF_DIR        = "data/exercises"   # ← chỉ RAG hình từ folder này
DB_DIR         = "data/chroma_db"
COLLECTION_NAME = "physbot_sgk"

# Groq Vision model (hỗ trợ image)
VISION_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"

# Chỉ xử lý hình có kích thước đủ lớn (tránh icon, dấu chấm...)
MIN_IMG_WIDTH  = 80   # px
MIN_IMG_HEIGHT = 80   # px

# Số hình tối đa mỗi trang (tránh trang bìa nhiều ảnh trang trí)
MAX_IMGS_PER_PAGE = 4

# File lưu danh sách ID đã embed để không embed lại khi chạy lần 2
CHECKPOINT_FILE = "data/processed/image_chunks_done.txt"
# ──────────────────────────────────────────────────────────


def load_done_ids() -> set:
    p = Path(CHECKPOINT_FILE)
    if not p.exists():
        return set()
    return set(p.read_text(encoding="utf-8").splitlines())


def save_done_id(chunk_id: str):
    Path(CHECKPOINT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_FILE, "a", encoding="utf-8") as f:
        f.write(chunk_id + "\n")


def image_to_base64(img_bytes: bytes) -> str:
    return base64.b64encode(img_bytes).decode("utf-8")


def describe_image(groq_client: Groq, img_b64: str, page_num: int,
                   filename: str) -> str | None:
    """
    Gọi Groq Vision để mô tả hình.
    Trả về None nếu hình không liên quan vật lý.
    """
    prompt = (
        "Đây là một hình ảnh trích từ sách giáo khoa Vật lý Việt Nam. "
        "Hãy mô tả ngắn gọn bằng tiếng Việt (tối đa 120 từ):\n"
        "- Đây là loại hình gì? (đồ thị, sơ đồ lực, mạch điện, thí nghiệm, ảnh minh hoạ...)\n"
        "- Các đại lượng, ký hiệu, trục tọa độ xuất hiện trong hình\n"
        "- Nội dung vật lý chính hình muốn truyền đạt\n"
        "Nếu hình chỉ là trang trí, logo hoặc không liên quan vật lý, "
        "trả lời đúng 1 từ: SKIP"
    )
    try:
        response = groq_client.chat.completions.create(
            model=VISION_MODEL,
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }]
        )
        text = response.choices[0].message.content.strip()
        if text.upper() == "SKIP" or len(text) < 20:
            return None
        return text
    except Exception as e:
        print(f"    [Vision API lỗi] {e}")
        return None


def extract_and_ingest_images(pdf_path: str, collection, model, groq_client,
                               done_ids: set):
    filename = Path(pdf_path).stem
    doc      = fitz.open(pdf_path)
    total_pages = len(doc)
    added = 0

    print(f"\n📄 {filename} ({total_pages} trang)")

    for page_num in range(total_pages):
        page      = doc[page_num]
        img_list  = page.get_images(full=True)

        # Lọc ảnh quá nhỏ
        valid_imgs = []
        for img_info in img_list:
            xref = img_info[0]
            base_img = doc.extract_image(xref)
            w, h = base_img.get("width", 0), base_img.get("height", 0)
            if w >= MIN_IMG_WIDTH and h >= MIN_IMG_HEIGHT:
                valid_imgs.append((xref, base_img))

        if not valid_imgs:
            continue

        # Giới hạn số hình/trang
        valid_imgs = valid_imgs[:MAX_IMGS_PER_PAGE]

        print(f"  Trang {page_num+1}: {len(valid_imgs)} hình hợp lệ")

        for img_idx, (xref, base_img) in enumerate(valid_imgs):
            chunk_id = f"{filename}_img_p{page_num+1}_{img_idx+1}"

            # Bỏ qua nếu đã embed rồi (checkpoint)
            if chunk_id in done_ids:
                print(f"    [{img_idx+1}] {chunk_id} → đã embed, bỏ qua")
                continue

            # Mô tả hình bằng Groq Vision
            img_b64 = image_to_base64(base_img["image"])
            print(f"    [{img_idx+1}] Đang mô tả {chunk_id}...", end=" ", flush=True)
            description = describe_image(groq_client, img_b64, page_num, filename)

            if description is None:
                print("→ SKIP (không liên quan)")
                continue

            print(f"→ OK ({len(description)} ký tự)")

            # Tạo text chunk để embed
            chunk_text = (
                f"[HÌNH MINH HOẠ - trang {page_num+1}] "
                f"Nguồn: {filename}\n"
                f"{description}"
            )

            # Embed và lưu vào ChromaDB
            embedding = model.encode(chunk_text).tolist()
            collection.add(
                documents=[chunk_text],
                embeddings=[embedding],
                ids=[chunk_id],
                metadatas=[{
                    "source":      filename,
                    "chunk_index": page_num * 100 + img_idx,  # index giả để merge_neighbor không crash
                    "type":        "image_description",
                    "page":        page_num + 1,
                }]
            )

            save_done_id(chunk_id)
            done_ids.add(chunk_id)
            added += 1

    doc.close()
    print(f"  ✅ Đã thêm {added} image chunks từ {filename}")
    return added


def main():
    # Kiểm tra GROQ_API_KEY
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Thiếu GROQ_API_KEY trong .env")

    groq_client = Groq(api_key=api_key)
    model       = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    client      = chromadb.PersistentClient(path=DB_DIR)

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        collection = client.create_collection(COLLECTION_NAME)

    done_ids  = load_done_ids()
    pdf_files = glob.glob(f"{PDF_DIR}/**/*.pdf", recursive=True)

    if not pdf_files:
        print(f"Không tìm thấy PDF trong {PDF_DIR}")
        return

    print(f"Tìm thấy {len(pdf_files)} PDF. Checkpoint: {len(done_ids)} hình đã embed trước đó.\n")

    total_added = 0
    for pdf_path in pdf_files:
        total_added += extract_and_ingest_images(
            pdf_path, collection, model, groq_client, done_ids
        )

    # Thống kê collection sau khi chạy
    count = collection.count()
    print(f"\n{'═'*50}")
    print(f"✅ Hoàn tất! Đã thêm {total_added} image chunks.")
    print(f"   Collection '{COLLECTION_NAME}' hiện có {count} chunks tổng cộng.")
    print(f"   (text chunks cũ được giữ nguyên)")


if __name__ == "__main__":
    main()