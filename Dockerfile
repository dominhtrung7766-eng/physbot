# ── Base image ─────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System deps ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Không cần Tesseract/Poppler trên Render
# (ingest.py chạy local, không deploy lên server)

# ── Workdir ────────────────────────────────────────────────────────
WORKDIR /app

# ── Copy requirements trước (tận dụng layer cache) ─────────────────
COPY requirements.txt .

# ── Cài Python deps (bỏ audio/OCR deps không cần cho server) ───────
RUN pip install --no-cache-dir \
    groq \
    sentence-transformers \
    "transformers>=4.40.0" \
    "torch>=2.2.0" \
    chromadb \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    python-dotenv \
    pydantic \
    slowapi \
    trafilatura \
    beautifulsoup4 \
    requests \
    Pillow \
    PyPDF2 \
    pdfplumber \
    pymupdf \
    selenium \
    webdriver-manager \
    rich

# ── Copy source code ────────────────────────────────────────────────
COPY main.py .
COPY backend/ ./backend/

# ── Copy ChromaDB đã build sẵn (commit vào repo hoặc dùng volume) ──
# Nếu commit ChromaDB vào repo thì uncomment dòng dưới:
# COPY data/chroma_db/ ./data/chroma_db/

# ── Tạo thư mục logs ────────────────────────────────────────────────
RUN mkdir -p logs data/chroma_db

# ── Port ─────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start ─────────────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]