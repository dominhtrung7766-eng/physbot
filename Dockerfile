# ── Base image ─────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System deps ────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# ── Workdir ────────────────────────────────────────────────────────
WORKDIR /app

# ── Copy requirements trước (tận dụng layer cache) ─────────────────
COPY requirements.txt .

# ── Cài Python deps ─────────────────────────────────────────────────
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

# ── Copy toàn bộ folder backend/ (chứa main.py + các module) ────────
COPY backend/ ./backend/

# ── Tạo thư mục logs và data ─────────────────────────────────────────
RUN mkdir -p logs data/chroma_db

# ── Port ─────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Start — main.py nằm trong backend/ nên dùng backend.main:app ─────
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
