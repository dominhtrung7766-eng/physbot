FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

COPY --chown=user requirements.txt .

# Cài torch CPU riêng trước
RUN pip install --no-cache-dir --user \
    torch --index-url https://download.pytorch.org/whl/cpu

# Cài phần còn lại từ PyPI bình thường
RUN pip install --no-cache-dir --user \
    groq \
    sentence-transformers \
    "transformers>=4.40.0" \
    chromadb \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    python-dotenv \
    pydantic \
    slowapi \
    huggingface_hub \
    trafilatura \
    beautifulsoup4 \
    requests \
    Pillow \
    PyPDF2 \
    pdfplumber \
    pymupdf \
    rich
ENV SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/torch/sentence_transformers
ENV TRANSFORMERS_CACHE=/home/user/.cache/huggingface/transformers
ENV HF_HOME=/home/user/.cache/huggingface
# Pre-download model vào cache trong image luôn
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

ENV TRANSFORMERS_OFFLINE=1
ENV HF_DATASETS_OFFLINE=1

COPY --chown=user . .

RUN mkdir -p logs data/chroma_db

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
