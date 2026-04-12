🔬 PhysBot — Voice Assistant Vật Lý THPT
Trợ lý giọng nói chuyên vật lý THPT lớp 10–12, tích hợp RAG, tool calling tính toán, TTS tiếng Việt, và conversation history.

✨ Tính năng

🎙️ Voice I/O — STT bằng Whisper (Groq), TTS bằng gTTS + pygame
📚 RAG Pipeline — Tìm kiếm ngữ nghĩa trên SGK PDF + lời giải web + hình ảnh
🧮 Tool Calling — LLM gọi calculate để tính số chính xác (tránh hallucination), hỗ trợ hằng số vật lý THPT đầy đủ
🖼️ Vision — Chụp ảnh đề bài → OCR → giải tự động (Llama 4 Scout, có fallback model)
🔧 Text Correction — Sửa lỗi nhận dạng giọng nói thuật ngữ vật lý (regex + LLM)
🌐 Web Ingest — Crawl lời giải từ loigiaihay.com và các trang tương tự (Selenium + CLIP image embed)
💬 Conversation History — Bot nhớ ngữ cảnh suốt buổi học qua session_id
📊 Feedback & Logging — Ghi nhận đánh giá 1–5 sao và implicit feedback hành vi người dùng
🔐 Admin API — Ingest PDF/web từ xa qua Bearer token, không cần SSH

RASPBERRY PI  ──────────────────────────────────  SERVER (VPS/máy tính)
  app.py                                            main.py (FastAPI)
  ├── Mic → STT (Whisper/Groq)                      ├── RAG (ChromaDB)
  ├── text_correction (local)                        ├── LLM (Groq)
  ├── detect_mode()                                  ├── Tool calling
  ├── Camera + HC-SR04 (nếu OCR)                    └── Session history
  └── TTS (edge-tts) → Loa
         │  HTTP /ask /ocr /solve_image  │
         └──────────────────────────────┘
```
 
---
 
## SETUP SERVER (máy tính / VPS)
 
### 1. Clone repo
```bash
git clone https://github.com/your-username/physbot.git
cd physbot
pip install -r requirements.txt
```
 
### 2. Cấu hình .env
```bash
cp .env.example .env
# Điền GROQ_API_KEY, ADMIN_TOKEN, HF_TOKEN, HF_REPO_ID
```
 
### 3. Download ChromaDB từ Hugging Face
```bash
# Tự động khi chạy main.py, hoặc chạy thủ công:
python scripts/download_db.py
```
 
### 4. Chạy server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```
 
Kiểm tra: `curl http://localhost:8000/health`
 
---
 
## SETUP RASPBERRY PI
 
### 1. Clone repo (chỉ cần code, không cần data/)
```bash
git clone https://github.com/your-username/physbot.git
cd physbot
pip install -r requirements-pi.txt
```
 
### 2. Cấu hình .env trên Pi
```bash
cp .env.example .env
# Điền:
#   GROQ_API_KEY  (cho Whisper STT)
#   API_BASE_URL  (IP server, ví dụ: http://192.168.1.100:8000)
#   SESSION_ID    (ví dụ: pi-device-001)
```
 
### 3. Kết nối phần cứng HC-SR04
```
Pi GPIO:
  TRIG → GPIO 23 (pin 16)
  ECHO → GPIO 24 (pin 18)
  VCC  → 5V (pin 2)
  GND  → GND (pin 6)
```
 
### 4. Chạy
```bash
python app.py
```
 
---
 
## UPLOAD CHROMADB LÊN HUGGING FACE (1 lần)
 
```bash
# Trên máy dev có sẵn data/chroma_db/
pip install huggingface_hub
python scripts/upload_db.py --version v1.0
```
🗂️ Cấu trúc project
physbot/
├── app.py                  # Voice assistant (mic → STT → LLM → TTS)
├── main.py                 # FastAPI server v3.0
├── backend/
│   ├── rag_pipeline.py     # RAG: embed, retrieve, rerank, out-of-scope detection
│   ├── rag_classifier.py   # Phân loại dòng ngữ nghĩa vật lý (FORMULA/DEFINITION/...)
│   ├── calculator.py       # Safe eval + Groq tool schema + hằng số THPT
│   ├── text_correction.py  # Sửa lỗi STT (regex + LLM)
│   ├── web_ingest.py       # Crawl web → ChromaDB (text + CLIP image)
│   └── prompts.py          # System prompts
├── scripts/ 
│   ├── ingest.py               # Ingest PDF → ChromaDB (PyPDF2 / pdfplumber / OCR)
│   ├── ingest_images.py        # Ingest hình ảnh PDF → ChromaDB (Groq Vision)
│   ├── test_web.py             # Batch ingest ~53 bài lời giải từ loigiaihay.com
├── data/
│   ├── raw/                # PDF sách giáo khoa
│   ├── exercises/          # PDF bài tập
│   ├── processed/          # Cache text đã extract + checkpoint OCR
│   └── chroma_db/          # Vector database (ChromaDB persistent)
├── logs/
│   ├── physbot.jsonl       # Structured JSON logs
│   ├── feedback.jsonl      # Đánh giá câu trả lời từ người dùng
│   └── implicit_feedback.jsonl  # Log hành vi tự động (repeat, not_understand, ...)
├── requirements.txt
└── .env

⚙️ Cài đặt
1. Clone & tạo môi trường
bashgit clone https://github.com/your-username/physbot.git
cd physbot
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate
pip install -r requirements.txt
2. Cài phụ thuộc hệ thống
Tesseract OCR (chỉ cần nếu PDF không có text layer):

Windows: tải từ https://github.com/UB-Mannheim/tesseract/wiki → cài vào C:\Program Files\Tesseract-OCR\
Ubuntu: sudo apt install tesseract-ocr tesseract-ocr-vie

Poppler (chỉ cần nếu dùng OCR):

Windows: tải từ https://github.com/oschwartz10612/poppler-windows/releases
Ubuntu: sudo apt install poppler-utils

Chrome + ChromeDriver (cho web ingest):
bashpip install selenium webdriver-manager
3. Tạo file .env
envGROQ_API_KEY=your_groq_api_key_here
ADMIN_TOKEN=your-secret-admin-token
Lấy API key miễn phí tại https://console.groq.com

🚀 Sử dụng
Ingest dữ liệu (chạy 1 lần)
bash# Ingest PDF sách giáo khoa + bài tập
python ingest.py

# Ingest hình ảnh từ PDF bài tập (dùng Groq Vision)
python ingest_images.py

# Ingest ~53 bài lời giải web từ loigiaihay.com
python test_web.py
Chạy voice assistant (local)
bashpython app.py
Nhấn Enter để bắt đầu nói, bot tự dừng ghi sau 3 giây im lặng. Mỗi lần chạy tạo 1 session UUID mới — bot nhớ toàn bộ ngữ cảnh trong buổi học đó.
Chạy API server
bashuvicorn main:app --reload --port 8000
Endpoints
EndpointMethodGiới hạnMô tả/askPOST30/phútHỏi bằng text, có session history/ocrPOST20/phútĐọc nội dung ảnh (không giải)/solve_imagePOST5/phútOCR + giải bài từ ảnh/feedbackPOST—Đánh giá câu trả lời (1–5 sao)/admin/ingest/webPOSTBearerIngest URL lời giải mới/admin/ingest/pdfPOSTBearerTrigger ingest PDF mới/admin/sessionsGETBearerXem session đang active/healthGET—Kiểm tra trạng thái
Ví dụ /ask với session
jsonPOST /ask
{
  "question": "Tính vận tốc rơi tự do h=20m",
  "session_id": "hoc-sinh-001"
}
Bot sẽ nhớ ngữ cảnh nếu bạn tiếp tục gửi câu hỏi với cùng session_id. Session tự hết hạn sau 30 phút không dùng.

🧪 Test
bash# Test text correction (regex only, không cần API)
python backend/text_correction.py

# Test calculator + hằng số vật lý
python backend/calculator.py

# Test clean text PDF
python ingest.py --test

📦 Models sử dụng
ModelDùng chollama-3.1-8b-instantChat + tool callingmeta-llama/llama-4-scout-17b-16e-instructVision OCR (fallback: llama-4-maverick)whisper-large-v3Speech-to-Textparaphrase-multilingual-MiniLM-L12-v2Embedding text tiếng Việtopenai/clip-vit-base-patch32Embedding hình ảnh web

📝 Lưu ý

Đặt PDF sách giáo khoa vào data/raw/, bài tập vào data/exercises/
Nếu PDF bị lỗi encoding, bot tự thử pdfplumber → OCR Tesseract (có checkpoint từng trang)
ChromaDB lưu tại data/chroma_db/ — không commit folder này
Đường dẫn Tesseract/Poppler trong ingest.py mặc định cho Windows — sửa nếu dùng Linux
ADMIN_TOKEN trong .env bảo vệ các endpoint /admin/* — đổi giá trị mặc định trước khi deploy


📄 License
MIT