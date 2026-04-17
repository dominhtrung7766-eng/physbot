"""
app.py  —  PhysBot Pi Client
─────────────────────────────
Chạy trên Raspberry Pi theo đúng flow SVG:

  Mic → sounddevice (local)
    ↓
  STT Whisper → Groq API
    ↓
  text_correction (regex local)
    ↓
  detect mode (NORMAL / OCR / SOLVE_IMAGE)
    ↓
  Gọi API server (main.py) qua HTTP
  ├── /ask           : câu hỏi vật lý thường
  ├── /ocr           : đọc văn bản từ ảnh
  └── /solve_image   : giải bài từ ảnh
    ↓
  TTS edge-tts / gTTS → pygame → Loa

TÍNH NĂNG:
  [S1] Session ID cố định mỗi lần chạy — server nhớ ngữ cảnh suốt buổi học
  [S2] _last_numbers làm fallback khi history bị trim
  [S3] Implicit feedback tự động — log repeat / not_understand / rephrase_same / session_end
  [CAM] Camera + HC-SR04 — chỉ kích hoạt khi mode OCR / SOLVE_IMAGE

Cài đặt trên Pi:
    pip install -r requirements-pi.txt
"""

import time
import threading
import asyncio
import sys
import os
import re
import tempfile
import json
import uuid
from datetime import datetime
from pathlib import Path
from queue import Queue

import numpy as np
import sounddevice as sd
import httpx
from rich.console import Console
from dotenv import load_dotenv
from groq import Groq

from backend.text_correction import correct_physics_text, log_correction

load_dotenv()
console = Console()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── Cấu hình server ───────────────────────────────────────────────
API_BASE    = os.getenv("API_BASE_URL", "http://localhost:8000")
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "120"))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# ══════════════════════════════════════════════════════════════════
# [S1] SESSION — UUID cố định mỗi lần chạy app
# Gửi session_id lên server → server lưu history & nhớ ngữ cảnh
# Mỗi lần chạy app.py = 1 session mới
# ══════════════════════════════════════════════════════════════════

SESSION_ID         = os.getenv("SESSION_ID") or str(uuid.uuid4())[:8]
SESSION_START_TIME = time.time()

console.print(f"[dim]Session ID: {SESSION_ID}[/dim]")


# ══════════════════════════════════════════════════════════════════
# [S3] IMPLICIT FEEDBACK LOGGER
# Ghi tín hiệu hành vi tự động — không cần người dùng làm gì thêm
# ══════════════════════════════════════════════════════════════════

_IMPLICIT_LOG_PATH = Path("logs/implicit_feedback.jsonl")


def _log_implicit(event: str, **kwargs):
    """
    Ghi 1 sự kiện implicit feedback ra file JSONL.

    Events:
      repeat_request  — người dùng nói "nói lại" → câu trước chưa rõ
      not_understand  — người dùng nói "không hiểu"
      rephrase_same   — hỏi lại cùng chủ đề trong session
      out_of_scope    — server từ chối (câu hỏi ngoài vật lý THPT)
      api_error       — server trả lỗi / timeout
      session_end     — tổng lượt + thời gian buổi học
    """
    record = {
        "ts":         datetime.utcnow().isoformat() + "Z",
        "session_id": SESSION_ID,
        "event":      event,
        **kwargs,
    }
    try:
        _IMPLICIT_LOG_PATH.parent.mkdir(exist_ok=True)
        with open(_IMPLICIT_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        console.print(f"[dim red]Implicit log lỗi: {e}[/dim red]")


# ══════════════════════════════════════════════════════════════════
# STATE
# ══════════════════════════════════════════════════════════════════

_last_response: str = ""
_last_numbers:  str = ""   # [S2] fallback khi server history bị trim
_last_topic:    str = ""   # [S3] detect hỏi lại cùng chủ đề

_REPEAT_TRIGGERS = [
    "nói lại", "đọc lại", "nhắc lại", "lặp lại",
    "không nghe", "nghe không rõ", "cho nghe lại",
    "lại đi", "lại nha", "lại một lần", "nói lại cho",
    "đọc lại cho", "không nghe rõ", "nghe không thấy",
]

_NOT_UNDERSTAND_TRIGGERS = [
    "không hiểu", "chưa hiểu",
    "giải thích lại", "nói dễ hơn", "đơn giản hơn",
    "khó hiểu", "không rõ", "chưa rõ",
]

_KEEP_DATA_TRIGGERS = [
    "giữ nguyên số liệu", "giữ nguyên dữ liệu", "cùng số liệu",
    "số liệu cũ", "số liệu trên", "bài trên", "đề trên",
    "vẫn vậy", "như cũ", "cũng vậy", "vẫn số đó",
    "giữ nguyên", "tương tự vậy", "tương tự đó",
    "cùng bài", "cùng đề", "đề cũ", "bài cũ",
]


def _is_repeat_request(text: str) -> bool:
    return any(t in text.lower() for t in _REPEAT_TRIGGERS)

def _is_not_understand(text: str) -> bool:
    return any(t in text.lower() for t in _NOT_UNDERSTAND_TRIGGERS)

def _is_keep_data_request(text: str) -> bool:
    return any(t in text.lower() for t in _KEEP_DATA_TRIGGERS)

def _extract_topic(text: str) -> str:
    """[S3] 3 từ đầu có nghĩa làm topic fingerprint để detect hỏi lại."""
    words = re.findall(r'\b\w{3,}\b', text.lower())
    return " ".join(words[:3]) if words else ""

_NUMBER_PATTERN = re.compile(
    r'\d+(?:[,\.]\d+)?'
    r'(?:\s*(?:kilôgam|kilômét|mét|xentimét|milimét|giây|niutơn|jun|oát|'
    r'vôn|ampe|héc|culông|tesla|fara|henry|pascal|ôm|'
    r'kg|km|cm|mm|m|s|N|J|W|V|A|Hz|C|T|F|H|Pa)[\w\s]*)?',
    re.IGNORECASE
)

def _extract_numbers(text: str) -> str:
    matches = _NUMBER_PATTERN.findall(text)
    cleaned = [m.strip() for m in matches if m.strip() and re.search(r'\d', m)]
    return ", ".join(cleaned)


# ══════════════════════════════════════════════════════════════════
# DETECT MODE — phân loại lệnh voice
# ══════════════════════════════════════════════════════════════════

_OCR_TRIGGERS = [
    "đọc văn bản", "đọc sách", "đọc chữ",
    "đọc cho tao", "đọc trang này", "chức năng đọc",
    "đọc hộ", "đọc giúp",
]

_SOLVE_IMAGE_TRIGGERS = [
    "giải bài", "bài có hình", "hình vẽ",
    "xem hình", "chụp bài", "giải hình",
    "bài tập này", "đề này",
]


def detect_mode(text: str) -> str:
    """
    NORMAL      → /ask
    OCR         → camera + /ocr
    SOLVE_IMAGE → camera + /solve_image
    """
    t = text.lower()
    if any(kw in t for kw in _OCR_TRIGGERS):
        return "OCR"
    if any(kw in t for kw in _SOLVE_IMAGE_TRIGGERS):
        return "SOLVE_IMAGE"
    return "NORMAL"


# ══════════════════════════════════════════════════════════════════
# CAMERA + HC-SR04 — chỉ dùng khi mode OCR / SOLVE_IMAGE
# ══════════════════════════════════════════════════════════════════

def get_distance_cm() -> float:
    """Đọc HC-SR04 qua GPIO. Trả về cm. Fallback 25.0 nếu không có Pi."""
    try:
        import RPi.GPIO as GPIO

        TRIG = int(os.getenv("HCSR04_TRIG", "23"))
        ECHO = int(os.getenv("HCSR04_ECHO", "24"))

        GPIO.setmode(GPIO.BCM)
        GPIO.setup(TRIG, GPIO.OUT)
        GPIO.setup(ECHO, GPIO.IN)
        GPIO.output(TRIG, False)
        time.sleep(0.05)
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)

        timeout = time.time() + 1.0
        while GPIO.input(ECHO) == 0:
            start = time.time()
            if time.time() > timeout:
                return -1.0
        timeout = time.time() + 1.0
        while GPIO.input(ECHO) == 1:
            end = time.time()
            if time.time() > timeout:
                return -1.0

        GPIO.cleanup()
        return round((end - start) * 17150, 1)

    except Exception:
        return 25.0   # dev mode: giả lập đúng tầm


def capture_image_bytes() -> bytes | None:
    """Chụp ảnh từ webcam / Pi Camera. Trả về JPEG bytes."""
    try:
        import cv2

        cam_index = int(os.getenv("CAMERA_INDEX", "0"))
        cap = cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            console.print("[red]Không mở được camera!")
            return None
        time.sleep(0.5)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return None
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return buf.tobytes()

    except ImportError:
        console.print("[yellow]OpenCV chưa cài. Chạy: pip install opencv-python")
        return None
    except Exception as e:
        console.print(f"[red]Camera lỗi: {e}")
        return None


def _check_image_quality(img_bytes: bytes) -> bool:
    """Laplacian blur detection — True = đủ rõ."""
    try:
        import cv2
        nparr = np.frombuffer(img_bytes, np.uint8)
        img   = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        score = cv2.Laplacian(img, cv2.CV_64F).var()
        min_score = float(os.getenv("BLUR_MIN_SCORE", "80"))
        console.print(f"[dim]Blur score: {score:.1f} (min={min_score})[/dim]")
        return score >= min_score
    except Exception:
        return True


def guide_and_capture() -> bytes | None:
    """
    Hướng dẫn căn chỉnh bằng giọng nói → chụp ảnh.
    Flow: HC-SR04 đo khoảng cách → voice guide → OpenCV check → trả bytes.
    """
    DIST_MIN     = float(os.getenv("DIST_MIN_CM", "10"))
    DIST_MAX     = float(os.getenv("DIST_MAX_CM", "40"))
    MAX_ATTEMPTS = 5

    for attempt in range(MAX_ATTEMPTS):
        dist = get_distance_cm()
        console.print(f"[dim]Khoảng cách: {dist} cm[/dim]")

        if dist < 0:
            asyncio.run(speak("Cảm biến lỗi, tui chụp luôn nhé."))
            return capture_image_bytes()

        if dist > DIST_MAX:
            diff = int(dist - (DIST_MIN + DIST_MAX) / 2)
            asyncio.run(speak(
                f"Đưa sách lại gần hơn khoảng {diff} xăng-ti-mét nữa nhé."
            ))
            time.sleep(2)
            continue

        if dist < DIST_MIN:
            diff = int((DIST_MIN + DIST_MAX) / 2 - dist)
            asyncio.run(speak(
                f"Lùi sách ra xa hơn khoảng {diff} xăng-ti-mét nhé."
            ))
            time.sleep(2)
            continue

        # Đúng tầm → chụp
        asyncio.run(speak("Ổn rồi, đang chụp..."))
        img_bytes = capture_image_bytes()
        if img_bytes is None:
            continue

        if not _check_image_quality(img_bytes):
            asyncio.run(speak("Ảnh hơi mờ, giữ yên tay và thử lại nhé."))
            time.sleep(1.5)
            continue

        console.print("[green]Ảnh đạt chất lượng!")
        return img_bytes

    asyncio.run(speak("Thử nhiều lần rồi, tui gửi ảnh hiện tại lên nhé."))
    return capture_image_bytes()


# ══════════════════════════════════════════════════════════════════
# GỌI API SERVER
# [S1] Truyền session_id → server lưu history & nhớ ngữ cảnh
# ══════════════════════════════════════════════════════════════════

def call_api_ask(text: str) -> str:
    """POST /ask — câu hỏi vật lý thường."""
    try:
        with httpx.Client(timeout=API_TIMEOUT) as client:
            r = client.post(
                f"{API_BASE}/ask",
                json={"question": text, "session_id": SESSION_ID},
            )
            r.raise_for_status()
            answer = r.json().get("answer", "")

            # [S3] Detect out_of_scope từ response server
            if "bó tay" in answer.lower() or "ngoài phạm vi" in answer.lower():
                _log_implicit("out_of_scope", question=text[:80])

            return answer
    except httpx.ConnectError:
        _log_implicit("api_error", error="ConnectError", endpoint="/ask")
        return "Tui không kết nối được server, bạn kiểm tra wifi nha!"
    except httpx.TimeoutException:
        _log_implicit("api_error", error="Timeout", endpoint="/ask")
        return "Server trả lời quá lâu, bạn thử lại nha!"
    except Exception as e:
        _log_implicit("api_error", error=str(e)[:80], endpoint="/ask")
        console.print(f"[red]API /ask lỗi: {e}")
        return "Có lỗi xảy ra, bạn thử lại sau nha!"


def call_api_ocr(img_bytes: bytes) -> str:
    """POST /ocr — đọc văn bản từ ảnh."""
    try:
        with httpx.Client(timeout=API_TIMEOUT) as client:
            r = client.post(
                f"{API_BASE}/ocr",
                files={"file": ("photo.jpg", img_bytes, "image/jpeg")},
            )
            r.raise_for_status()
            return r.json().get("answer", "")
    except httpx.ConnectError:
        return "Tui không kết nối được server, bạn kiểm tra wifi nha!"
    except httpx.TimeoutException:
        return "Server trả lời quá lâu, bạn thử lại nha!"
    except Exception as e:
        console.print(f"[red]API /ocr lỗi: {e}")
        return "Có lỗi xảy ra, bạn thử lại sau nha!"


def call_api_solve_image(img_bytes: bytes) -> str:
    """POST /solve_image — giải bài từ ảnh."""
    try:
        with httpx.Client(timeout=API_TIMEOUT) as client:
            r = client.post(
                f"{API_BASE}/solve_image",
                files={"file": ("photo.jpg", img_bytes, "image/jpeg")},
                params={"session_id": SESSION_ID},
            )
            r.raise_for_status()
            return r.json().get("answer", "")
    except httpx.ConnectError:
        return "Tui không kết nối được server, bạn kiểm tra wifi nha!"
    except httpx.TimeoutException:
        return "Server trả lời quá lâu, bạn thử lại nha!"
    except Exception as e:
        console.print(f"[red]API /solve_image lỗi: {e}")
        return "Có lỗi xảy ra, bạn thử lại sau nha!"


def get_response(text: str) -> str:
    """
    Entry point — detect mode → gọi đúng API.
    [S1] session_id truyền lên server → server nhớ history.
    """
    mode = detect_mode(text)
    console.print(f"[dim]Mode: {mode}[/dim]")

    if mode == "NORMAL":
        return call_api_ask(text)

    # OCR / SOLVE_IMAGE → cần camera
    asyncio.run(speak("Oke, tui chuẩn bị chụp ảnh nhé."))
    img_bytes = guide_and_capture()

    if img_bytes is None:
        return "Tui không chụp được ảnh, bạn thử lại nhé."

    if mode == "OCR":
        return call_api_ocr(img_bytes)
    else:
        return call_api_solve_image(img_bytes)


# ══════════════════════════════════════════════════════════════════
# RECORD AUDIO
# ══════════════════════════════════════════════════════════════════

def record_audio(stop_event, data_queue, silence_threshold=0.01, silence_duration=3.0):
    sample_rate    = 16000
    chunk_duration = 0.5
    silent_chunks  = 0
    started        = False

    def callback(indata, frames, time_info, status):
        nonlocal silent_chunks, started
        if status:
            console.print(f"[dim]{status}[/dim]")
        data_queue.put(bytes(indata))
        audio_np = (
            np.frombuffer(bytes(indata), dtype=np.int16)
            .astype(np.float32) / 32768.0
        )
        energy = np.abs(audio_np).mean()
        if not started:
            if energy > silence_threshold:
                started = True
                console.print("[dim]Đã phát hiện giọng nói...[/dim]")
            return
        if energy < silence_threshold:
            silent_chunks += 1
        else:
            silent_chunks = 0
        if silent_chunks >= silence_duration / chunk_duration:
            console.print(f"[dim]Im lặng {silence_duration}s, dừng ghi...[/dim]")
            stop_event.set()

    while not data_queue.empty():
        data_queue.get()

    with sd.RawInputStream(
        samplerate=sample_rate, dtype="int16", channels=1,
        callback=callback, blocksize=int(sample_rate * chunk_duration),
    ):
        console.print(f"[dim]Đang nghe... (tự dừng sau {silence_duration}s im lặng)[/dim]")
        while not stop_event.is_set():
            time.sleep(0.1)


# ══════════════════════════════════════════════════════════════════
# TRANSCRIBE — Whisper qua Groq
# ══════════════════════════════════════════════════════════════════

def transcribe(audio_np: np.ndarray) -> str:
    try:
        import soundfile as sf

        energy = np.abs(audio_np).mean()
        if energy < 0.05:
            gain     = min(0.05 / (energy + 1e-9), 10.0)
            audio_np = np.clip(audio_np * gain, -1.0, 1.0)
            console.print(f"[dim]Khuếch đại x{gain:.1f}[/dim]")

        if len(audio_np) / 16000 < 0.8:
            return ""

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            sf.write(tmp_path, audio_np.astype(np.float32), 16000)
            with open(tmp_path, "rb") as af:
                result = groq_client.audio.transcriptions.create(
                    model="whisper-large-v3", file=af, language="vi",
                )
        os.unlink(tmp_path)
        return result.text.strip()
    except Exception as e:
        console.print(f"[red]STT lỗi: {e}")
        return ""


# ══════════════════════════════════════════════════════════════════
# TTS — edge-tts (fallback gTTS)
# ══════════════════════════════════════════════════════════════════

def sanitize_for_tts(text: str) -> str:
    """Chuyển ký hiệu toán học / đơn vị / markdown → tiếng Việt đọc được."""
    # Strip markdown
    text = re.sub(r'\*{1,3}([^*\n]+)\*{1,3}', r'\1', text)
    text = re.sub(r'(?m)^#{1,6}\s*', '', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'(?m)^\s*[-•]\s+', '', text)

    # Đơn vị ghép
    unit_compounds = [
        (r'\bm/s²\b', 'mét trên giây bình phương'),
        (r'\bm/s2\b', 'mét trên giây bình phương'),
        (r'\bkg/m³\b', 'kilôgam trên mét khối'),
        (r'\bkg/m3\b', 'kilôgam trên mét khối'),
        (r'\bN/m²\b', 'niutơn trên mét vuông'),
        (r'\bN/m2\b', 'niutơn trên mét vuông'),
        (r'\bW/m²\b', 'oát trên mét vuông'),
        (r'\bJ/kg\b', 'jun trên kilôgam'),
        (r'\bJ/mol\b', 'jun trên mol'),
        (r'\bN/C\b', 'niutơn trên culông'),
        (r'\bV/m\b', 'vôn trên mét'),
        (r'\bA/m\b', 'ampe trên mét'),
        (r'\bΩ\.m\b', 'ôm nhân mét'),
        (r'\bm/s\b', 'mét trên giây'),
        (r'\bcm/s\b', 'xentimét trên giây'),
        (r'\bkm/h\b', 'kilômét trên giờ'),
        (r'\brad/s\b', 'radian trên giây'),
        (r'\brad/s²\b', 'radian trên giây bình phương'),
        (r'\bN/m\b', 'niutơn trên mét'),
        (r'\bμF\b', 'micrô fara'), (r'\bμH\b', 'micrô henry'),
        (r'\bμC\b', 'micrô culông'), (r'\bμA\b', 'micrô ampe'),
        (r'\bGHz\b', 'giga héc'), (r'\bMHz\b', 'mêga héc'),
        (r'\bkHz\b', 'kilô héc'), (r'\bMΩ\b', 'mêga ôm'),
        (r'\bkΩ\b', 'kilô ôm'), (r'\bkW\b', 'kilô oát'),
        (r'\bkV\b', 'kilô vôn'), (r'\bkJ\b', 'kilô jun'),
        (r'\bkm\b', 'kilômét'), (r'\bnF\b', 'nano fara'),
        (r'\bnC\b', 'nano culông'), (r'\bnm\b', 'nano mét'),
        (r'\bpF\b', 'picô fara'), (r'\beV\b', 'êlectrôn vôn'),
        (r'\bMeV\b', 'mêga êlectrôn vôn'),
        (r'(\d)\s*mH\b', r'\1 mili henry'), (r'(\d)\s*mA\b', r'\1 mili ampe'),
        (r'(\d)\s*mV\b', r'\1 mili vôn'), (r'(\d)\s*ms\b', r'\1 mili giây'),
        (r'(\d)\s*mm\b', r'\1 milimét'),
        (r'\bm²\b', 'mét vuông'), (r'\bm2\b', 'mét vuông'),
        (r'\bcm²\b', 'xentimét vuông'), (r'\bcm2\b', 'xentimét vuông'),
        (r'\bm³\b', 'mét khối'), (r'\bm3\b', 'mét khối'),
        (r'\bcm³\b', 'xentimét khối'), (r'\bcm3\b', 'xentimét khối'),
        (r'\bs²\b', 'giây bình phương'), (r'\bs2\b', 'giây bình phương'),
        (r'\bHz\b', 'héc'), (r'\bPa\b', 'pascal'), (r'\batm\b', 'atmôtphe'),
        (r'\bWb\b', 'vêbe'), (r'\brad\b', 'radian'), (r'\bmol\b', 'mol'),
        (r'\bkg\b', 'kilôgam'), (r'\bcm\b', 'xentimét'),
        (r'\bmm\b', 'milimét'), (r'\bmin\b', 'phút'),
    ]
    for pattern, repl in unit_compounds:
        text = re.sub(pattern, repl, text)

    # Đơn vị 1 ký tự sau chữ số
    for sym, name in [('N','niutơn'),('J','jun'),('W','oát'),('V','vôn'),
                      ('A','ampe'),('F','fara'),('H','henry'),('T','tesla'),
                      ('C','culông'),('K','ken-vin')]:
        text = re.sub(rf'(\d)\s*{re.escape(sym)}\b', rf'\1 {name}', text)
    text = re.sub(r'(\d)\s*Ω', r'\1 ôm', text)
    text = re.sub(r'\bΩ\b', 'ôm', text)

    # Lũy thừa ^
    for pattern, repl in [
        (r'10\^-34','mười mũ trừ ba mươi bốn'),(r'10\^-31','mười mũ trừ ba mươi mốt'),
        (r'10\^-27','mười mũ trừ hai mươi bảy'),(r'10\^-23','mười mũ trừ hai mươi ba'),
        (r'10\^-19','mười mũ trừ mười chín'),(r'10\^-15','mười mũ trừ mười lăm'),
        (r'10\^-12','mười mũ trừ mười hai'),(r'10\^-9','mười mũ trừ chín'),
        (r'10\^-6','mười mũ trừ sáu'),(r'10\^-3','mười mũ trừ ba'),
        (r'10\^9','mười mũ chín'),(r'10\^8','mười mũ tám'),
        (r'10\^6','mười mũ sáu'),(r'10\^3','mười mũ ba'),(r'10\^2','mười mũ hai'),
        (r'\^2\b',' bình phương'),(r'\^3\b',' lập phương'),
        (r'\^-1\b',' mũ trừ một'),(r'\^-2\b',' mũ trừ hai'),(r'\^-3\b',' mũ trừ ba'),
    ]:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    # Subscript / Superscript Unicode
    for old, new in [('₀',' không'),('₁',' một'),('₂',' hai'),('₃',' ba'),
                     ('₄',' bốn'),('₅',' năm'),('₆',' sáu'),('₇',' bảy'),
                     ('₈',' tám'),('₉',' chín')]:
        text = text.replace(old, new)
    for old, new in [('²',' bình phương'),('³',' lập phương'),('⁻',' trừ '),
                     ('⁰',' mũ không'),('⁴',' mũ bốn'),('⁵',' mũ năm'),
                     ('⁶',' mũ sáu'),('⁷',' mũ bảy'),('⁸',' mũ tám'),
                     ('⁹',' mũ chín'),('ⁿ',' mũ n'),('¹',' mũ một')]:
        text = text.replace(old, new)

    # Căn bậc
    text = re.sub(r'√\(([^)]+)\)', r'căn bậc hai của \1', text)
    text = text.replace('√', 'căn bậc hai của ')

    # Phân số Unicode
    for old, new in [('½',' một phần hai '),('¼',' một phần bốn '),
                     ('¾',' ba phần bốn '),('⅓',' một phần ba '),('⅔',' hai phần ba ')]:
        text = text.replace(old, new)

    # Ký hiệu Hy Lạp
    for old, new in [
        ('α','anpha'),('β','bê-ta'),('γ','gama'),('ω','ô-mê-ga'),
        ('λ','lăm-đa'),('θ','tê-ta'),('π','pi'),('Δ','delta '),('δ','delta '),
        ('Φ','phi '),('φ','phi '),('Σ','tổng '),('σ','sigma '),('μ','muy'),
        ('η','êta'),('ρ','rô'),('ε','êp-xi-lông'),('τ','tô'),('ξ','xi'),
        ('Λ','lăm-đa'),('Ω','ô-mê-ga hoa'),
    ]:
        text = text.replace(old, new)

    # Space giữa chữ-số dính
    text = re.sub(r'([a-zA-ZÀ-ỹ])(\d)', r'\1 \2', text)

    # Toán tử
    text = re.sub(r'\s+x\s+', ' nhân ', text)
    for old, new in [('×',' nhân '),('÷',' chia '),('≈',' xấp xỉ '),
                     ('≠',' khác '),('≥',' lớn hơn hoặc bằng '),
                     ('≤',' nhỏ hơn hoặc bằng '),('>',' lớn hơn '),('<',' nhỏ hơn '),
                     ('→',' suy ra '),('⇒',' suy ra '),('∞','vô cực'),
                     ('%',' phần trăm'),('°',' độ'),('=',' bằng ')]:
        text = text.replace(old, new)
    text = text.replace('+', ' cộng ')

    # Ký tự định dạng
    for old, new in [('—',', '),('–',', '),('−',' trừ '),
                     ('━',''),('═',''),('│',''),('┃',''),
                     ('❌',''),('✅',''),('✔',''),('✗',''),('_',' ')]:
        text = text.replace(old, new)

    text = re.sub(r'\n+', '. ', text)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r',{2,}', ',', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\s([.,;:])', r'\1', text)
    text = re.sub(r'([.,;:]){2,}', r'\1', text)
    return text.strip()


async def speak(text: str):
    text = sanitize_for_tts(text).strip()
    if not text:
        return
    if text[-1] not in '.!?':
        text += '.'
    console.print(f"[magenta]TTS ({len(text)} ký tự): {repr(text[:120])}[/magenta]")

    # Thử edge-tts trước (chất lượng tốt hơn gTTS)
    try:
        import edge_tts
        import pygame

        voice = os.getenv("TTS_VOICE", "vi-VN-HoaiMyNeural")
        communicate = edge_tts.Communicate(text, voice)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name
        await communicate.save(tmp_path)
        pygame.mixer.init()
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
        time.sleep(0.3)
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        os.unlink(tmp_path)
        return
    except ImportError:
        pass
    except Exception as e:
        console.print(f"[yellow]edge-tts lỗi: {e}, fallback gTTS")

    # Fallback: gTTS
    try:
        from gtts import gTTS
        import pygame

        tts = gTTS(text=text, lang='vi', slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name
        tts.save(tmp_path)
        pygame.mixer.init()
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(100)
        time.sleep(0.3)
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        os.unlink(tmp_path)
    except Exception as e:
        console.print(f"[red]TTS lỗi hoàn toàn: {e}")


# ══════════════════════════════════════════════════════════════════
# MAIN LOOP
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    console.print("[cyan]PhysBot Pi Client sẵn sàng!")
    console.print(f"[cyan]Server  : {API_BASE}")
    console.print(f"[cyan]Session : {SESSION_ID}")
    console.print("[cyan]━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    console.print("[cyan]Nhấn Enter để bắt đầu nói, tự dừng sau 3 giây im lặng.")
    console.print("[cyan]Ctrl+C để thoát.\n")

    # Kiểm tra kết nối server
    try:
        with httpx.Client(timeout=5) as client:
            r    = client.get(f"{API_BASE}/health")
            info = r.json()
            console.print(
                f"[green]Server OK — model: {info.get('model','?')} | "
                f"chromadb: {info.get('chromadb','?')}"
            )
    except Exception:
        console.print(f"[red]Cảnh báo: Không kết nối được server tại {API_BASE}")

    turn_count = 0

    try:
        while True:
            console.input("\nNhấn Enter để bắt đầu ghi âm...")

            stop_event = threading.Event()
            data_queue = Queue()
            recording_thread = threading.Thread(
                target=record_audio,
                args=(stop_event, data_queue, 0.01, 3.0),
            )
            recording_thread.start()
            recording_thread.join()

            chunks = []
            while not data_queue.empty():
                chunks.append(data_queue.get())
            audio_data = b"".join(chunks)

            audio_np = (
                np.frombuffer(audio_data, dtype=np.int16)
                .astype(np.float32) / 32768.0
            )

            if audio_np.size == 0:
                console.print("[red]Không nghe thấy gì. Kiểm tra lại mic.")
                continue

            # ── STT ──────────────────────────────────────────────
            with console.status("Đang nhận dạng giọng nói...", spinner="dots"):
                t0       = time.time()
                raw_text = transcribe(audio_np)
                t1       = time.time()

            # ── Text correction (regex local, không cần Groq) ─────
            text = correct_physics_text(raw_text, use_llm=False)
            log_correction(raw_text, text)

            console.print(f"[yellow]Bạn (raw) : {raw_text}")
            if text != raw_text:
                console.print(f"[yellow]Bạn (fixed): {text}")
            console.print(f"[dim]STT: {t1-t0:.2f}s[/dim]")

            if not text.strip():
                console.print("[red]Không nhận ra giọng nói, thử lại nhé!")
                continue

            # ── [S3] Detect "nói lại" ─────────────────────────────
            if _is_repeat_request(text) and _last_response:
                _log_implicit("repeat_request",
                               question=text[:80],
                               last_response_len=len(_last_response))
                console.print("[yellow]→ Phát lại response trước[/yellow]")
                asyncio.run(speak(_last_response))
                continue

            # ── [S3] Detect "không hiểu" — log, vẫn gọi API ──────
            if _is_not_understand(text) and _last_response:
                _log_implicit("not_understand",
                               question=text[:80],
                               last_response_preview=_last_response[:80])
                console.print("[dim yellow]Implicit: chưa hiểu → gọi API giải thích lại[/dim yellow]")

            # ── [S3] Detect hỏi lại cùng chủ đề ─────────────────
            current_topic = _extract_topic(text)
            if _last_topic and current_topic and _last_topic == current_topic:
                _log_implicit("rephrase_same",
                               topic=current_topic,
                               question=text[:80])
                console.print(f"[dim yellow]Implicit: hỏi lại chủ đề '{current_topic}'[/dim yellow]")
            _last_topic = current_topic

            # ── [S2] Keep data fallback ───────────────────────────
            # Server đã có session history → ít cần, nhưng giữ làm safety net
            if _is_keep_data_request(text) and _last_numbers:
                text = f"{text}. Số liệu đã cho từ bài trước: {_last_numbers}"
                console.print(f"[dim]→ Inject số liệu cũ (fallback): {_last_numbers}[/dim]")

            extracted = _extract_numbers(text)
            if extracted:
                _last_numbers = extracted
                console.print(f"[dim]→ Lưu số liệu: {_last_numbers}[/dim]")

            # ── Gọi API server ────────────────────────────────────
            with console.status("Đang xử lý...", spinner="dots"):
                t2       = time.time()
                response = get_response(text)
                t3       = time.time()

            _last_response = response
            turn_count += 1

            # Cắt nếu quá dài
            if len(response) > 2000:
                chunk = response[:2000]
                cut   = max(chunk.rfind('.'), chunk.rfind('!'), chunk.rfind('?'))
                if cut == -1:
                    cut = 2000
                response = response[:cut+1] + " (tui rút gọn để đọc nhanh hơn nha)"

            console.print(f"[cyan]PhysBot: {response}")
            console.print(f"[dim]API: {t3-t2:.2f}s | Turn #{turn_count}[/dim]")

            # ── TTS ───────────────────────────────────────────────
            t4 = time.time()
            asyncio.run(speak(response))
            t5 = time.time()

            console.print(f"[dim]TTS: {t5-t4:.2f}s | Tổng: {t5-t0:.2f}s[/dim]")

    except KeyboardInterrupt:
        # ── [S3] Log session_end ──────────────────────────────────
        duration_min = round((time.time() - SESSION_START_TIME) / 60, 1)
        _log_implicit("session_end",
                       total_turns=turn_count,
                       duration_minutes=duration_min)
        console.print(
            f"\n[dim]Session {SESSION_ID}: {turn_count} lượt, "
            f"{duration_min} phút → đã log[/dim]"
        )
        console.print("[red]Thoát...")
