"""
app.py  —  PhysBot Pi Client v2.9.5
──────────────────────────────────
THAY ĐỔI v2.9.5:
  [FIX] Stream chạy liên tục, không stop/start khi TTS
        → triệt để ALSA underrun
        → dùng _tts_playing flag để drop data trong callback
"""

import os
os.environ["PA_ALSA_PLUGHW"]    = "1"
os.environ["ORT_LOGGING_LEVEL"] = "3"
os.environ["SDL_AUDIODRIVER"]   = "alsa"
os.environ["AUDIODEV"]          = "default"

import time
import threading
import asyncio
import sys
import re
import tempfile
import json
import uuid
import math
import warnings
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue, Empty

warnings.filterwarnings("ignore", message="Specified provider 'CUDAExecutionProvider'")

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

API_BASE    = os.getenv("API_BASE_URL", "http://localhost:8000")
API_TIMEOUT = float(os.getenv("API_TIMEOUT", "45"))

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


# ══════════════════════════════════════════════════════════════════
# AUDIO CONFIG
# ══════════════════════════════════════════════════════════════════

_dev_raw     = os.getenv("INPUT_DEVICE", "0")
INPUT_DEVICE = int(_dev_raw) if _dev_raw.lstrip("-").isdigit() else _dev_raw

HW_SR       = int(os.getenv("HW_SAMPLERATE", "48000"))
TARGET_SR   = int(os.getenv("TARGET_SR",      "16000"))
HW_CHANNELS = int(os.getenv("HW_CHANNELS",   "2"))

UNIFIED_CHUNK_MS     = 80
UNIFIED_BLOCK_FRAMES = int(HW_SR * UNIFIED_CHUNK_MS / 1000)

STREAM_WARMUP_SEC = float(os.getenv("STREAM_WARMUP_SEC", "3.0"))


def _auto_detect_device() -> int | str:
    try:
        devices = sd.query_devices()
        if not devices:
            console.print("[yellow]⚠ PortAudio không thấy device nào[/yellow]")
            return INPUT_DEVICE

        console.print("[dim]─── Danh sách audio devices ───[/dim]")
        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                console.print(f"[dim]  [{i}] {d['name']} "
                               f"(in={d['max_input_channels']} "
                               f"sr={int(d['default_samplerate'])})[/dim]")

        keywords = ["googlevoice", "voicehat", "google voice", "sndrpigoogle", "spdif"]
        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                if any(kw in d["name"].lower() for kw in keywords):
                    console.print(f"[green]✓ Tìm thấy Google Voice HAT: [{i}] {d['name']}[/green]")
                    return i

        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) > 0:
                console.print(f"[yellow]⚠ Dùng device [{i}] {d['name']}[/yellow]")
                return i

    except Exception as e:
        console.print(f"[yellow]⚠ Auto-detect lỗi: {e}[/yellow]")

    return INPUT_DEVICE


def _resample(audio_hw: np.ndarray) -> np.ndarray:
    from scipy.signal import resample_poly
    if audio_hw.dtype == np.int32:
        audio_f = audio_hw.astype(np.float32) / 2147483648.0
    else:
        audio_f = audio_hw.astype(np.float32)
    if audio_f.ndim == 2:
        mono = audio_f.mean(axis=1)
    else:
        mono = audio_f.copy()
    g    = math.gcd(TARGET_SR, HW_SR)
    up   = TARGET_SR // g
    down = HW_SR     // g
    return resample_poly(mono, up, down).astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# WAKE WORD CONFIG
# ══════════════════════════════════════════════════════════════════

WAKE_MODEL        = os.getenv("WAKE_MODEL", "hey_jarvis")
WAKE_THRESHOLD    = float(os.getenv("WAKE_THRESHOLD", "0.95"))
ENERGY_MIN        = float(os.getenv("WAKE_ENERGY_MIN", "0.02"))
WAKE_COOLDOWN_SEC = 3.0
POST_TTS_MUTE_SEC = float(os.getenv("POST_TTS_MUTE_SEC", "3.0"))
OWW_CHUNK_SAMPLES = int(TARGET_SR * UNIFIED_CHUNK_MS / 1000)


# ══════════════════════════════════════════════════════════════════
# STATE MACHINE
# ══════════════════════════════════════════════════════════════════

STATE_IDLE   = "IDLE"
STATE_ACTIVE = "ACTIVE"

_bot_state       = STATE_IDLE
_state_lock      = threading.Lock()
_last_active_ts  = 0.0
IDLE_TIMEOUT_SEC = float(os.getenv("IDLE_TIMEOUT_SEC", "30"))
_activated_event = threading.Event()


# ══════════════════════════════════════════════════════════════════
# UNIFIED STREAM
# ══════════════════════════════════════════════════════════════════

_record_queue: Queue       = Queue()
_wake_last_trigger: float  = 0.0
_tts_playing: bool         = False
_last_deactivate_ts: float = 0.0
WAKE_POST_DEACTIVATE_COOLDOWN = float(os.getenv("WAKE_POST_DEACTIVATE_COOLDOWN", "5.0"))

_stream_start_ts: float  = 0.0
_stream_warmed_up: bool  = False

_oww_builtin_model = None
_oww_custom_sess   = None
_oww_custom_iname  = None
_oww_audio_feats   = None
_use_custom_model  = False

_custom_audio_buf: np.ndarray = np.zeros(0, dtype=np.int16)


def _unified_audio_callback(indata: np.ndarray, frames: int, time_info, status):
    global _wake_last_trigger, _custom_audio_buf, _stream_warmed_up

    # DROP data khi TTS đang phát — không cần stop stream
    if _tts_playing:
        return

    if not _stream_warmed_up:
        if time.time() - _stream_start_ts < STREAM_WARMUP_SEC:
            return
        _custom_audio_buf = np.zeros(0, dtype=np.int16)
        _stream_warmed_up = True
        console.print(f"[dim]✓ Warmup xong — bắt đầu lắng nghe wake word[/dim]")

    audio_16k = _resample(indata)

    if _bot_state == STATE_IDLE:
        score = 0.0

        if _use_custom_model and _oww_custom_sess is not None:
            audio_int16 = (audio_16k * 32767).astype(np.int16)
            _custom_audio_buf = np.concatenate([_custom_audio_buf, audio_int16])
            while len(_custom_audio_buf) >= OWW_CHUNK_SAMPLES:
                chunk = _custom_audio_buf[:OWW_CHUNK_SAMPLES]
                _custom_audio_buf = _custom_audio_buf[OWW_CHUNK_SAMPLES:]
                try:
                    _oww_audio_feats(chunk)
                    emb = _oww_audio_feats.get_features(n_feature_frames=1)
                    if emb is not None and len(emb) > 0:
                        raw = _oww_custom_sess.run(
                            None,
                            {_oww_custom_iname: emb[0].reshape(1, -1).astype(np.float32)}
                        )[0][0][0]
                        score = max(score, float(raw))
                except Exception:
                    pass

        elif _oww_builtin_model is not None:
            audio_int16 = (audio_16k * 32767).astype(np.int16)
            try:
                _oww_builtin_model.predict(audio_int16)
            except Exception:
                return
            for _, buf in _oww_builtin_model.prediction_buffer.items():
                if buf:
                    score = max(score, float(buf[-1]))

        if score >= WAKE_THRESHOLD:
            energy = np.abs(audio_16k).mean()
            if energy < 0.01:
                return
            now = time.time()
            if (now - _wake_last_trigger > WAKE_COOLDOWN_SEC
                    and now - _last_deactivate_ts > WAKE_POST_DEACTIVATE_COOLDOWN):
                _wake_last_trigger = now
                console.print(f"[bold green]✓ Wake word! score={score:.3f}[/bold green]")
                threading.Thread(target=_activate_bot, daemon=True).start()

    else:
        _record_queue.put(audio_16k)


def _start_unified_stream():
    global _stream_start_ts, _stream_warmed_up

    device   = _auto_detect_device()
    errors   = []
    candidates = [device]
    for fb in [0, 1, 2]:
        if fb not in candidates:
            candidates.append(fb)

    for dev in candidates:
        try:
            stream = sd.InputStream(
                samplerate=HW_SR,
                channels=HW_CHANNELS,
                dtype="int32",
                blocksize=UNIFIED_BLOCK_FRAMES,
                callback=_unified_audio_callback,
                device=dev,
            )
            stream.start()
            _stream_start_ts  = time.time()
            _stream_warmed_up = False
            console.print(
                f"[cyan]✓ Unified stream: device={dev} "
                f"{HW_SR}Hz/{HW_CHANNELS}ch → {TARGET_SR}Hz mono[/cyan]"
            )
            console.print(f"[dim]  Warmup {STREAM_WARMUP_SEC:.0f}s...[/dim]")
            return stream
        except Exception as e:
            errors.append(f"device={dev}: {e}")
            console.print(f"[yellow]⚠ Stream device={dev} lỗi: {e}[/yellow]")

    console.print("[red]✗ Không mở được audio stream![/red]")
    for err in errors:
        console.print(f"[red]  {err}[/red]")
    return None


# ══════════════════════════════════════════════════════════════════
# SESSION & LOGGING
# ══════════════════════════════════════════════════════════════════

SESSION_ID         = os.getenv("SESSION_ID") or str(uuid.uuid4())[:8]
SESSION_START_TIME = time.time()
console.print(f"[dim]Session ID: {SESSION_ID}[/dim]")

_IMPLICIT_LOG_PATH = Path("logs/implicit_feedback.jsonl")

HF_TOKEN        = os.getenv("HF_TOKEN", "")
HF_DATASET_REPO = os.getenv("HF_DATASET_REPO", "")


def _log_implicit(event: str, **kwargs):
    record = {
        "ts":         datetime.now(timezone.utc).isoformat(),
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


def _log_to_hf():
    if not HF_TOKEN or not HF_DATASET_REPO:
        console.print("[dim]HF logging chưa cấu hình (thiếu HF_TOKEN / HF_DATASET_REPO)[/dim]")
        return

    if not _IMPLICIT_LOG_PATH.exists() or _IMPLICIT_LOG_PATH.stat().st_size == 0:
        console.print("[dim]Không có log để upload[/dim]")
        return

    try:
        import base64

        ts_str  = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        hf_path = f"logs/{SESSION_ID}_{ts_str}.jsonl"

        with open(_IMPLICIT_LOG_PATH, "rb") as f:
            file_bytes = f.read()

        encoded = base64.b64encode(file_bytes).decode()

        api_url = f"https://huggingface.co/api/datasets/{HF_DATASET_REPO}/commit/main"
        payload = {
            "commit_message": f"log {SESSION_ID} {ts_str}",
            "summary":        f"log {SESSION_ID} {ts_str}",
            "operations": [
                {
                    "operation": "addOrUpdate",
                    "path":      hf_path,
                    "encoding":  "base64",
                    "content":   encoded,
                }
            ],
        }

        with httpx.Client(timeout=30) as client:
            r = client.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {HF_TOKEN}",
                    "Content-Type":  "application/json",
                },
                json=payload,
            )

        if r.status_code in (200, 201):
            console.print(f"[green]✓ Log uploaded: {HF_DATASET_REPO}/{hf_path}[/green]")
        else:
            console.print(f"[yellow]⚠ HF upload lỗi {r.status_code}: {r.text[:120]}[/yellow]")

    except Exception as e:
        console.print(f"[yellow]⚠ HF upload exception: {e}[/yellow]")


# ══════════════════════════════════════════════════════════════════
# STATE HELPERS
# ══════════════════════════════════════════════════════════════════

_last_response: str = ""
_last_numbers:  str = ""
_last_topic:    str = ""

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
# BEEP
# ══════════════════════════════════════════════════════════════════

_pygame_mixer_ready = False

def _init_pygame_mixer():
    global _pygame_mixer_ready
    if _pygame_mixer_ready:
        return True
    try:
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=4096)
        _pygame_mixer_ready = True
        return True
    except Exception as e:
        console.print(f"[dim red]pygame mixer init lỗi: {e}[/dim red]")
        return False


def _beep(freq: float = 880.0, duration: float = 0.25, volume: float = 0.4):
    try:
        import pygame
        if not _init_pygame_mixer():
            return
        sr    = 44100
        n     = int(sr * duration)
        t     = np.linspace(0, duration, n, endpoint=False)
        wave  = np.sin(2 * math.pi * freq * t)
        fade  = int(sr * 0.02)
        ramp  = np.linspace(0, 1, fade)
        wave[:fade]  *= ramp
        wave[-fade:] *= ramp[::-1]
        pcm    = (wave * volume * 32767).astype(np.int16)
        stereo = np.column_stack([pcm, pcm])
        sound  = pygame.sndarray.make_sound(stereo)
        sound.play()
        pygame.time.wait(int(duration * 1000) + 50)
    except ImportError:
        pass
    except Exception as e:
        console.print(f"[dim red]Beep lỗi: {e}[/dim red]")


# ══════════════════════════════════════════════════════════════════
# DETECT MODE
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
    t = text.lower()
    if any(kw in t for kw in _OCR_TRIGGERS):
        return "OCR"
    if any(kw in t for kw in _SOLVE_IMAGE_TRIGGERS):
        return "SOLVE_IMAGE"
    return "NORMAL"


# ══════════════════════════════════════════════════════════════════
# CAMERA + HC-SR04
# ══════════════════════════════════════════════════════════════════

def get_distance_cm() -> float:
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
        return 25.0


def capture_image_bytes() -> bytes | None:
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
            asyncio.run(speak(f"Đưa sách lại gần hơn khoảng {diff} xăng-ti-mét nữa nhé."))
            time.sleep(2)
            continue
        if dist < DIST_MIN:
            diff = int((DIST_MIN + DIST_MAX) / 2 - dist)
            asyncio.run(speak(f"Lùi sách ra xa hơn khoảng {diff} xăng-ti-mét nhé."))
            time.sleep(2)
            continue
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
# ══════════════════════════════════════════════════════════════════

def call_api_ask(text: str) -> str:
    try:
        with httpx.Client(timeout=API_TIMEOUT) as client:
            r = client.post(
                f"{API_BASE}/ask",
                json={"question": text, "session_id": SESSION_ID},
            )
            r.raise_for_status()
            answer = r.json().get("answer", "")
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
    try:
        with httpx.Client(timeout=API_TIMEOUT) as client:
            r = client.post(
                f"{API_BASE}/ocr",
                files={"file": ("photo.jpg", img_bytes, "image/jpeg")},
            )
            r.raise_for_status()
            return r.json().get("answer", "")
    except Exception as e:
        console.print(f"[red]API /ocr lỗi: {e}")
        return "Có lỗi xảy ra, bạn thử lại sau nha!"


def call_api_solve_image(img_bytes: bytes) -> str:
    try:
        with httpx.Client(timeout=API_TIMEOUT) as client:
            r = client.post(
                f"{API_BASE}/solve_image",
                files={"file": ("photo.jpg", img_bytes, "image/jpeg")},
                params={"session_id": SESSION_ID},
            )
            r.raise_for_status()
            return r.json().get("answer", "")
    except Exception as e:
        console.print(f"[red]API /solve_image lỗi: {e}")
        return "Có lỗi xảy ra, bạn thử lại sau nha!"


def get_response(text: str) -> str:
    mode = detect_mode(text)
    console.print(f"[dim]Mode: {mode}[/dim]")
    if mode == "NORMAL":
        return call_api_ask(text)
    asyncio.run(speak("Oke, tui chuẩn bị chụp ảnh nhé."))
    img_bytes = guide_and_capture()
    if img_bytes is None:
        return "Tui không chụp được ảnh, bạn thử lại nhé."
    if mode == "OCR":
        return call_api_ocr(img_bytes)
    return call_api_solve_image(img_bytes)


# ══════════════════════════════════════════════════════════════════
# TRANSCRIBE
# ══════════════════════════════════════════════════════════════════

def transcribe(audio_np: np.ndarray, language: str = "vi") -> str:
    try:
        import soundfile as sf
        energy = np.abs(audio_np).mean()
        if energy < 0.005:
            return ""
        if energy < 0.05:
            gain     = min(0.05 / (energy + 1e-9), 10.0)
            audio_np = np.clip(audio_np * gain, -1.0, 1.0)
            console.print(f"[dim]Khuếch đại x{gain:.1f}[/dim]")
        if len(audio_np) / TARGET_SR < 0.5:
            return ""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            sf.write(tmp_path, audio_np.astype(np.float32), TARGET_SR)
            with open(tmp_path, "rb") as af:
                result = groq_client.audio.transcriptions.create(
                    model="whisper-large-v3", file=af, language=language,
                )
        os.unlink(tmp_path)
        return result.text.strip()
    except Exception as e:
        console.print(f"[red]STT lỗi: {e}")
        return ""


# ══════════════════════════════════════════════════════════════════
# WAKE WORD MODEL LOADING
# ══════════════════════════════════════════════════════════════════

def _load_wake_model():
    global _oww_builtin_model, _oww_custom_sess, _oww_custom_iname
    global _oww_audio_feats, _use_custom_model

    if WAKE_MODEL.endswith(".onnx"):
        model_path = Path(WAKE_MODEL)
        if not model_path.exists():
            console.print(f"[red]Không tìm thấy: {WAKE_MODEL}[/red]")
            return False
        try:
            import onnxruntime as ort
            from openwakeword.utils import AudioFeatures

            sess              = ort.InferenceSession(str(model_path))
            _oww_custom_sess  = sess
            _oww_custom_iname = sess.get_inputs()[0].name
            _oww_audio_feats  = AudioFeatures()
            _use_custom_model = True

            console.print(f"[green]✓ Custom model: {model_path.name}[/green]")
            console.print(f"[dim]  Pipeline: AudioFeatures → {_oww_custom_iname} → score[/dim]")
            return True

        except ImportError as e:
            console.print(f"[red]Thiếu thư viện: {e}[/red]")
            return False
        except Exception as e:
            console.print(f"[red]Lỗi load custom model: {e}[/red]")
            return False

    try:
        from openwakeword.model import Model
        import openwakeword

        oww_dir    = Path(openwakeword.__file__).parent
        models_dir = oww_dir / "resources" / "models"
        candidates = list(models_dir.glob(f"{WAKE_MODEL}*.onnx"))

        if not candidates:
            available = [f.stem for f in models_dir.glob("*.onnx")
                         if not any(x in f.stem for x in
                                    ["embedding", "melspectrogram", "silero"])]
            console.print(f"[red]Không tìm thấy model '{WAKE_MODEL}'[/red]")
            console.print(f"[yellow]Có sẵn: {available}[/yellow]")
            return False

        model_path = str(sorted(candidates)[-1])
        oww = Model(wakeword_model_paths=[model_path])
        _oww_builtin_model = oww
        _use_custom_model  = False

        console.print(f"[green]✓ Built-in model: {Path(model_path).name}[/green]")
        return True

    except ImportError:
        console.print("[red]openwakeword chưa cài.[/red]")
        return False
    except Exception as e:
        console.print(f"[red]Lỗi load built-in model: {e}[/red]")
        return False


def _fallback_enter_listener():
    console.print("[yellow]⚠ Fallback: nhấn Enter để kích hoạt PhysBot[/yellow]")
    while True:
        try:
            input()
            _activate_bot()
        except EOFError:
            time.sleep(3)


# ══════════════════════════════════════════════════════════════════
# STATE TRANSITIONS
# ══════════════════════════════════════════════════════════════════

def _activate_bot():
    global _bot_state, _last_active_ts
    with _state_lock:
        if _bot_state == STATE_ACTIVE:
            return
        _bot_state      = STATE_ACTIVE
        _last_active_ts = time.time()
    while not _record_queue.empty():
        try:
            _record_queue.get_nowait()
        except Empty:
            break
    console.print("[cyan]══ ACTIVE — Tui đang nghe! ══[/cyan]")
    _beep(880, 0.25)
    _activated_event.set()


def _deactivate_bot(reason: str = "timeout"):
    global _bot_state, _last_deactivate_ts
    with _state_lock:
        _bot_state = STATE_IDLE
    _last_deactivate_ts = time.time()
    console.print(f"[dim]── IDLE ({reason}) ──[/dim]")
    _beep(440, 0.5, 0.8)
    _activated_event.clear()


def _idle_timeout_watcher():
    global _last_active_ts
    while True:
        time.sleep(5)
        if _bot_state == STATE_ACTIVE:
            if _tts_playing:
                _last_active_ts = time.time()
                continue
            elapsed = time.time() - _last_active_ts
            if elapsed >= IDLE_TIMEOUT_SEC:
                console.print(f"[dim]Timeout {IDLE_TIMEOUT_SEC:.0f}s → về IDLE[/dim]")
                _deactivate_bot("timeout")


# ══════════════════════════════════════════════════════════════════
# RECORD QUESTION
# ══════════════════════════════════════════════════════════════════

def record_question(
    silence_threshold: float = ENERGY_MIN,
    silence_duration:  float = 3.0,
    max_duration:      float = 30.0,
) -> np.ndarray:
    chunk_duration = UNIFIED_CHUNK_MS / 1000.0
    silent_chunks  = 0
    started        = False
    collected      = []
    deadline       = time.time() + max_duration

    console.print(
        f"[dim]Ghi âm từ queue (ngưỡng={silence_threshold:.3f}, "
        f"dừng sau {silence_duration:.0f}s im lặng)...[/dim]"
    )

    while True:
        if time.time() >= deadline:
            console.print(f"[dim]Max {max_duration:.0f}s → tự dừng[/dim]")
            break
        if _bot_state != STATE_ACTIVE:
            break

        try:
            chunk = _record_queue.get(timeout=0.5)
        except Empty:
            if started:
                silent_chunks += 1
                if silent_chunks >= silence_duration / chunk_duration:
                    break
            continue

        collected.append(chunk)
        energy = np.abs(chunk).mean()

        if not started:
            if energy > silence_threshold:
                started = True
                console.print("[dim]Đang nghe câu hỏi...[/dim]")
            continue

        if energy < silence_threshold:
            silent_chunks += 1
        else:
            silent_chunks = 0

        if silent_chunks >= silence_duration / chunk_duration:
            break

    if not collected:
        return np.zeros(TARGET_SR, dtype=np.float32)
    return np.concatenate(collected).astype(np.float32)


# ══════════════════════════════════════════════════════════════════
# TTS
# ══════════════════════════════════════════════════════════════════

def sanitize_for_tts(text: str) -> str:
    text = re.sub(r'\*{1,3}([^*\n]+)\*{1,3}', r'\1', text)
    text = re.sub(r'(?m)^#{1,6}\s*', '', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'(?m)^\s*[-•]\s+', '', text)
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
    for sym, name in [('N','niutơn'),('J','jun'),('W','oát'),('V','vôn'),
                      ('A','ampe'),('F','fara'),('H','henry'),('T','tesla'),
                      ('C','culông'),('K','ken-vin')]:
        text = re.sub(rf'(\d)\s*{re.escape(sym)}\b', rf'\1 {name}', text)
    text = re.sub(r'(\d)\s*Ω', r'\1 ôm', text)
    text = re.sub(r'\bΩ\b', 'ôm', text)
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
    for old, new in [('₀',' không'),('₁',' một'),('₂',' hai'),('₃',' ba'),
                     ('₄',' bốn'),('₅',' năm'),('₆',' sáu'),('₇',' bảy'),
                     ('₈',' tám'),('₉',' chín')]:
        text = text.replace(old, new)
    for old, new in [('²',' bình phương'),('³',' lập phương'),('⁻',' trừ '),
                     ('⁰',' mũ không'),('⁴',' mũ bốn'),('⁵',' mũ năm'),
                     ('⁶',' mũ sáu'),('⁷',' mũ bảy'),('⁸',' mũ tám'),
                     ('⁹',' mũ chín'),('ⁿ',' mũ n'),('¹',' mũ một')]:
        text = text.replace(old, new)
    text = re.sub(r'√\(([^)]+)\)', r'căn bậc hai của \1', text)
    text = text.replace('√', 'căn bậc hai của ')
    for old, new in [('½',' một phần hai '),('¼',' một phần bốn '),
                     ('¾',' ba phần bốn '),('⅓',' một phần ba '),('⅔',' hai phần ba ')]:
        text = text.replace(old, new)
    for old, new in [
        ('α','anpha'),('β','bê-ta'),('γ','gama'),('ω','ô-mê-ga'),
        ('λ','lăm-đa'),('θ','tê-ta'),('π','pi'),('Δ','delta '),('δ','delta '),
        ('Φ','phi '),('φ','phi '),('Σ','tổng '),('σ','sigma '),('μ','muy'),
        ('η','êta'),('ρ','rô'),('ε','êp-xi-lông'),('τ','tô'),('ξ','xi'),
        ('Λ','lăm-đa'),('Ω','ô-mê-ga hoa'),
    ]:
        text = text.replace(old, new)
    text = re.sub(r'([a-zA-ZÀ-ỹ])(\d)', r'\1 \2', text)
    text = re.sub(r'\s+x\s+', ' nhân ', text)
    for old, new in [('×',' nhân '),('÷',' chia '),('≈',' xấp xỉ '),
                     ('≠',' khác '),('≥',' lớn hơn hoặc bằng '),
                     ('≤',' nhỏ hơn hoặc bằng '),('>',' lớn hơn '),('<',' nhỏ hơn '),
                     ('→',' suy ra '),('⇒',' suy ra '),('∞','vô cực'),
                     ('%',' phần trăm'),('°',' độ'),('=',' bằng ')]:
        text = text.replace(old, new)
    text = text.replace('+', ' cộng ')
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


def _flush_record_queue():
    flushed = 0
    while not _record_queue.empty():
        try:
            _record_queue.get_nowait()
            flushed += 1
        except Empty:
            break
    if flushed:
        console.print(f"[dim]Flush {flushed} chunk echo sau TTS[/dim]")


def _post_tts_cleanup(extra_mute: float = 0.0):
    global _tts_playing, _last_active_ts
    total_mute = POST_TTS_MUTE_SEC + extra_mute
    time.sleep(total_mute)
    _flush_record_queue()
    _last_active_ts = time.time()
    _tts_playing    = False
    if _bot_state == STATE_ACTIVE:
        _beep(880, 0.15, 0.5)
    console.print(f"[dim]Mic bật lại (mute {total_mute:.1f}s)[/dim]")


async def speak(text: str):
    global _tts_playing, _last_active_ts
    text = sanitize_for_tts(text).strip()
    if not text:
        return
    if text[-1] not in '.!?':
        text += '.'
    console.print(f"[magenta]TTS ({len(text)} ký tự): {repr(text[:80])}[/magenta]")
    _tts_playing    = True
    _last_active_ts = time.time()
    _init_pygame_mixer()

    try:
        import edge_tts
        import pygame
        voice = os.getenv("TTS_VOICE", "vi-VN-HoaiMyNeural")
        communicate = edge_tts.Communicate(text, voice)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name
        await communicate.save(tmp_path)
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            _last_active_ts = time.time()
            pygame.time.wait(100)
        time.sleep(0.3)
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        os.unlink(tmp_path)
        extra_mute = min(len(text) / 200, 2.0)
        threading.Thread(target=_post_tts_cleanup, args=(extra_mute,), daemon=True).start()
        return
    except ImportError:
        pass
    except Exception as e:
        console.print(f"[yellow]edge-tts lỗi: {e}, fallback gTTS")

    try:
        from gtts import gTTS
        import pygame
        tts = gTTS(text=text, lang='vi', slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp_path = f.name
        tts.save(tmp_path)
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            _last_active_ts = time.time()
            pygame.time.wait(100)
        time.sleep(0.3)
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        os.unlink(tmp_path)
    except Exception as e:
        console.print(f"[red]TTS lỗi hoàn toàn: {e}")
    finally:
        extra_mute = min(len(text) / 200, 2.0)
        threading.Thread(target=_post_tts_cleanup, args=(extra_mute,), daemon=True).start()


# ══════════════════════════════════════════════════════════════════
# PROCESS ONE TURN
# ══════════════════════════════════════════════════════════════════

def process_turn(text: str):
    global _last_response, _last_numbers, _last_topic, _last_active_ts
    _last_active_ts = time.time()

    corrected = correct_physics_text(text, use_llm=False)
    log_correction(text, corrected)
    if corrected != text:
        console.print(f"[yellow]Fixed: {corrected}")

    if _is_repeat_request(corrected) and _last_response:
        _log_implicit("repeat_request", question=corrected[:80])
        console.print("[yellow]→ Phát lại response trước[/yellow]")
        asyncio.run(speak(_last_response))
        return

    if _is_not_understand(corrected) and _last_response:
        _log_implicit("not_understand", question=corrected[:80])

    current_topic = _extract_topic(corrected)
    if _last_topic and current_topic and _last_topic == current_topic:
        _log_implicit("rephrase_same", topic=current_topic, question=corrected[:80])
    _last_topic = current_topic

    if _is_keep_data_request(corrected) and _last_numbers:
        corrected = f"{corrected}. Số liệu đã cho từ bài trước: {_last_numbers}"
        console.print(f"[dim]→ Inject số liệu cũ: {_last_numbers}[/dim]")

    extracted = _extract_numbers(corrected)
    if extracted:
        _last_numbers = extracted

    _log_implicit("question", question=corrected[:120])

    asyncio.run(speak("Oke, đợi tui một chút nhé!"))

    with console.status("Đang xử lý...", spinner="dots"):
        t0       = time.time()
        response = get_response(corrected)
        t1       = time.time()

    _last_response  = response
    _last_active_ts = time.time()

    _log_implicit("answer", question=corrected[:80], answer=response[:120],
                  api_sec=round(t1 - t0, 2))

    if len(response) > 2000:
        chunk = response[:2000]
        cut   = max(chunk.rfind('.'), chunk.rfind('!'), chunk.rfind('?'))
        response = response[:cut+1] + " (tui rút gọn nha)" if cut > 0 else response[:2000]

    console.print(f"[cyan]PhysBot: {response}")
    console.print(f"[dim]API: {t1-t0:.2f}s[/dim]")
    asyncio.run(speak(response))
    _last_active_ts = time.time()


# ══════════════════════════════════════════════════════════════════
# ACTIVE LOOP
# ══════════════════════════════════════════════════════════════════

def active_loop():
    turn_count = 0
    while True:
        if _bot_state != STATE_ACTIVE:
            break

        audio_np = record_question()

        if _bot_state != STATE_ACTIVE:
            break

        if audio_np.size == 0:
            time.sleep(0.1)
            continue

        energy = np.abs(audio_np).mean()
        if energy < ENERGY_MIN * 0.5:
            console.print(f"[dim]Energy {energy:.4f} quá thấp, bỏ qua[/dim]")
            time.sleep(0.1)
            continue

        with console.status("Nhận dạng giọng nói...", spinner="dots"):
            raw_text = transcribe(audio_np, language="vi")

        if not raw_text.strip():
            console.print("[dim]Không nhận ra giọng nói[/dim]")
            continue

        console.print(f"[yellow]Bạn: {raw_text}")
        turn_count += 1
        process_turn(raw_text)

    console.print(f"[dim]Active loop kết thúc sau {turn_count} lượt[/dim]")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    console.print("[cyan]╔══════════════════════════════════════╗[/cyan]")
    console.print("[cyan]║     PhysBot Pi Client v2.9.5         ║[/cyan]")
    console.print("[cyan]║     No ALSA Underrun Edition         ║[/cyan]")
    console.print("[cyan]╚══════════════════════════════════════╝[/cyan]")
    console.print(f"[cyan]Server   : {API_BASE}")
    console.print(f"[cyan]Session  : {SESSION_ID}")
    console.print(f"[cyan]Wake     : model='{WAKE_MODEL}' threshold={WAKE_THRESHOLD}")
    console.print(f"[cyan]Energy   : {ENERGY_MIN} | PostTTS mute: {POST_TTS_MUTE_SEC}s")
    console.print(f"[cyan]Timeout  : {IDLE_TIMEOUT_SEC:.0f}s → IDLE")
    console.print(f"[cyan]Warmup   : {STREAM_WARMUP_SEC:.0f}s sau khi stream start")
    console.print(f"[cyan]Mic      : device={INPUT_DEVICE} @ {HW_SR}Hz/{HW_CHANNELS}ch → {TARGET_SR}Hz")
    hf_status = HF_DATASET_REPO if (HF_TOKEN and HF_DATASET_REPO) else "không cấu hình"
    console.print(f"[cyan]HF Log   : {hf_status}")
    console.print()

    try:
        from scipy.signal import resample_poly
        console.print("[green]✓ scipy OK[/green]")
    except ImportError:
        console.print("[red]✗ scipy chưa cài! Chạy: pip install scipy[/red]")
        sys.exit(1)

    try:
        with httpx.Client(timeout=5) as client:
            r    = client.get(f"{API_BASE}/health")
            info = r.json()
            console.print(
                f"[green]Server OK — model: {info.get('model','?')} | "
                f"chromadb: {info.get('chromadb','?')}"
            )
    except Exception:
        console.print(f"[yellow]⚠ Không kết nối được server tại {API_BASE}")

    _init_pygame_mixer()
    model_ok = _load_wake_model()
    _stream = _start_unified_stream()

    if not model_ok:
        threading.Thread(target=_fallback_enter_listener, daemon=True).start()

    if _stream is None:
        console.print("[yellow]⚠ Chạy không có mic — chỉ dùng Enter mode[/yellow]")
        threading.Thread(target=_fallback_enter_listener, daemon=True).start()

    threading.Thread(target=_idle_timeout_watcher, daemon=True).start()

    wake_hint = "PhysBot" if WAKE_MODEL.endswith(".onnx") else WAKE_MODEL.replace("_", " ").title()
    console.print(f"\n[dim]IDLE — Nói '{wake_hint}' để bắt đầu![/dim]\n")

    try:
        while True:
            _activated_event.wait()
            _activated_event.clear()
            active_loop()
            console.print("[dim]IDLE — đang chờ wake word...[/dim]")
    except KeyboardInterrupt:
        duration_min = round((time.time() - SESSION_START_TIME) / 60, 1)
        _log_implicit("session_end", duration_minutes=duration_min)
        console.print(f"\n[dim]Session {SESSION_ID}: {duration_min} phút[/dim]")
        _log_to_hf()
        console.print("[red]Thoát...")
        if _stream:
            _stream.stop()
            _stream.close()
