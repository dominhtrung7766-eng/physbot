"""
main.py
──────
PhysBot API — FastAPI server v3.2

THAY ĐỔI v3.2 so với v3.1:
  - [FIX] ChromaDB download chạy trong background thread
          → server bind port ngay lập tức, không bị Render restart loop
  - Giữ nguyên toàn bộ logic v3.1
"""
import re
import json
import os
import base64
import time
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from groq import Groq
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import sys

load_dotenv()
print("=== PhysBot starting ===", flush=True)
sys.stdout.flush()

# ══════════════════════════════════════════════════════════════════
# [HF] AUTO-DOWNLOAD CHROMADB
# ══════════════════════════════════════════════════════════════════

def _ensure_chromadb():
    db_path    = Path("data/chroma_db")
    hf_repo_id = os.getenv("HF_REPO_ID")
    hf_token   = os.getenv("HF_TOKEN")

    if db_path.exists() and any(db_path.iterdir()):
        print(f"[startup] ChromaDB đã có tại {db_path}")
        return

    if not hf_repo_id:
        print("[startup] CẢNH BÁO: ChromaDB chưa có và HF_REPO_ID chưa cấu hình.")
        print("[startup] Chạy: python scripts/download_db.py  hoặc  python scripts/ingest.py")
        return

    print(f"[startup] ChromaDB chưa có, đang download từ {hf_repo_id}...")
    try:
        from huggingface_hub import snapshot_download
        import shutil

        db_path.mkdir(parents=True, exist_ok=True)
        tmp_path = Path("data/_hf_tmp")

        snapshot_download(
            repo_id=hf_repo_id,
            repo_type="dataset",
            local_dir=str(tmp_path),
            token=hf_token,
        )

        src = tmp_path / "chroma_db"
        if src.exists():
            if db_path.exists():
                shutil.rmtree(db_path)
            shutil.move(str(src), str(db_path))
        else:
            if db_path.exists():
                shutil.rmtree(db_path)
            shutil.move(str(tmp_path), str(db_path))

        if tmp_path.exists():
            shutil.rmtree(tmp_path, ignore_errors=True)

        print(f"[startup] ChromaDB download xong!")

    except ImportError:
        print("[startup] Thiếu huggingface_hub. Chạy: pip install huggingface_hub")
    except Exception as e:
        print(f"[startup] Lỗi download ChromaDB: {e}")
        print("[startup] Server vẫn chạy nhưng RAG sẽ không hoạt động.")


# ── [FIX v3.2] Download trong background — KHÔNG block startup ───
# Server bind port ngay lập tức → Render health check pass
# Request đến trong lúc DB chưa ready → trả 503 thay vì crash
_db_ready = threading.Event()

def _download_db_background():
    print("[startup] Bắt đầu download ChromaDB...", flush=True)
    _ensure_chromadb()
    print("[startup] ChromaDB sẵn sàng!", flush=True)
    _db_ready.set()
threading.Thread(target=_download_db_background, daemon=True).start()


# ── Import sau khi thread đã start (rag_pipeline load lazy) ──────
from backend.rag_pipeline import (
    retrieve_context,
    build_rag_prompt,
    classify_query,
    is_out_of_scope,
)
from backend.calculator import handle_tool_call, CALCULATOR_TOOL_SCHEMA
from backend.prompts import TTS_RULES, PHYSBOT_SYSTEM_PROMPT, VOICE_INPUT_ADDON



# ══════════════════════════════════════════════════════════════════
# [D1] STRUCTURED LOGGING
# ══════════════════════════════════════════════════════════════════

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_obj = {
            "ts":      datetime.utcnow().isoformat() + "Z",
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
        }
        if record.exc_info:
            log_obj["exc"] = self.formatException(record.exc_info)
        for key in ("session_id", "query_type", "mode", "duration_ms",
                    "tool_expr", "tool_result", "rating"):
            if hasattr(record, key):
                log_obj[key] = getattr(record, key)
        return json.dumps(log_obj, ensure_ascii=False)


def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("physbot")
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setFormatter(_JsonFormatter())
    logger.addHandler(ch)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(log_dir / "physbot.jsonl", encoding="utf-8")
    fh.setFormatter(_JsonFormatter())
    logger.addHandler(fh)

    return logger


log = _setup_logger()


# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

GROQ_API_KEY        = os.getenv("GROQ_API_KEY")
ADMIN_TOKEN         = os.getenv("ADMIN_TOKEN", "change-me-in-dotenv")
MAIN_MODEL          = "llama-3.1-8b-instant"
MAX_TOKENS          = 1024
MAX_HISTORY_TURNS   = 3
MAX_TOOL_LOOPS      = 4
SESSION_MAX_TURNS   = 10
SESSION_TTL_MINUTES = 30
FEEDBACK_LOG_PATH   = Path("logs/feedback.jsonl")

VISION_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
]

if not GROQ_API_KEY:
    raise RuntimeError("Thiếu GROQ_API_KEY trong .env")

groq_client        = Groq(api_key=GROQ_API_KEY)
FULL_SYSTEM_PROMPT = TTS_RULES + "\n\n" + PHYSBOT_SYSTEM_PROMPT + "\n\n" + VOICE_INPUT_ADDON


# ══════════════════════════════════════════════════════════════════
# [T2] RATE LIMITING
# ══════════════════════════════════════════════════════════════════

limiter = Limiter(key_func=get_remote_address)
app     = FastAPI(title="PhysBot API", version="3.2")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════
# [T1] SESSION STORE
# ══════════════════════════════════════════════════════════════════

class _Session:
    def __init__(self):
        self.messages: list[dict] = []
        self.last_used: datetime  = datetime.utcnow()

    def touch(self):
        self.last_used = datetime.utcnow()

    def is_expired(self) -> bool:
        return (datetime.utcnow() - self.last_used) > timedelta(minutes=SESSION_TTL_MINUTES)


_sessions: dict[str, _Session] = {}
_sessions_lock = threading.Lock()


def _get_session(session_id: str) -> _Session:
    with _sessions_lock:
        if session_id not in _sessions or _sessions[session_id].is_expired():
            _sessions[session_id] = _Session()
            log.debug("Session created/reset", extra={"session_id": session_id})
        sess = _sessions[session_id]
        sess.touch()
        return sess


def _prune_expired_sessions():
    with _sessions_lock:
        expired = [k for k, v in _sessions.items() if v.is_expired()]
        for k in expired:
            del _sessions[k]


def _trim_history_for_llm(history: list[dict]) -> list[dict]:
    max_msgs = MAX_HISTORY_TURNS * 2
    return history[-max_msgs:] if len(history) > max_msgs else history


# ══════════════════════════════════════════════════════════════════
# [D2] ADMIN AUTH
# ══════════════════════════════════════════════════════════════════

_bearer_scheme = HTTPBearer()


def _verify_admin(credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme)):
    if credentials.credentials != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Token không hợp lệ")
    return True


# ══════════════════════════════════════════════════════════════════
# [T3] VISION MODEL CALL VỚI FALLBACK
# ══════════════════════════════════════════════════════════════════

def _call_vision(messages: list, max_tokens: int = 512) -> str:
    last_err = None
    for model in VISION_MODELS:
        try:
            response = groq_client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            log.warning(f"Vision model {model} failed: {e}")
            last_err = e
    raise HTTPException(status_code=503,
                        detail="Vision model tạm thời không khả dụng, thử lại sau.")


# ══════════════════════════════════════════════════════════════════
# HELPER — xử lý tool calls
# ══════════════════════════════════════════════════════════════════

def _process_tool_calls(tool_calls, messages: list, loop_idx: int,
                        session_id: str = ""):
    for tool_call in tool_calls:
        if tool_call.function.name != "calculate":
            messages.append({"role": "tool", "tool_call_id": tool_call.id,
                              "content": "Tool không tồn tại"})
            continue

        result = handle_tool_call(tool_call)

        log.info(
            f"Tool calc {'OK' if result['success'] else 'ERR'}: "
            f"{result['expression']} = {result['result']}",
            extra={
                "session_id":  session_id,
                "tool_expr":   result["expression"],
                "tool_result": result["result"],
            },
        )

        tool_content = (
            f"Kết quả: {result['result']}"
            if result["success"]
            else f"Lỗi: {result['result']}. Thử lại."
        )
        messages.append({
            "role":         "tool",
            "tool_call_id": result["tool_call_id"],
            "content":      tool_content,
        })


# ══════════════════════════════════════════════════════════════════
# MODE HANDLERS
# ══════════════════════════════════════════════════════════════════

def _build_messages(question: str, rag_prompt: str,
                    history: list[dict], use_rag: bool, use_history: bool) -> list[dict]:
    user_content = rag_prompt if use_rag else question
    hist         = _trim_history_for_llm(history) if use_history else []
    return (
        [{"role": "system", "content": FULL_SYSTEM_PROMPT}]
        + hist
        + [{"role": "user", "content": user_content}]
    )


def _llm_call(messages: list[dict], use_tools: bool) -> object:
    kwargs = dict(model=MAIN_MODEL, max_tokens=MAX_TOKENS, messages=messages)
    if use_tools:
        kwargs["tools"]       = [CALCULATOR_TOOL_SCHEMA]
        kwargs["tool_choice"] = "required" if use_tools else "auto"
    return groq_client.chat.completions.create(**kwargs)


def handle_normal(question: str, session_id: str = "") -> str:
    t0 = time.perf_counter()

    # Đảm bảo DB sẵn sàng trước khi RAG
    if not _db_ready.wait(timeout=120):
        log.warning("DB not ready after 120s", extra={"session_id": session_id})

    if is_out_of_scope(question):
        return "Haha tui chỉ giỏi Vật lý THPT thôi nha, mấy thứ khác tui bó tay!"

    context    = retrieve_context(question=question,
                                  system_prompt=FULL_SYSTEM_PROMPT,
                                  verbose=True, groq_client=groq_client)
    rag_prompt = build_rag_prompt(question, context)

    sess       = _get_session(session_id) if session_id else None
    history    = sess.messages if sess else []
    query_type = classify_query(question)
    has_numbers = bool(re.search(r'\d+', question))
    use_tools = query_type in ("exercise", "mixed") or has_numbers

    log.info("LLM call start",
             extra={"session_id": session_id, "query_type": query_type})

    fallback_levels = [
        (True,  True),
        (False, True),
        (False, False),
    ]

    for level, (use_rag, use_hist) in enumerate(fallback_levels):
        messages = _build_messages(question, rag_prompt, history, use_rag, use_hist)
        if level > 0:
            log.warning(f"413 fallback level {level}",
                        extra={"session_id": session_id})
        try:
            for loop_idx in range(MAX_TOOL_LOOPS):
                response = _llm_call(messages, use_tools)
                msg      = response.choices[0].message

                if not msg.tool_calls:
                    answer = msg.content or ""
                    if sess is not None:
                        sess.messages.append({"role": "user",      "content": question})
                        sess.messages.append({"role": "assistant", "content": answer})
                        sess.messages = sess.messages[-(SESSION_MAX_TURNS * 2):]
                    duration_ms = int((time.perf_counter() - t0) * 1000)
                    log.info("LLM done",
                             extra={"session_id": session_id, "duration_ms": duration_ms})
                    return answer

                messages.append(msg)
                _process_tool_calls(msg.tool_calls, messages, loop_idx, session_id)

            final = groq_client.chat.completions.create(
                model=MAIN_MODEL, max_tokens=MAX_TOKENS, messages=messages)
            return final.choices[0].message.content or ""

        except Exception as e:
            err_str = str(e)
            if "413" in err_str or "Request too large" in err_str:
                if level < len(fallback_levels) - 1:
                    continue
                return "Câu hỏi quá dài, bạn thử hỏi ngắn gọn hơn nha!"
            raise

    return "Có lỗi xảy ra, bạn thử lại sau nha!"


def handle_ocr(img_b64: str, content_type: str) -> str:
    return _call_vision([{
        "role": "user",
        "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:{content_type};base64,{img_b64}"}},
            {"type": "text", "text": (
                "Đọc toàn bộ văn bản trong ảnh. "
                "Chỉ đọc nguyên văn — không giải thích, không giải bài. "
                "TTS-friendly: viết bằng chữ thay vì ký hiệu toán học."
            )},
        ],
    }])


def handle_solve_image(img_b64: str, content_type: str, session_id: str = "") -> str:
    extracted = _call_vision(
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{content_type};base64,{img_b64}"}},
                {"type": "text", "text": (
                    "Đây là ảnh bài tập vật lý. Trích xuất toàn bộ đề bài: "
                    "số liệu, yêu cầu, mô tả hình vẽ nếu có. "
                    "Trả về text thuần để giải tiếp."
                )},
            ],
        }],
    )
    if not extracted.strip():
        return "Không đọc được nội dung ảnh. Vui lòng chụp lại rõ hơn."

    log.info(f"SOLVE_IMAGE extracted: {extracted[:80]}...",
             extra={"session_id": session_id})
    return handle_normal(extracted, session_id=session_id)


# ══════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════════════

class AskRequest(BaseModel):
    question:   str
    session_id: Optional[str] = Field(default="")


class FeedbackRequest(BaseModel):
    session_id: str
    question:   str
    answer:     str
    rating:     int           = Field(..., ge=1, le=5)
    comment:    Optional[str] = ""


class IngestWebRequest(BaseModel):
    url:       str
    save_name: str
    force:     bool = Field(default=False)


class IngestPDFRequest(BaseModel):
    force: bool = Field(default=False)


# ══════════════════════════════════════════════════════════════════
# ENDPOINTS — CORE
# ══════════════════════════════════════════════════════════════════

@app.post("/ask")
@limiter.limit("30/minute")
async def ask(req: AskRequest, request: Request):
    # Chờ ChromaDB sẵn sàng, tối đa 60 giây
    if not _db_ready.wait(timeout=90):
        raise HTTPException(status_code=503,
                            detail="Database đang khởi động, thử lại sau 30 giây")
    _prune_expired_sessions()
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Câu hỏi không được để trống")
    try:
        answer = handle_normal(req.question.strip(), session_id=req.session_id)
        return {"answer": answer, "mode": "NORMAL", "session_id": req.session_id}
    except Exception as e:
        log.error(f"/ask error: {e}", extra={"session_id": req.session_id})
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ocr")
@limiter.limit("20/minute")
async def ocr(request: Request, file: UploadFile = File(...)):
    if not _db_ready.wait(timeout=60):
        raise HTTPException(status_code=503,
                            detail="Database đang khởi động, thử lại sau 30 giây")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file ảnh")
    try:
        img_b64 = base64.b64encode(await file.read()).decode()
        answer  = handle_ocr(img_b64, file.content_type)
        return {"answer": answer, "mode": "OCR"}
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"/ocr error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/solve_image")
@limiter.limit("5/minute")
async def solve_image(
    request:    Request,
    file:       UploadFile = File(...),
    session_id: str        = "",
):
    if not _db_ready.wait(timeout=60):
        raise HTTPException(status_code=503,
                            detail="Database đang khởi động, thử lại sau 30 giây")
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Chỉ chấp nhận file ảnh")
    try:
        img_b64 = base64.b64encode(await file.read()).decode()
        answer  = handle_solve_image(img_b64, file.content_type, session_id=session_id)
        return {"answer": answer, "mode": "SOLVE_IMAGE", "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"/solve_image error: {e}", extra={"session_id": session_id})
        raise HTTPException(status_code=500, detail=str(e))


# ══════════════════════════════════════════════════════════════════
# [D3] FEEDBACK
# ══════════════════════════════════════════════════════════════════

@app.post("/feedback")
async def feedback(req: FeedbackRequest):
    record = {
        "ts":         datetime.utcnow().isoformat() + "Z",
        "session_id": req.session_id,
        "question":   req.question,
        "answer":     req.answer,
        "rating":     req.rating,
        "comment":    req.comment or "",
    }
    try:
        FEEDBACK_LOG_PATH.parent.mkdir(exist_ok=True)
        with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        log.error(f"Ghi feedback lỗi: {e}")
        raise HTTPException(status_code=500, detail="Không ghi được feedback")

    log.info(f"Feedback rating={req.rating}",
             extra={"session_id": req.session_id, "rating": req.rating})
    return {"status": "ok", "received": record}


# ══════════════════════════════════════════════════════════════════
# [D2] ADMIN ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.post("/admin/ingest/web", dependencies=[Depends(_verify_admin)])
async def admin_ingest_web(req: IngestWebRequest):
    try:
        from backend.web_ingest import ingest_web
        result = ingest_web(req.url, req.save_name, force=req.force)
        return {
            "status":      "ok",
            "url":         req.url,
            "save_name":   req.save_name,
            "text_chunks": result.get("text", 0),
            "images":      result.get("images", 0),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/admin/ingest/pdf", dependencies=[Depends(_verify_admin)])
async def admin_ingest_pdf(req: IngestPDFRequest):
    try:
        from scripts.ingest import ingest
        ingest()
        return {"status": "ok", "message": "PDF ingest hoàn tất"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/admin/sessions", dependencies=[Depends(_verify_admin)])
async def admin_sessions():
    with _sessions_lock:
        return {
            "active_sessions": len(_sessions),
            "sessions": [
                {
                    "session_id": sid,
                    "turns":      len(s.messages) // 2,
                    "last_used":  s.last_used.isoformat() + "Z",
                    "expired":    s.is_expired(),
                }
                for sid, s in _sessions.items()
            ],
        }


@app.delete("/admin/sessions/{session_id}", dependencies=[Depends(_verify_admin)])
async def admin_delete_session(session_id: str):
    with _sessions_lock:
        if session_id in _sessions:
            del _sessions[session_id]
            return {"status": "ok", "deleted": session_id}
    raise HTTPException(status_code=404, detail="Session không tồn tại")


# ══════════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════════
@app.get("/debug/db")
async def debug_db():
    import chromadb as _chromadb
    db_path = Path("data/chroma_db")
    files = [str(f.relative_to(db_path)) for f in db_path.rglob("*")][:20] if db_path.exists() else []
    try:
        client = _chromadb.PersistentClient(path=str(db_path))
        collections = client.list_collections()
        counts = {c.name: c.count() for c in collections}
    except Exception as e:
        counts = {"ERROR": str(e)}
    return {
        "db_path_exists": db_path.exists(),
        "files": files,
        "collections": counts,
    }


@app.post("/debug/rag")
async def debug_rag(req: AskRequest):
    context = retrieve_context(
        question=req.question,
        system_prompt=FULL_SYSTEM_PROMPT,
        verbose=True,
        groq_client=groq_client,
    )
    return {
        "question": req.question,
        "context_length": len(context),
        "context_preview": context[:500] if context else "EMPTY",
    }
@app.get("/health")
async def health():
    db_path = Path("data/chroma_db")
    db_ok   = db_path.exists() and any(db_path.iterdir())

    return {
        "status":          "ok" if _db_ready.is_set() else "starting",
        "version":         "3.2",
        "model":           MAIN_MODEL,
        "vision_models":   VISION_MODELS,
        "chromadb":        "ok" if db_ok else "downloading",
        "db_ready":        _db_ready.is_set(),
        "active_sessions": len(_sessions),
    }
  
