"""
scripts/watch_log.py
────────────────────
Xem log app.py real-time thay vì dùng curl.

Dùng:
    python scripts/watch_log.py              # xem tất cả (app + server)
    python scripts/watch_log.py app          # chỉ logs/app.jsonl
    python scripts/watch_log.py server       # chỉ logs/physbot.jsonl
    python scripts/watch_log.py ERROR        # chỉ level ERROR (cả 2 file)
    python scripts/watch_log.py app ERROR    # chỉ app.jsonl + level ERROR
"""

import sys
import time
import json
import threading
from pathlib import Path

# ── Parse args ───────────────────────────────────────────────────
args       = [a.upper() for a in sys.argv[1:]]
LEVELS     = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
SOURCES    = {"APP", "SERVER"}

level_filter  = next((a for a in args if a in LEVELS),  None)
source_filter = next((a for a in args if a in SOURCES), None)  # None = cả hai

LOG_FILES = {
    "app":    Path("logs/app.jsonl"),
    "server": Path("logs/physbot.jsonl"),
}

# Màu cho từng level (ANSI)
COLORS = {
    "DEBUG":    "\033[2m",       # dim
    "INFO":     "\033[0m",       # normal
    "WARNING":  "\033[33m",      # yellow
    "ERROR":    "\033[31m",      # red
    "CRITICAL": "\033[1;31m",    # bold red
}
RESET = "\033[0m"


def fmt_line(line: str, source: str) -> str | None:
    """Parse JSON log line → readable string. Return None nếu bị lọc."""
    line = line.strip()
    if not line:
        return None
    try:
        r = json.loads(line)
    except Exception:
        return f"[raw] {line}"

    lvl = r.get("level", "INFO")
    if level_filter and lvl != level_filter:
        return None

    ts      = r.get("ts", "")
    # Lấy HH:MM:SS từ ISO string
    if "T" in ts:
        ts_short = ts.split("T")[1][:8]
    else:
        ts_short = ts[:8]

    msg     = r.get("msg", "")
    session = r.get("session_id", "")
    turn    = r.get("turn", "")
    stt_s   = r.get("stt_s", "")
    api_s   = r.get("api_s", "")
    tts_s   = r.get("tts_s", "")
    event   = r.get("event", "")
    question = r.get("question", "")
    mode    = r.get("mode", "")

    # Build extra string
    extras = []
    if session:  extras.append(f"sid={session}")
    if turn:     extras.append(f"turn={turn}")
    if mode:     extras.append(f"mode={mode}")
    if stt_s:    extras.append(f"stt={stt_s}s")
    if api_s:    extras.append(f"api={api_s}s")
    if tts_s:    extras.append(f"tts={tts_s}s")
    if event:    extras.append(f"event={event}")
    if question: extras.append(f"q={question[:40]}")

    extra_str = "  " + "  ".join(extras) if extras else ""
    src_tag   = f"[{source[:3].upper()}]"
    color     = COLORS.get(lvl, "")

    return f"{color}{ts_short} {src_tag} {lvl[:4]} │ {msg}{extra_str}{RESET}"


def tail_file(path: Path, source: str):
    """Đọc file kiểu tail -f, in ra stdout."""
    # Tạo file nếu chưa có để tránh lỗi
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)

    with open(path, "r", encoding="utf-8") as f:
        f.seek(0, 2)   # nhảy xuống cuối
        while True:
            line = f.readline()
            if not line:
                time.sleep(0.15)
                continue
            output = fmt_line(line, source)
            if output:
                print(output, flush=True)


# ── Entry point ──────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\033[36mPhysBot Log Watcher\033[0m", flush=True)
    print(f"  Files  : { {k: str(v) for k, v in LOG_FILES.items()} }", flush=True)
    print(f"  Filter : level={level_filter or 'ALL'}  source={source_filter or 'ALL'}", flush=True)
    print(f"  Ctrl+C để thoát\n", flush=True)

    threads = []
    for src, path in LOG_FILES.items():
        if source_filter and src.upper() != source_filter:
            continue
        t = threading.Thread(target=tail_file, args=(path, src), daemon=True)
        t.start()
        threads.append(t)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nWatcher thoát.")
