"""
test_rag_llm.py — Chạy thẳng pipeline RAG + LLM, không cần mic/TTS
Usage:
    python test_rag_llm.py                     # chạy bộ test cases mặc định
    python test_rag_llm.py "câu hỏi của bạn"  # hỏi 1 câu tùy ý
"""

import sys
import os
import time
import re
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from dotenv import load_dotenv

load_dotenv()
from backend.calculator import safe_eval, _BANNED_KEYWORDS, _SAFE_GLOBALS , _SAFE_FUNCTIONS , _PHYSICS_CONSTANTS, CONSTANTS_HINT 
from backend.prompts import PHYSBOT_SYSTEM_PROMPT, CORRECTION_ADDON, TTS_RULES, VOICE_INPUT_ADDON
from backend.text_correction import correct_physics_text, log_correction
from backend.rag_pipeline import retrieve_context, build_rag_prompt, is_out_of_scope
from groq import Groq

console = Console()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ══════════════════════════════════════════════════════════════════
# SYSTEM PROMPT (giống app.py)
# ══════════════════════════════════════════════════════════════════

FULL_SYSTEM_PROMPT = (
    TTS_RULES
    + PHYSBOT_SYSTEM_PROMPT
    + VOICE_INPUT_ADDON
    + CORRECTION_ADDON
    + """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUY TẮC ĐỘ DÀI — BẮT BUỘC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Câu lý thuyết: TỐI ĐA 100 từ
- Câu tính toán: TỐI ĐA 150 từ
- KHÔNG viết dài dòng, KHÔNG giải thích lan man
- Trả lời đủ ý, súc tích, vào thẳng vấn đề
"""
)

# ══════════════════════════════════════════════════════════════════
# DETECT + COT BUILD (giống app.py)
# ══════════════════════════════════════════════════════════════════

_CALC_KEYWORDS = [
    "tính", "tìm", "bằng bao nhiêu", "bao nhiêu",
    "cho biết", "cho g", "vận tốc", "gia tốc", "quãng đường",
    "thời gian", "lực", "khối lượng", "điện tích", "điện trở",
    "hiệu điện thế", "công suất", "nhiệt lượng", "độ cao",
    "góc", "độ dài", "chu kỳ", "tần số", "bước sóng",
]

def _is_calculation_problem(text: str) -> bool:
    t = text.lower()
    return bool(re.search(r'\d', t)) and any(kw in t for kw in _CALC_KEYWORDS)

def _build_user_message(text: str) -> str:
    if not _is_calculation_problem(text):
        return text
    cot_prefix = (
        "[Hướng dẫn nội bộ — KHÔNG đọc phần này ra loa]: "
        "Đây là bài tập tính toán. "
        "Trước khi trả lời, xác định đúng dạng bài, "
        "dùng đúng công thức SGK, thay số từng bước, kiểm tra đơn vị. "
        "Nếu là ném ngang/xiên: tách 2 phương. "
        "Nếu có ma sát nghiêng: a = g(sinα − μcosα). "
        "Nếu Coulomb: nhân Q1×Q2. "
        "Bắt đầu giải:\n\n"
    )
    return cot_prefix + text

# ══════════════════════════════════════════════════════════════════
# PIPELINE CHÍNH
# ══════════════════════════════════════════════════════════════════

def run_pipeline(raw_input: str, max_retries: int = 3) -> dict:
    result = {}
 
    # 1. Text correction
    t0 = time.time()
    corrected = correct_physics_text(raw_input)
    log_correction(raw_input, corrected)
    result["raw_input"] = raw_input
    result["corrected_input"] = corrected
    result["t_correction"] = time.time() - t0
 
    # 2. Scope check trên raw_input (TRƯỚC correction)
    #    → tránh false positive do correct_physics_text thay đổi từ khóa
    if is_out_of_scope(raw_input):
        console.print("[dim]🚫 Out-of-scope detected trên raw input[/dim]")
        result["context_chars"]    = 0
        result["context_preview"]  = ""
        result["is_calculation"]   = False
        result["t_rag"]            = 0.0
        result["response"]         = "Haha tui chỉ giỏi Vật lý THPT thôi nha, mấy thứ khác tui bó tay!"
        result["t_llm"]            = 0.0
        result["error"]            = None
        return result
 
    # 3. RAG — chạy trên corrected (đã chuẩn hóa đơn vị)
    console.print(f"[green]RAG query: {corrected}[/green]")
    t1 = time.time()
    context = retrieve_context(corrected)
    result["t_rag"] = time.time() - t1
    result["context_chars"]   = len(context) if context else 0
    result["context_preview"] = (context[:300] + "...") if context and len(context) > 300 else (context or "")
 
    if context:
        console.print(f"[dim]📚 RAG: {len(context)} ký tự context[/dim]")
    else:
        console.print("[dim]📚 RAG: không tìm được context[/dim]")
 
    # 4. Build user message (CoT prefix nếu là bài tính toán)
    user_msg = _build_user_message(corrected)
    result["is_calculation"] = _is_calculation_problem(corrected)
    enhanced_msg = build_rag_prompt(user_msg, context) if context else user_msg
 
    # 5. LLM với retry
    t2 = time.time()
    for attempt in range(max_retries):
        try:
            r = groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": FULL_SYSTEM_PROMPT},
                    {"role": "user",   "content": enhanced_msg},
                ],
                max_tokens=250,
                temperature=0.7,
                timeout=45.0
            )
            result["response"] = r.choices[0].message.content
            result["t_llm"]    = time.time() - t2
            result["error"]    = None
            break
        except Exception as e:
            err = str(e).lower()
            if "413" in str(e) or "request too large" in err:
                enhanced_msg = user_msg
                wait = 1
            elif "rate_limit" in err:
                wait = 15
            elif "timeout" in err:
                wait = 1
            else:
                wait = 2
            if attempt < max_retries - 1:
                time.sleep(wait)
            else:
                result["response"] = f"[LỖI] {e}"
                result["t_llm"]    = time.time() - t2
                result["error"]    = str(e)
 
    return result
 

# ══════════════════════════════════════════════════════════════════
# HIỂN THỊ KẾT QUẢ
# ══════════════════════════════════════════════════════════════════

def print_result(result: dict, idx: int = None):
    label = f"Test #{idx}" if idx is not None else "Kết quả"
    console.print()
    console.rule(f"[bold cyan]{label}[/bold cyan]")

    console.print(f"[yellow]📝 Input (raw)  :[/yellow] {result['raw_input']}")
    if result["corrected_input"] != result["raw_input"]:
        console.print(f"[yellow]🔧 Input (fixed):[/yellow] {result['corrected_input']}")

    rag_status = f"[green]✅ {result['context_chars']} ký tự[/green]" if result["context_chars"] else "[red]❌ Không tìm được[/red]"
    console.print(f"[blue]📚 RAG context  :[/blue] {rag_status}")
    if result["context_preview"]:
        console.print(Panel(result["context_preview"], title="RAG Preview", border_style="dim blue", expand=False))

    cot_label = "[green]✅ Có CoT prefix[/green]" if result["is_calculation"] else "[dim]Không (lý thuyết)[/dim]"
    console.print(f"[blue]🧠 CoT          :[/blue] {cot_label}")

    console.print(Panel(result["response"], title="[bold green]PhysBot trả lời[/bold green]", border_style="green"))

    t_total = result["t_correction"] + result["t_rag"] + result["t_llm"]
    console.print(
        f"[dim]⏱  fix={result['t_correction']:.2f}s | RAG={result['t_rag']:.2f}s | LLM={result['t_llm']:.2f}s | Tổng={t_total:.2f}s[/dim]"
    )
    if result.get("error"):
        console.print(f"[red]⚠ Lỗi: {result['error']}[/red]")

# ══════════════════════════════════════════════════════════════════
# TEST CASES MẶC ĐỊNH
# ══════════════════════════════════════════════════════════════════

DEFAULT_TEST_CASES = [
    # Lý thuyết
    "Định luật Ôm là gì?",
    "Lực ma sát là gì? Khi nào xuất hiện?",
    # Bài tính toán → trigger CoT
    " Một xe máy khối lượng 150 kg đang di chuyển trên đường với tốc độ 72 km/h thì bỗng nhiên xe phía trước gặp sự cố và dừng lại đột ngột. Biết khoảng cách giữa hai xe là 50 m. Xác định lực cản tối thiểu để xe máy dừng lại an toàn.",
    "Một quả bóng có khối lượng 400g đang nằm yên trên mặt đất thì bị một cầu thủ đá bằng một lực 320 N. Bỏ qua mọi lực cản của môi trường. Gia tốc mà quả bóng thu được là bao nhiêu m/s2?",
    # Lỗi STT điển hình
    "ma fast trên mặt phẳng nghiên góc 30 độ hệ số 0,2 tính gia tốc",
    # Ngoài phạm vi
    "Cho tui hỏi môn Hóa học nhé",
]

# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    console.print(Panel(
        "[bold cyan]PhysBot — RAG + LLM Test Runner[/bold cyan]\n"
        "Pipeline giống app.py — không mic, không TTS.",
        border_style="cyan"
    ))

    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        console.print(f"\n[bold]Chạy 1 câu:[/bold] {question}")
        result = run_pipeline(question)
        print_result(result)
    else:
        console.print(f"\n[bold]Chạy {len(DEFAULT_TEST_CASES)} test cases...[/bold]")
        summary = []

        for i, question in enumerate(DEFAULT_TEST_CASES, 1):
            console.print(f"\n[dim]--- Test {i}/{len(DEFAULT_TEST_CASES)} ---[/dim]")
            result = run_pipeline(question)
            print_result(result, idx=i)
            summary.append({
                "idx": i,
                "q": question[:55] + "..." if len(question) > 55 else question,
                "rag": f"{result['context_chars']}c" if result["context_chars"] else "❌",
                "cot": "✅" if result["is_calculation"] else "—",
                "llm": f"{result['t_llm']:.1f}s",
                "ok": "✅" if not result.get("error") else "❌",
            })
            if i < len(DEFAULT_TEST_CASES):
                time.sleep(1.5)  # tránh rate limit

        # Bảng tổng kết
        console.print()
        console.rule("[bold]SUMMARY[/bold]")
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("#", width=3)
        table.add_column("Câu hỏi", width=58)
        table.add_column("RAG", width=8)
        table.add_column("CoT", width=5)
        table.add_column("LLM", width=7)
        table.add_column("OK", width=4)
        for s in summary:
            table.add_row(str(s["idx"]), s["q"], s["rag"], s["cot"], s["llm"], s["ok"])
        console.print(table)
        console.print()