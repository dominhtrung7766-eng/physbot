"""
backend/text_correction.py
──────────────────────────
Sửa lỗi nhận dạng giọng nói tiếng Việt cho thuật ngữ vật lý THPT.
Hybrid approach:
  Bước 1 — Regex word-boundary: fix lỗi cố định, KHÔNG đụng từ đúng
  Bước 2 — LLM: chỉ sửa lỗi ngữ nghĩa/ngữ cảnh còn lại
"""

import re
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
_groq: Groq | None = None

def _get_groq() -> Groq:
    global _groq
    if _groq is None:
        _groq = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq


# ══════════════════════════════════════════════════════════════════
# BƯỚC 1 — REGEX RULES
#
# Dùng re.sub với word-boundary (\b) để KHÔNG đụng vào từ đúng
# bên trong từ khác.
#
# Format: (pattern_regex, replacement)
# Thứ tự: cụm DÀI trước, từ đơn sau — tránh partial match.
#
# CHUẨN ĐƠN VỊ: theo prompts.py TTS tiếng Việt
#   C (culông) → "culông"  KHÔNG phải "Coulomb"
#   N (niutơn) → "niutơn"  KHÔNG phải "Newton"
#   J (jun)    → "jun"
# ══════════════════════════════════════════════════════════════════

_RULES: list[tuple[str, str]] = [
    # ── COMPOUND UNITS — ĐẶT TRƯỚC rule đơn lẻ ──────────────────────
    (r"\bN/C\b",    "niutơn trên culông"),
    (r"\bN/m²\b",   "niutơn trên mét vuông"),
    (r"\bN/m2\b",   "niutơn trên mét vuông"),
    (r"\bN/m\b",    "niutơn trên mét"),
    (r"\bJ/kg\b",   "jun trên kilôgam"),
    (r"\bJ/mol\b",  "jun trên mol"),
    (r"\bW/m²\b",   "oát trên mét vuông"),
    (r"\bkJ/kg\b",  "kilôjun trên kilôgam"),
    (r"\bV/m\b",    "vôn trên mét"),
    (r"\bA/m\b",    "ampe trên mét"),
    (r"\bkg/m³\b",  "kilôgam trên mét khối"),
    (r"\bkg/m3\b",  "kilôgam trên mét khối"),
    (r"\bC/m²\b",   "culông trên mét vuông"),
    (r"\bC/m2\b",   "culông trên mét vuông"),
    (r"\bΩ\.m\b",   "ôm nhân mét"),
    (r"\bΩm\b",     "ôm nhân mét"),
    (r"\bm/s²\b",   "mét trên giây bình phương"),
    (r"\bm/s2\b",   "mét trên giây bình phương"),

    # ── SỬA KÝ HIỆU TOÁN HỌC (LLM vẫn viết) ──────────────────────────
    (r"(?<![a-zA-Z0-9])=(?![a-zA-Z0-9])",   " bằng "),
    (r"×",                                   " nhân "),
    (r"\*",                                  " nhân "),
    (r"(?<![a-zA-Z0-9])/(?![a-zA-Z0-9])",    " chia "),
    (r"\+",                                  " cộng "),
    (r"(?<![a-zA-Z0-9])-(?![a-zA-Z0-9])",    " trừ "),

    # ── SỬA ĐƠN VỊ (ưu tiên trước) ────────────────────────────────────
    (r"m/s²",                   "mét trên giây bình phương"),
    (r"m/s2",                   "mét trên giây bình phương"),
    (r"km/h",                   "kilômét trên giờ"),
    (r"\bkg\b",                 "kilôgam"),
    (r"\bN\b",                  "niutơn"),
    (r"\bJ\b",                  "jun"),
    (r"\bW\b",                  "oát"),
    (r"\bHz\b",                 "héc"),
    (r"\bPa\b",                 "pascal"),
    (r"\bC\b",                  "culông"),
    (r"\bV\b",                  "vôn"),
    (r"\bΩ\b",                  "ôm"),
    (r"\bT\b",                  "tesla"),
    (r"\bmN\b",     "mili niutơn"),
    (r"\bkN\b",     "kilô niutơn"),
    (r"\bμN\b",     "micrô niutơn"),
    (r"\bnC\b",     "nano culông"),
    (r"\bμC\b",     "micrô culông"),
    (r"\bmC\b",     "mili culông"),
    (r"\bkV\b",     "kilô vôn"),
    (r"\bmV\b",     "mili vôn"),
    (r"\bμV\b",     "micrô vôn"),
    (r"\bkΩ\b",     "kilô ôm"),
    (r"\bMΩ\b",     "mêga ôm"),
    (r"\bμF\b",     "micrô fara"),
    (r"\bnF\b",     "nano fara"),
    (r"\bpF\b",     "picô fara"),
    (r"\bmH\b",     "mili henry"),
    (r"\bμH\b",     "micrô henry"),
    (r"\bkHz\b",    "kilô héc"),
    (r"\bMHz\b",    "mêga héc"),
    (r"\bGHz\b",    "giga héc"),

    # ── SỬA LŨY THỪA DẠNG CHỮ (x², x³) ──────────────────────────────
    (r"([a-zA-Z0-9]+)²",        r"\1 bình phương"),
    (r"([a-zA-Z0-9]+)³",        r"\1 lập phương"),

    # ── SỬA CĂN BẬC HAI ──────────────────────────────────────────────
    (r"√([a-zA-Z0-9]+)",        r"căn bậc hai của \1"),
    (r"sqrt\(([^)]+)\)",        r"căn bậc hai của \1"),

    # ── SỬA KÝ HIỆU HY LẠP ───────────────────────────────────────────
    (r"π",                      "pi"),
    (r"λ",                      "lăm-đa"),
    (r"ω",                      "ô-mê-ga"),
    (r"Δ",                      "biến thiên"),
    (r"(?<![a-zA-Z])μ(?![a-zA-Z])", "muy"),

    # ── LŨY THỪA DẠNG ^ ────────────────────────────────────────────
    (r"10\^-12",            "mười mũ trừ mười hai"),
    (r"10\^-9",             "mười mũ trừ chín"),
    (r"10\^-6",             "mười mũ trừ sáu"),
    (r"10\^-3",             "mười mũ trừ ba"),
    (r"10\^3",              "mười mũ ba"),
    (r"10\^6",              "mười mũ sáu"),
    (r"10\^9",              "mười mũ chín"),
    (r"\bx\^2\b",           "x bình phương"),
    (r"\bx\^3\b",           "x lập phương"),
    (r"\bv\^2\b",           "v bình phương"),
    (r"\br\^2\b",           "r bình phương"),

    # ── ĐƠN VỊ / PHÉP ĐO — cụm dài trước ──────────────────────────
    (r"\bmét trên dây bình\b",      "mét trên giây bình phương"),
    (r"\bmét trên dây vuông\b",     "mét trên giây bình phương"),
    (r"\bmét trên dây\b",           "mét trên giây"),
    (r"\bkm trên dây\b",            "km trên giây"),
    (r"(?<!\d\s)(?<!\d)(?<![a-zA-Z])\btrên dây bình\b",  "trên giây bình phương"),
    (r"(?<!\d\s)(?<!\d)(?<![a-zA-Z])\btrên dây vuông\b", "trên giây bình phương"),
    (r"(?<!\d\s)(?<!\d)(?<![a-zA-Z])\btrên dây\b",       "trên giây"),
    (r"\bki lô gam\b",              "kilôgam"),
    (r"\bki lo gam\b",              "kilôgam"),
    (r"\bki lô mét\b",              "kilômét"),
    (r"\bmicro cu lông\b",          "micrôculông"),
    (r"\bmicro cu lom\b",           "micrôculông"),
    (r"\bnano cu lông\b",           "nanoculông"),
    (r"\bnano cu lon\b",            "nanoculông"),
    (r"\bmili am pe\b",             "miliampe"),
    (r"\bkilo ôm\b",                "kilôôm"),
    (r"\bmê ga ôm\b",               "mêgaôm"),

    # ── TÊN NHÀ KHOA HỌC / ĐỊNH LUẬT / ĐƠN VỊ TTS CHUẨN ──────────
    (r"\bcu lôm\b",                 "culông"),
    (r"\bcu lom\b",                 "culông"),
    (r"\bcu lông\b",                "culông"),
    (r"\bniu tơn\b",                "niutơn"),
    (r"\bniu ton\b",                "niutơn"),
    (r"\bam pe\b",                  "ampe"),
    (r"\bpát can\b",                "pascal"),
    (r"\bpa can\b",                 "pascal"),
    (r"\bhen ri\b",                 "henry"),
    (r"\bte la\b",                  "tesla"),
    (r"\bfa ra\b",                  "fara"),
    (r"\bđốp lơ\b",                 "Doppler"),
    (r"\bđốp le\b",                 "Doppler"),
    (r"\bai n xtanh\b",             "Einstein"),
    (r"\banh xtanh\b",              "Einstein"),

    # ── SQRT — TIẾNG ANH BỊ GIỮ NGUYÊN ────────────────────────────
    (r"\bsqrt\b",                   "căn bậc hai"),

    # ── THUẬT NGỮ ĐỘNG HỌC ─────────────────────────────────────────
    (r"\bnem ngang\b",              "ném ngang"),
    (r"\bnem xiên\b",               "ném xiên"),
    (r"\bđọng học\b",               "động học"),
    (r"\bđọng lực\b",               "động lực"),
    (r"\bmặt phẳng nghiên\b",       "mặt phẳng nghiêng"),
    (r"\bmặt phẳng ngan\b",         "mặt phẳng ngang"),
    (r"\bquán đường\b",             "quãng đường"),
    (r"\bgia tốt\b",                "gia tốc"),
    (r"\bvận tốt\b",                "vận tốc"),
    (r"\btần sổ\b",                 "tần số"),
    (r"\bbước xóng\b",              "bước sóng"),
    (r"\bsống ngang\b",             "sóng ngang"),
    (r"\bsống dọc\b",               "sóng dọc"),
    (r"\bsống âm\b",                "sóng âm"),
    (r"\bsống điện từ\b",           "sóng điện từ"),
    (r"\bđao động\b",               "dao động"),
    (r"\bcon lắt lò xo\b",          "con lắc lò xo"),
    (r"\bcon lắt\b",                "con lắc"),

    # ── LỰC / MA SÁT ───────────────────────────────────────────────
    (r"\bhệ số ma fast\b",          "hệ số ma sát"),
    (r"\bhệ số ma fát\b",           "hệ số ma sát"),
    (r"\bma fast\b",                "ma sát"),
    (r"\bma fát\b",                 "ma sát"),
    (r"\bma xát\b",                 "ma sát"),
    (r"\bmassat\b",                 "ma sát"),

    # ── ĐIỆN HỌC ────────────────────────────────────────────────────
    (r"\bhiệu điện thể\b",          "hiệu điện thế"),
    (r"\bđiện thể\b",               "điện thế"),
    (r"\bmạch r l c\b",             "mạch RLC"),
    (r"\bmạch rlc\b",               "mạch RLC"),

    # ── SỐ LIỆU THƯỜNG GẶP ─────────────────────────────────────────
    (r"\bg bằng 10\b",  "g bằng 10 mét trên giây bình phương"),
    (r"\bg bằng 9,8\b", "g bằng 9 phẩy 8 mét trên giây bình phương"),
    (r"\bk bằng 9\b",   "k bằng 9 nhân mười mũ chín"),

    # ── TIẾNG ANH KHÁC BỊ GIỮ NGUYÊN ──────────────────────────────
    (r"\bdelta\b",      "biến thiên"),
    (r"\bomega\b",      "tần số góc"),
    (r"\blambda\b",     "bước sóng"),
    (r"\bepsilon\b",    "suất điện động"),
    (r"\bgamma\b",      "tia gamma"),
    (r"\btheta\b",      "góc theta"),
]

_COMPILED: list[tuple[re.Pattern, str]] = [
    (re.compile(p, re.IGNORECASE), r) for p, r in _RULES
]


# ── FIX VẤN ĐỀ 3: STRICT LLM PROMPT ───────────────────────────
# LLM không được thêm nội dung mới, đặc biệt KHÔNG được điền đáp án
# vào câu hỏi bỏ lửng ("gia tốc rơi tự do có giá trị là...")
_LLM_PROMPT = """Bạn là trợ lý sửa lỗi nhận dạng giọng nói tiếng Việt. Nhiệm vụ DUY NHẤT: sửa lỗi phiên âm sai.

QUY TẮC TUYỆT ĐỐI — VI PHẠM LÀ SAI:
1. CHỈ sửa từ bị nhận dạng SAI do âm thanh gần giống (vd: "ma fast" → "ma sát")
2. KHÔNG thêm bất kỳ từ/số/thông tin nào không có trong input
3. KHÔNG hoàn thành câu bị bỏ lửng — nếu câu kết thúc bằng "..." hoặc bỏ dở, GIỮ NGUYÊN y chang
4. KHÔNG sửa từ đã đúng
5. Trả về ĐÚNG 1 dòng, không giải thích

QUAN TRỌNG: Nếu câu kết thúc bằng "...", "là", "bằng" mà không có giá trị → GIỮ NGUYÊN phần đó, KHÔNG điền vào.

VÍ DỤ ĐÚNG:
  Input:  "Gia tốc rơi tự do gần mặt đất có giá trị là..."
  Output: "Gia tốc rơi tự do gần mặt đất có giá trị là..."
  (GIỮ NGUYÊN — TUYỆT ĐỐI không thêm "9,8 m/s^2" hay bất cứ thứ gì)

  Input:  "Vật có khối lượng 2kg trượt trên mặt phẳng ngan"
  Output: "Vật có khối lượng 2kg trượt trên mặt phẳng ngang"

VÍ DỤ SAI (TUYỆT ĐỐI KHÔNG làm):
  Input:  "tính vận tốc đầu 20m trên giây"
  Output: "tính vận tốcc đầu 20m trên giây"  ← thêm ký tự vào từ đúng

  Input:  "Gia tốc rơi tự do có giá trị là..."
  Output: "Gia tốc rơi tự do có giá trị là 9,8 m/s^2."  ← TUYỆT ĐỐI CẤM thêm nội dung"""


# ══════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════

def correct_physics_text(text: str, use_llm: bool = True) -> str:
    """
    Sửa lỗi nhận dạng giọng nói cho văn bản vật lý.

    Args:
        text:    Chuỗi text thô từ Whisper STT
        use_llm: True = dùng LLM fallback sau regex (mặc định)
                 False = chỉ dùng regex (nhanh hơn, offline)

    Returns:
        Chuỗi đã được sửa
    """
    if not text or not text.strip():
        return text

    # ── Bước 1: Regex rules ─────────────────────────────────────
    result = text
    for pattern, replacement in _COMPILED:
        result = pattern.sub(replacement, result)

    if result:
        result = result[0].upper() + result[1:]

    # ── Bước 2: LLM fallback ────────────────────────────────────
    if use_llm:
        result = _llm_correct(result)

    return result


def _llm_correct(text: str) -> str:
    """Gọi LLM để sửa lỗi ngữ nghĩa/ngữ cảnh còn lại sau regex."""
    try:
        r = _get_groq().chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": _LLM_PROMPT},
                {"role": "user",   "content": text},
            ],
            max_tokens=300,
            temperature=0.0,
        )
        corrected = r.choices[0].message.content.strip()

        # FIX VẤN ĐỀ 3: Reject nếu LLM thêm nội dung mới
        # Threshold +20 ký tự: đủ cho sửa lỗi nhỏ, không đủ để thêm câu mới
        if len(corrected) > len(text) + 20:
            print(f"[text_correction] LLM bị reject — thêm nội dung: {len(text)} → {len(corrected)} ký tự")
            return text

        return corrected
    except Exception as e:
        print(f"[text_correction] LLM fallback lỗi: {e}")
        return text


def log_correction(original: str, corrected: str) -> None:
    """In log nếu có thay đổi — dùng để debug và bổ sung rules."""
    if original.strip().lower() != corrected.strip().lower():
        print(f"[correction] '{original}' → '{corrected}'")


# ══════════════════════════════════════════════════════════════════
# TEST — python backend/text_correction.py
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    test_cases = [
        (
            "Ném ngang từ độ cao 80m với vận tốc đầu 20m trên dây",
            "Ném ngang từ độ cao 80m với vận tốc đầu 20m trên giây",
        ),
        (
            "Vật có hệ số ma fast bằng 0,3 trên mặt phẳng ngan",
            "Vật có hệ số ma sát bằng 0,3 trên mặt phẳng ngang",
        ),
        (
            "2 điện tích Q1 Q2 đặt cách nhau 3cm tính lực cu lôm",
            "2 điện tích Q1 Q2 đặt cách nhau 3cm tính lực culông",
        ),
        (
            "sqrt của 82,6 bằng 9,1",
            "Căn bậc hai của 82,6 bằng 9,1",
        ),
        (
            "10^-9 cu lông",
            "Mười mũ trừ chín culông",
        ),
        # Fix 3: câu bỏ lửng — KHÔNG được thêm đáp án
        (
            "Gia tốc rơi tự do gần mặt đất có giá trị là...",
            "Gia tốc rơi tự do gần mặt đất có giá trị là...",
        ),
    ]

    print("=" * 60)
    print("TEST TEXT_CORRECTION.PY  (regex only, no LLM)")
    print("=" * 60)
    all_pass = True
    for i, (inp, expected) in enumerate(test_cases, 1):
        result = correct_physics_text(inp, use_llm=False)
        ok = result.strip().lower() == expected.strip().lower()
        status = "✓" if ok else "✗"
        if not ok:
            all_pass = False
        print(f"\n[Test {i}] {status}")
        print(f"  Input   : {inp}")
        print(f"  Output  : {result}")
        print(f"  Expected: {expected}")

    print("\n" + "=" * 60)
    print("KẾT QUẢ:", "TẤT CẢ PASS ✓" if all_pass else "CÓ LỖI ✗")
    print("=" * 60)