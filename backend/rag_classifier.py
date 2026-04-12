"""
backend/rag_classifier.py
─────────────────────────
Phân loại dòng RAG context theo ngữ nghĩa vật lý THPT 10-11-12.

Nhãn trả về:
  FORMULA          – công thức, hệ thức, định luật dạng ký hiệu
  DEFINITION       – định nghĩa khái niệm
  CONSTANT         – hằng số vật lý / bảng giá trị chuẩn
  UNIT             – đổi đơn vị, ký hiệu đơn vị
  CONDITION        – điều kiện áp dụng, giới hạn, chú ý
  EXAMPLE_PROBLEM  – đầu bài bài toán có số liệu cụ thể
  SOLUTION_STEP    – bước giải, thay số, tính toán
  ANSWER           – đáp số, kết quả cuối
  CHAPTER_HEADER   – tiêu đề chương / bài học
  OTHER            – không thuộc nhãn nào trên
"""

import re


# ══════════════════════════════════════════════════════════════════
# FORMULA
# ══════════════════════════════════════════════════════════════════

_FORMULA_SYMBOL_PATTERNS = [
    # ── Lớp 10: Cơ học ───────────────────────────────────────────
    r'\bv\s*=\s*v[₀0]',
    r'\bv[²2]\s*=',
    r'\bs\s*=\s*v[₀0]',
    r'\ba\s*=\s*[ΔdF\(]',
    r'\bF\s*=\s*m\s*[a\*×]',
    r'\bF\s*=\s*[kμ]\w*\s*[nN\*]',
    r'\bP\s*=\s*m\s*[gv\*×]',
    r'\bW\s*=\s*[FmP\d]',
    r'\bE[kđ]\s*=',
    r'\bE[pt]\s*=',
    r'\bT\s*=\s*2\s*[πp\*]',
    r'\bf\s*=\s*1\s*/\s*T',
    r'\bω\s*=\s*[2v\d]',
    r'\bx\s*=\s*A\s*[cs]',
    r'\bτ\s*=\s*[FrI]',
    r'\bM\s*=\s*F\s*[dr\.]',
    r'\bI\s*=\s*[mr\d].*[r²2]',
    r'\bp\s*=\s*m\s*[v\*×]',
    r'\bJ\s*=\s*[FΔ]',
    r'\bL\s*=\s*[Iωmr]',
    # ── Lớp 10: Nhiệt động lực học ───────────────────────────────
    r'\bQ\s*=\s*[mcΔnR]',
    r'\bΔU\s*=',
    r'\bη\s*=\s*[WAQT\d]',
    r'\bpV\s*=\s*nRT',
    r'\bp[₁1]\s*V[₁1]\s*/\s*T[₁1]',
    # ── Lớp 11: Điện học ─────────────────────────────────────────
    r'\bF\s*=\s*k\s*[q\|]',
    r'\bE\s*=\s*k\s*[qQ]',
    r'\bE\s*=\s*[FU]\s*/\s*[qd]',
    r'\bV\s*=\s*k\s*[qQ]',
    r'\bU\s*=\s*[VEI]\w*\s*[\-\+\*]',
    r'\bC\s*=\s*[qQε]',
    r'\bW\s*=\s*½\s*C',
    r'\bI\s*=\s*[qΔne]\w*\s*[/\*]',
    r'\bR\s*=\s*[ρUI]',
    r'\bρ\s*=\s*R\s*S',
    r'\bP\s*=\s*[UI²R].*\b(?:I|R|U)\b',
    r'\bξ\s*=',
    r'\bB\s*=\s*[μIk]',
    r'\bF\s*=\s*[qBI].*[vIl]',
    r'\bΦ\s*=\s*[BNS]',
    r'\be\s*=\s*[-NBω]',
    r'\bL\s*=\s*[μNn].*[²2]',
    # ── Lớp 12: Dao động – Sóng ──────────────────────────────────
    r'\bλ\s*=\s*[cvf]',
    r'\bn\s*=\s*[cv]\s*/',
    r'\b1\s*/\s*f\s*=',
    r'\bk\s*=\s*d[\'`′]',
    r'\bD\s*=\s*1\s*/\s*f',
    r'\bi\s*=\s*λ[Dl]',
    r'\bx[kns]?\s*=.*λ',
    r'\bsin\s*[iαθ]\s*=\s*n',
    r'\bn[₁1]\s*sin',
    # ── Lớp 12: Lượng tử ─────────────────────────────────────────
    r'\bE\s*=\s*h\s*[fν]',
    r'\bE\s*=\s*h\s*c\s*/',
    r'\bhf\s*=\s*A',
    r'\beU\s*[hb]\s*=',
    r'\bλ\s*=\s*h\s*/\s*[pmv]',
    r'\bE[n]\s*=\s*-\s*13',
    r'\bhf\s*=\s*E[nm]',
    # ── Lớp 12: Hạt nhân ─────────────────────────────────────────
    r'\bΔm\s*=',
    r'\bE[lk]\s*=\s*Δ',
    r'\bE\s*=\s*Δm\s*c[²2]',
    r'\bN\s*=\s*N[₀0]\s*[e\*]',
    r'\bT\s*=\s*ln\s*2',
    r'\bλ\s*=\s*ln\s*2\s*/',
    r'\bA\s*=\s*λ\s*N',
    # ── Máy biến áp / Mạch AC ────────────────────────────────────
    r'\bU[₁1]\s*/\s*U[₂2]\s*=\s*N',
    r'\bI[₁1]\s*/\s*I[₂2]\s*=\s*N',
    r'\bZ[LC]\s*=',
    r'\bZ\s*=\s*sqrt\b',
    r'\bZ\s*=\s*√',
    r'\btan\s*φ\s*=',
    r'\bω[₀0]?\s*=\s*1\s*/\s*√',
    r'\bT\s*=\s*2.*√.*LC',
    # ── Tổng quát ─────────────────────────────────────────────────
    r'[A-Za-zÀ-ỹ_][A-Za-zÀ-ỹ₀₁₂_\s]{0,8}\s*=\s*[A-Za-zÀ-ỹ\d\(√½⅓πμΔ]',
]

_FORMULA_KEYWORD_PATTERN = re.compile(
    r'(?:công thức|hệ thức|biểu thức|định luật|phương trình\s+\w|'
    r'nguyên lý|bảo toàn\s+\w|định lý|quy tắc\s+(?:bàn tay|vặn|nắm))',
    re.IGNORECASE,
)

_RE_FORMULA = re.compile(
    '|'.join(f'(?:{p})' for p in _FORMULA_SYMBOL_PATTERNS),
    re.IGNORECASE,
)


# ══════════════════════════════════════════════════════════════════
# DEFINITION
# ══════════════════════════════════════════════════════════════════

_RE_DEFINITION = re.compile(
    r'(?:'
    r'là\s+(?:đại lượng|lực|khả năng|quá trình|hiện tượng|số đo|'
    r'tác dụng|tổng|hiệu|tích|thước đo|đặc trưng|năng lượng|'
    r'vectơ|đại lượng vô hướng|đại lượng vectơ|công|công suất|'
    r'điện tích|điện thế|điện trường|từ trường|từ thông|áp suất|'
    r'phép đo|chiết suất|hiệu suất)'
    r'|được\s+(?:định nghĩa|hiểu|xác định)\s+là'
    r'|đặc trưng\s+cho\s+(?:tính chất|mức độ|khả năng|độ mạnh|hướng)'
    r'|gọi là\s+(?:đại lượng|lực|vận tốc|gia tốc|điện trường|từ trường|'
    r'năng lượng|công suất|điện thế|cường độ|từ thông|áp suất|'
    r'chiết suất|hiệu suất|chu kỳ|tần số|bước sóng|biên độ)'
    r'|đo\s+bằng\s+(?:đơn vị|[A-ZĐÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ])'
    r')',
    re.IGNORECASE,
)


# ══════════════════════════════════════════════════════════════════
# CONSTANT
# ══════════════════════════════════════════════════════════════════

_RE_CONSTANT = re.compile(
    r'(?:'
    r'g\s*[=≈]\s*(?:9[.,]\d|10)\s*(?:m|$)'
    r'|G\s*=\s*6[.,]\d+\s*[×x\*e]'
    r'|k\s*[=≈]\s*9\s*[×x\*]\s*10'
    r'|c\s*[=≈]\s*3\s*[×x\*]\s*10[\^⁸]?8?(?:\s|$|[,;Jm])'
    r'|h\s*[=≈]\s*6[.,]\d+\s*[×x\*]\s*10'
    r'|e\s*[=≈]\s*1[.,]6\s*[×x\*]\s*10'
    r'|ε[₀0]\s*[=≈]\s*8[.,]\d+\s*[×x\*]\s*10'
    r'|μ[₀0]\s*[=≈]\s*4\s*[πp×x\*]'
    r'|N[_A]?\s*[=≈]\s*6[.,]\d+\s*[×x\*]\s*10'
    r'|k[_B]?\s*[=≈]\s*1[.,]38\s*[×x\*]\s*10'
    r'|m[_e]?\s*[=≈]\s*9[.,]\d+\s*[×x\*]\s*10'
    r'|m[_p]?\s*[=≈]\s*1[.,]67\s*[×x\*]\s*10'
    r'|1\s*u\s*[=≈]\s*(?:1[.,]66\s*[×x\*]\s*10|931)'
    r'|R\s*[=≈]\s*8[.,]3\d+\s*J'
    r'|(?:chiết suất|khối lượng riêng|nhiệt dung riêng|'
    r'nhiệt nóng chảy|nhiệt hóa hơi|điện trở suất|hệ số ma sát)\s+'
    r'(?:của\s+)?[\wÀ-ỹ]+\s*[=≈]\s*\d'
    r')',
    re.IGNORECASE,
)


# ══════════════════════════════════════════════════════════════════
# UNIT
# ══════════════════════════════════════════════════════════════════

_RE_UNIT = re.compile(
    r'(?:'
    r'1\s*(?:km|m|cm|mm|nm|μm|Å)\s*=\s*\d'
    r'|1\s*(?:kJ|MJ|GJ|kcal|cal|kWh|Wh)\s*=\s*\d'
    r'|1\s*(?:eV|keV|MeV|GeV)\s*=\s*\d'
    r'|1\s*(?:atm|bar|mmHg|torr)\s*=\s*\d'
    r'|1\s*(?:kg|g|mg)\s*=\s*\d'
    r'|1\s*u\s*=\s*\d'
    r'|1\s*(?:min|h|ngày|năm)\s*=\s*\d'
    r'|\[(?:m|kg|s|A|K|mol|cd|'
    r'N|J|W|V|C|T|F|H|Pa|Ω|Hz|Wb|Bq|'
    r'eV|MeV|rad|'
    r'm/s|m/s²|kg/m³|rad/s|rad/s²|'
    r'N/m|N/C|V/m|A/m)\]'
    r'|đơn vị\s+(?:của\s+)?(?:\w+\s+)?là\s+[A-Za-zΩ\[/²³]'
    r'|ký hiệu\s+(?:đơn vị\s+)?là\s+[A-Za-zΩ\[]'
    r')',
    re.IGNORECASE,
)


# ══════════════════════════════════════════════════════════════════
# CONDITION
# ══════════════════════════════════════════════════════════════════

_RE_CONDITION = re.compile(
    r'(?:'
    r'khi\s+(?:vật|điện|lực|v\s*=|a\s*=|t\s*=|bỏ qua|không có|chuyển động)'
    r'|điều kiện\s+(?:để|áp dụng|có|xảy ra)'
    r'|chỉ\s+(?:đúng|áp dụng|có giá trị)\s+khi'
    r'|lưu ý\s*[:\-]'
    r'|chú ý\s*[:\-]'
    r'|không\s+(?:áp dụng|đúng|xét)\s+khi'
    r'|giới hạn\s+(?:của|áp dụng|sử dụng)'
    r'|với\s+(?:điều kiện|giả thiết)'
    r'|trong\s+trường hợp\s+(?:này|đặc biệt|tổng quát)'
    r'|nếu\s+(?:bỏ qua|không có|có thêm|vật|điện)'
    r'|giả sử\s+(?:bỏ qua|không|vật)'
    r')',
    re.IGNORECASE,
)


# ══════════════════════════════════════════════════════════════════
# EXAMPLE_PROBLEM
# ══════════════════════════════════════════════════════════════════

_RE_EXAMPLE_PROBLEM = re.compile(
    r'(?:'
    r'(?:một|hai|ba)\s+(?:'
    r'vật|điện tích|mạch|con lắc|lò xo|hạt|electron|proton|'
    r'photon|tụ điện|cuộn dây|điện trở|nguồn điện|'
    r'gương|thấu kính|sóng|hạt nhân|'
    r'xe|ô tô|tàu|quả bóng|viên bi|vật nặng|người|'
    r'khối|thanh|tấm|dây|ống|bình|bể|bóng đèn|động cơ'
    r')'
    r'|cho\s+[a-zÀ-ỹ]\s*=\s*\d'
    r'|đề bài\s*[:\-]?'
    r'|dữ kiện\s*[:\-]?'
    r'|câu\s+\d+\s*[:\.\-]\s*(?:một|hai|ba|cho|tính|hỏi|điện|vật|hạt|xe|'
    r'người|quả|viên|thanh|nguồn|mạch|tụ|cuộn|dây)'
    r'|bài\s+\d+\s*[:\.\-]\s*(?:tính|hỏi|cho|một|hai|xác định|tìm)'
    r'|ví dụ\s+\d*\s*[:\.\-]?\s*(?:một|cho|tính|xét)'
    r')',
    re.IGNORECASE,
)


# ══════════════════════════════════════════════════════════════════
# SOLUTION_STEP
# ══════════════════════════════════════════════════════════════════

_RE_SOLUTION = re.compile(
    r'(?:'
    r'^(?:giải|lời giải)\s*[:\-]?'
    r'|áp dụng\s+(?:công thức|định luật|hệ thức|biểu thức)'
    r'|thay\s+(?:số|vào|các giá trị|vào công thức)'
    r'|ta\s+(?:có|được|tính được|thay|áp dụng|suy ra)'
    r'|suy\s+ra\s*[:\-]?'
    r'|vậy\s+(?:lực|vận tốc|gia tốc|công|năng lượng|điện tích|'
    r'kết quả|chu kỳ|tần số|điện trở|nhiệt lượng)'
    r'|từ\s+(?:\(\d+\)|công thức|hệ thức|phương trình)\s+(?:và|suy|ta)'
    r'|bước\s+\d+\s*[:\-]?'
    r'|tính\s+(?:được|ra|thấy)\s*[:\-]?'
    r'|=>\s*[A-Za-zÀ-ỹ\d]'
    r'|→\s*[A-Za-zÀ-ỹ\d]'
    r'|⇒\s*[A-Za-zÀ-ỹ\d]'
    r'|[A-Za-zÀ-ỹ_][A-Za-zÀ-ỹ_₀₁₂\s]{0,6}\s*=\s*[\w\d\.\(\)\/\*]+\s*=\s*\d'
    r')',
    re.IGNORECASE | re.MULTILINE,
)


# ══════════════════════════════════════════════════════════════════
# ANSWER
# ══════════════════════════════════════════════════════════════════

_RE_ANSWER = re.compile(
    r'(?:'
    r'(?:đáp số|kết quả|đáp án)\s*[:\-]?\s*\S'
    r'|(?:vậy|do đó|kết luận)\s*[,:]?\s*'
    r'(?:lực|vận tốc|gia tốc|công|năng lượng|điện tích|'
    r'điện thế|cường độ|từ thông|chu kỳ|tần số|bước sóng|'
    r'nhiệt lượng|công suất|hiệu suất|khối lượng|thời gian|'
    r'điện trở|hiệu điện thế|quãng đường|độ cao|bán kính)'
    r'\s*(?:bằng|là|=)\s*\d'
    r'|chọn\s+(?:đáp án|đáp|câu)\s*[:\.]?\s*[ABCD]'
    r'|(?:đáp án|câu trả lời)\s+đúng\s+là\s+[ABCD]'
    r'|kết quả\s+là\s+[ABCD\d]'
    r'|(?:A|B|C|D)\.\s*\d'
    r'|(?:A|B|C|D)\)\s*\d'
    r')',
    re.IGNORECASE,
)


# ══════════════════════════════════════════════════════════════════
# CHAPTER_HEADER
# ══════════════════════════════════════════════════════════════════

_RE_HEADER = re.compile(
    r'(?:'
    r'^(?:chương|phần|chủ đề)\s+\d+'
    r'|^bài\s+\d+\s*[:\.\-]\s*(?!(?:tính|hỏi|cho|một|hai|tìm|xác định)\b)'
    r'(?:[A-ZĐÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ]|\w{5,})'
    r'|^#+\s+'
    r'|^(?:I{1,3}|IV|V{1,3}|IX|X)\.\s+'
    r'|^={3,}'
    r'|^─{3,}'
    r'|^━{3,}'
    r'|^-{5,}'
    r')',
    re.IGNORECASE | re.MULTILINE,
)


# ══════════════════════════════════════════════════════════════════
# PIPELINE — thứ tự quan trọng
# ══════════════════════════════════════════════════════════════════

_CLASSIFIER_PIPELINE = [
    ("CHAPTER_HEADER",  _RE_HEADER),
    ("ANSWER",          _RE_ANSWER),
    ("SOLUTION_STEP",   _RE_SOLUTION),
    ("EXAMPLE_PROBLEM", _RE_EXAMPLE_PROBLEM),
    ("CONSTANT",        _RE_CONSTANT),
    ("UNIT",            _RE_UNIT),
    ("CONDITION",       _RE_CONDITION),
    ("FORMULA",         _RE_FORMULA),
    ("FORMULA",         _FORMULA_KEYWORD_PATTERN),
    ("DEFINITION",      _RE_DEFINITION),
]


def classify_line(line: str) -> str:
    """
    Phân loại một dòng RAG context thành nhãn ngữ nghĩa.

    Returns:
        FORMULA | DEFINITION | CONSTANT | UNIT | CONDITION |
        EXAMPLE_PROBLEM | SOLUTION_STEP | ANSWER | CHAPTER_HEADER | OTHER
    """
    stripped = line.strip()
    if not stripped:
        return "OTHER"
    for label, pattern in _CLASSIFIER_PIPELINE:
        if pattern.search(stripped):
            return label
    return "OTHER"


# ══════════════════════════════════════════════════════════════════
# PRESETS & EXTRACT
# ══════════════════════════════════════════════════════════════════

PRESET_FORMULA_ONLY  = {"FORMULA", "CONSTANT", "DEFINITION", "CONDITION", "UNIT"}
PRESET_FULL_THEORY   = {"FORMULA", "CONSTANT", "DEFINITION", "CONDITION", "UNIT", "CHAPTER_HEADER"}
PRESET_EXAM_REVIEW   = {"FORMULA", "CONSTANT", "EXAMPLE_PROBLEM", "ANSWER"}
PRESET_SOLUTION_ONLY = {"SOLUTION_STEP", "ANSWER", "FORMULA"}


def extract_for_rag(
    context: str,
    include: set | None = None,
    fallback_chars: int = 500,
) -> str:
    """
    Lọc context RAG theo nhãn, sẵn sàng nhúng vào prompt.

    Args:
        context       : raw text từ vector DB
        include       : tập nhãn muốn giữ (None → PRESET_FORMULA_ONLY)
        fallback_chars: fallback nếu kết quả < 80 ký tự

    Returns:
        str — context đã lọc
    """
    if include is None:
        include = PRESET_FORMULA_ONLY

    kept = [
        line.strip()
        for line in context.split("\n")
        if line.strip() and classify_line(line.strip()) in include
    ]
    result = "\n".join(kept)
    return result if len(result) >= 80 else context[:fallback_chars]


def extract_with_labels(context: str) -> list[tuple[str, str]]:
    """Trả về list (label, line) cho toàn bộ context — dùng debug."""
    return [
        (classify_line(line.strip()), line.strip())
        for line in context.split("\n")
        if line.strip()
    ]