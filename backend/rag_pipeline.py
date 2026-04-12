import re
import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = "data/chroma_db"
COLLECTION_NAME = "physbot_sgk"

TOP_K = 5
FINAL_TOP_K = 3
MAX_DISTANCE = 0.8
DEDUP_SIMILARITY = 0.6
TOKEN_BUDGET = 1800
MAX_CHUNK_CHARS = 800

_model = None
_collection = None


def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    return _model


def _get_collection():
    global _collection
    if _collection is None:
        print(f"[RAG] Connecting ChromaDB tại: {DB_DIR}")
        client = chromadb.PersistentClient(path=DB_DIR)
        cols = client.list_collections()
        print(f"[RAG] Collections: {[c.name for c in cols]}")
        _collection = client.get_collection(COLLECTION_NAME)
        print(f"[RAG] Collection '{COLLECTION_NAME}' count: {_collection.count()}")
    return _collection


# ══════════════════════════════════════════════════════════════════
# OUT-OF-SCOPE DETECTION
# Dùng ở app.py trên raw input (TRƯỚC correct_physics_text).
# KHÔNG dùng trong retrieve_context — RAG cứ chạy bình thường.
#
# Logic: whitelist-first.
#   - Có từ khóa vật lý → IN SCOPE (return False)
#   - Có tín hiệu môn khác RÕ RÀNG và không có từ khóa vật lý → OUT (return True)
#   - Còn lại → IN SCOPE (safe default)
# ══════════════════════════════════════════════════════════════════

_PHYSICS_KEYWORDS = [
    # Lớp 10 — Động học
    "động học", "chuyển động", "vận tốc", "gia tốc", "quãng đường",
    "rơi tự do", "ném ngang", "ném xiên", "ném lên", "chạm đất",
    "độ cao", "tầm xa", "tầm cao", "thời gian rơi",
    "chuyển động thẳng", "chuyển động đều", "chuyển động tròn",
    # Lớp 10 — Động lực học
    "lực", "hợp lực", "phân tích lực", "newton", "niutơn",
    "ma sát", "lực ma sát", "hệ số ma sát",
    "lực đàn hồi", "lực căng", "trọng lực", "trọng lượng",
    "mặt phẳng nghiêng", "lực hướng tâm",
    # Lớp 10 — Năng lượng & Động lượng
    "công", "công suất", "động năng", "thế năng", "cơ năng",
    "bảo toàn cơ năng", "bảo toàn năng lượng", "hiệu suất",
    "động lượng", "xung lực", "va chạm",
    # Lớp 10 — Momen & Nhiệt học
    "momen lực", "đòn bẩy", "cân bằng", "trọng tâm",
    "nhiệt lượng", "nội năng", "nhiệt dung riêng",
    "khí lí tưởng", "áp suất khí", "boyle", "charles",
    "đẳng nhiệt", "đẳng áp", "đẳng tích",
    # Lớp 11 — Điện tích & Điện trường
    "điện tích", "coulomb", "culông", "điện trường",
    "cường độ điện trường", "điện thế", "hiệu điện thế",
    "tụ điện", "điện dung",
    # Lớp 11 — Mạch điện
    "dòng điện", "cường độ dòng điện", "điện trở", "ohm", "ôm",
    "mạch điện", "nối tiếp", "song song", "nguồn điện",
    "suất điện động", "điện trở trong", "công suất điện",
    "điện năng", "nhiệt lượng tỏa", "joule", "jun",
    # Lớp 11 — Từ trường & Cảm ứng
    "từ trường", "cảm ứng từ", "lực từ", "lực lorentz", "lực ampe",
    "ống dây", "nam châm", "từ thông", "cảm ứng điện từ",
    "suất điện động cảm ứng", "faraday", "lenz",
    "tự cảm", "độ tự cảm", "cuộn cảm",
    # Lớp 11 — Quang học
    "khúc xạ", "chiết suất", "phản xạ toàn phần",
    "thấu kính", "tiêu cự", "tiêu điểm", "độ tụ",
    "ảnh thật", "ảnh ảo", "gương cầu",
    "mắt", "cận thị", "viễn thị", "kính lúp",
    # Lớp 12 — Dao động
    "dao động", "dao động điều hoà", "dao động điều hòa",
    "biên độ", "chu kỳ", "tần số", "tần số góc",
    "con lắc lò xo", "con lắc đơn",
    "dao động tắt dần", "cộng hưởng",
    # Lớp 12 — Sóng
    "sóng cơ", "sóng ngang", "sóng dọc",
    "bước sóng", "giao thoa", "sóng dừng",
    "sóng âm", "cường độ âm", "mức cường độ âm", "decibel",
    # Lớp 12 — Điện xoay chiều
    "điện xoay chiều", "dòng xoay chiều",
    "tổng trở", "cảm kháng", "dung kháng", "rlc",
    "hệ số công suất", "máy biến áp", "máy phát điện",
    # Lớp 12 — Sóng điện từ & Quang phổ
    "sóng điện từ", "quang phổ",
    "tia hồng ngoại", "tia tử ngoại", "tia x", "tia rơnghen",
    "tán sắc", "giao thoa ánh sáng",
    # Lớp 12 — Lượng tử
    "quang điện", "công thoát", "photon", "phôton",
    "planck", "hằng số planck", "thuyết bo",
    "mức năng lượng", "vạch quang phổ",
    # Lớp 12 — Hạt nhân
    "hạt nhân", "proton", "prôton", "neutron", "nơtron",
    "phóng xạ", "chu kỳ bán rã", "hằng số phóng xạ",
    "năng lượng liên kết", "độ hụt khối",
    "phân hạch", "nhiệt hạch",
    # Từ chung
    "vật lý", "vật lí", "sgk", "thpt",
    "lớp 10", "lớp 11", "lớp 12",
    "tính", "tìm", "bằng bao nhiêu", "cho biết",
    "kg", "m/s", "km/h", "newton", "jun", "vôn", "ampe",
]

# Tín hiệu rõ ràng ngoài phạm vi — chỉ dùng cụm từ dài, tránh false positive
_OUT_OF_SCOPE_SIGNALS = [
    "môn hóa", "hóa học", "phản ứng hóa học", "nguyên tố hóa học",
    "môn sinh", "sinh học", "tế bào", "di truyền", "tiến hóa",
    "môn toán", "toán học", "đạo hàm", "tích phân", "ma trận",
    "môn văn", "văn học", "tác phẩm văn", "phân tích thơ",
    "môn lịch sử", "lịch sử việt nam", "chiến tranh thế giới",
    "môn địa", "địa lý", "khí hậu", "địa hình",
    "tiếng anh", "ngữ pháp tiếng anh", "từ vựng tiếng anh",
    "lập trình", "code python", "code java", "thuật toán",
    "nấu ăn", "công thức nấu", "món ăn",
    "bóng đá", "thể thao", "giải đấu",
    "âm nhạc", "ca hát", "bài hát",
    "tâm lý học", "triết học", "kinh tế học",
    "y học", "bệnh", "thuốc chữa","hóa",
]


def is_out_of_scope(raw_question: str) -> bool:
    """
    Gọi từ app.py với raw_question (TRƯỚC correct_physics_text).
    Return True nếu câu hỏi rõ ràng ngoài phạm vi Vật lý THPT.

    Nguyên tắc whitelist-first:
      - Có bất kỳ từ khóa vật lý → False (in scope)
      - Có tín hiệu ngoài scope RÕ RÀNG và không có từ vật lý → True
      - Nghi ngờ → False (safe default, để LLM xử lý)
    """
    q = raw_question.lower()

    # Whitelist: nếu có từ khóa vật lý → in scope
    if any(kw in q for kw in _PHYSICS_KEYWORDS):
        return False

    # Blacklist: chỉ out_of_scope khi tín hiệu rõ ràng
    if any(sig in q for sig in _OUT_OF_SCOPE_SIGNALS):
        return True

    return False  # nghi ngờ → cứ cho qua


# ══════════════════════════════════════════════════════════════════
# QUERY EXPANSION
# ══════════════════════════════════════════════════════════════════

_QUERY_ALIASES = {
    "omega":   "ω tần số góc",
    "lambda":  "λ bước sóng",
    "delta":   "Δ biến thiên",
    "alpha":   "α",
    "beta":    "β",
    "gamma":   "γ tia gamma",
    "theta":   "θ góc",
    "epsilon": "ε suất điện động",
    "phi":     "φ Φ từ thông",
    "rho":     "ρ điện trở suất khối lượng riêng",
    "sigma":   "σ",
    "pi":      "π",
    "sqrt":    "√ căn bậc hai",
    "coulomb": "culông C điện tích",
    "newton":  "niutơn N lực",
    "joule":   "jun J năng lượng",
    "watt":    "oát W công suất",
    "farad":   "fara F tụ điện",
    "henry":   "henry H tự cảm",
    "tesla":   "tesla T cảm ứng từ",
    "hertz":   "héc Hz tần số",
    "pascal":  "pascal Pa áp suất",
    "ampere":  "ampe A dòng điện",
    "ohm":     "ôm Ω điện trở",
    "volt":    "vôn V hiệu điện thế",
    "^2":      "² bình phương",
    "^3":      "³ lập phương",
    "m/s2":    "m/s² mét trên giây bình phương",
}


def expand_query(question: str) -> str:
    q = question.lower()
    extras = []
    for key, alias in _QUERY_ALIASES.items():
        if key in q:
            extras.append(alias)
    if extras:
        return question + " " + " ".join(extras)
    return question


# ══════════════════════════════════════════════════════════════════
# QUERY REWRITING — dành riêng cho bài tính toán
# ══════════════════════════════════════════════════════════════════

_REWRITE_PROMPT = """Phân tích câu hỏi vật lý sau. Trả lời ĐÚNG format này, không thêm gì khác:

DẠNG: [động học/lực/năng lượng/điện tích/điện trường/mạch điện/từ trường/dao động/sóng/quang/hạt nhân]
TÌM: [đại lượng cần tính, ký hiệu vật lý]
BIẾT: [các đại lượng đã cho, viết ngắn]
CÔNG THỨC: [công thức áp dụng, dùng ký hiệu]

Câu hỏi: {question}"""


def rewrite_query_for_exercise(question: str, groq_client) -> str:
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            max_tokens=120,
            temperature=0,
            messages=[{"role": "user", "content": _REWRITE_PROMPT.format(question=question)}]
        )
        analysis = resp.choices[0].message.content.strip()
        return f"{question}\n{analysis}"
    except Exception as err:
        print(f"   [rewrite] Lỗi, fallback về query gốc: {err}")
        return question


# ══════════════════════════════════════════════════════════════════
# CLASSIFY QUERY & CHUNK
# ══════════════════════════════════════════════════════════════════

THEORY_STRONG_KEYWORDS = [
    "là gì", "định nghĩa", "khái niệm", "giải thích", "nêu", "trình bày",
    "phát biểu", "đặc trưng", "bản chất", "nguyên lý", "định luật",
    "tại sao", "vì sao", "ý nghĩa", "so sánh", "phân biệt", "đặc điểm",
    "tính chất", "câu nào", "câu nào sai", "câu nào đúng", "phát biểu nào",
    "nhận xét nào", "điều nào", "ý nào", "đúng hay sai", "đúng không",
    "có đúng không", "có phải", "có phải không",
]

EXERCISE_KEYWORDS = [
    "tính", "tìm", "bao nhiêu", "bằng bao nhiêu", "giải", "xác định",
    "cho biết", "vận tốc bằng", "lực tác dụng", "gia tốc bằng",
    "bài tập", "bài toán", "áp dụng công thức",
]

EXERCISE_CHUNK_PATTERNS = [
    r'\b\d+[\.,]?\d*\s*(m/s|km/h|m|s|kg|N|J|W|Pa|K|°C|rad|Hz|V|A|Ω)\b',
    r'(Lời giải|Giải:|Bài giải|Hướng dẫn giải)',
    r'(Bài\s+\d+[\.:]\s)',
    r'(Câu\s+\d+[\.:]\s)',
    r'(Cho biết|Cho rằng|Một vật|Một lò xo|Một chất điểm)',
    r'(😊 \?|tìm\s+\w+\s*=)',
    r'[a-zA-ZÀ-ỹ]\s*[=]\s*[\d½]',
    r'[a-zA-Z][²³]',
    r'√|∑|∞|≈|≠|≤|≥',
    r'\b(công thức|định luật|định nghĩa cơ năng|thế năng|động năng|cơ năng)\b',
]

EXERCISE_CHUNK_RE = re.compile("|".join(EXERCISE_CHUNK_PATTERNS), re.IGNORECASE)


def classify_query(question: str) -> str:
    q = question.lower()

    has_theory = any(k in q for k in THEORY_STRONG_KEYWORDS)
    if has_theory:
        return "theory"

    has_exercise_keyword = any(k in q for k in EXERCISE_KEYWORDS)
    has_number_with_unit = bool(re.search(
        r'\d+[\.,]?\d*\s*(m/s|km/h|m\b|s\b|kg|N\b|J\b|W\b|Hz|V\b|A\b|Ω|cm|mm|rad)',
        q
    ))

    if has_exercise_keyword and has_number_with_unit:
        return "exercise"

    if has_exercise_keyword:
        return "mixed"

    return "mixed"


_FORMULA_RE = re.compile(
    r'(?:'
    r'[A-Za-zÀ-ỹ]\s*=\s*[\d½]'
    r'|[a-zA-Z][²³]'
    r'|√|∑|∞|≈|≠|≤|≥'
    r'|công thức|định luật'
    r')',
    re.IGNORECASE
)


def classify_chunk(chunk_text: str) -> str:
    if chunk_text.startswith("[HÌNH MINH HOẠ"):
        return "image"
    if EXERCISE_CHUNK_RE.search(chunk_text):
        return "exercise"
    if _FORMULA_RE.search(chunk_text):
        has_numbers = bool(re.search(r'\d+[\.,]\d*\s*(m/s|kg|N|J|Hz|V|Pa)', chunk_text))
        return "exercise" if has_numbers else "theory"
    return "theory"


# ══════════════════════════════════════════════════════════════════
# RERANK
# ══════════════════════════════════════════════════════════════════

def word_overlap(text_a: str, text_b: str) -> float:
    words_a = set(re.findall(r'\w+', text_a.lower()))
    words_b = set(re.findall(r'\w+', text_b.lower()))
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def deduplicate_chunks(docs: list, metas: list) -> tuple:
    kept_docs, kept_metas = [], []
    for doc, meta in zip(docs, metas):
        is_dup = any(word_overlap(doc, kept) >= DEDUP_SIMILARITY for kept in kept_docs)
        if not is_dup:
            kept_docs.append(doc)
            kept_metas.append(meta)
    return kept_docs, kept_metas


def rerank_chunks(question: str, docs: list, metas: list, dists: list, query_type: str) -> tuple:
    q = expand_query(question).lower()

    important_phrases = [
        # Lớp 10
        "chuyển động", "chuyển động thẳng", "chuyển động đều",
        "chuyển động biến đổi đều", "vận tốc", "vận tốc tức thời",
        "gia tốc", "quãng đường", "tọa độ", "thời gian",
        "rơi tự do", "ném ngang", "ném xiên",
        "lực", "hợp lực", "phân tích lực",
        "định luật newton", "quán tính",
        "lực ma sát", "ma sát nghỉ", "ma sát trượt",
        "lực đàn hồi", "lực căng dây",
        "trọng lực", "trọng lượng",
        "công", "công suất",
        "động năng", "thế năng", "cơ năng",
        "định luật bảo toàn cơ năng",
        "momen lực", "cân bằng", "đòn bẩy",
        "khối lượng riêng", "áp suất",
        "nhiệt lượng", "nội năng", "nhiệt dung riêng",
        "phương trình trạng thái", "định luật charles",
        "định luật boyle", "khí lí tưởng",
        # Lớp 11
        "điện tích", "điện tích điểm",
        "định luật coulomb", "culông",
        "điện trường", "điện trường đều",
        "cường độ điện trường",
        "điện thế", "hiệu điện thế",
        "tụ điện", "điện dung",
        "dòng điện", "cường độ dòng điện",
        "điện trở", "định luật ohm",
        "công suất điện", "điện năng",
        "nguồn điện", "suất điện động",
        "mạch điện", "mạch nối tiếp", "mạch song song",
        "từ trường", "cảm ứng từ",
        "đường sức từ", "lực lorentz",
        "lực từ", "ống dây", "nam châm điện",
        "cảm ứng điện từ", "từ thông",
        "định luật faraday", "suất điện động cảm ứng",
        "khúc xạ", "phản xạ toàn phần",
        "thấu kính", "thấu kính hội tụ", "thấu kính phân kì",
        "ảnh thật", "ảnh ảo",
        # Lớp 12
        "dao động điều hoà", "biên độ", "chu kỳ", "tần số", "tần số góc",
        "con lắc lò xo", "con lắc đơn",
        "dao động tắt dần", "dao động cưỡng bức", "cộng hưởng",
        "sóng cơ", "sóng ngang", "sóng dọc",
        "bước sóng", "giao thoa sóng", "sóng dừng",
        "sóng âm", "cường độ âm", "mức cường độ âm",
        "điện xoay chiều", "mạch rlc",
        "cộng hưởng điện", "hệ số công suất",
        "máy biến áp", "truyền tải điện năng",
        "sóng điện từ", "quang phổ",
        "tia hồng ngoại", "tia tử ngoại", "tia x",
        "hiện tượng quang điện", "công thoát",
        "lượng tử ánh sáng", "photon",
        "thuyết bo", "mức năng lượng",
        "phóng xạ", "chu kỳ bán rã", "hằng số phóng xạ",
        "hạt nhân", "năng lượng liên kết", "độ hụt khối",
        "phản ứng hạt nhân", "phân hạch", "nhiệt hạch",
    ]

    important_phrases = sorted(important_phrases, key=len, reverse=True)
    dynamic_phrases = [phrase for phrase in important_phrases if phrase in q]

    q_words = set(re.findall(r'\w+', q))
    scored = []

    for doc, meta, dist in zip(docs, metas, dists):
        text = doc.lower()
        score = 0.0

        chunk_type = classify_chunk(doc)
        source_folder = meta.get("source_folder", "raw")

        phrase_score = sum(15 for phrase in dynamic_phrases if phrase in text)
        score += phrase_score

        doc_words = set(re.findall(r'\w+', text))
        overlap = len(q_words & doc_words)
        score += overlap * 3

        score += (1.0 - dist) * 8

        if query_type == "theory":
            if "giải thích" in text:    score += 6
            if "là gì" in text:         score += 7
            if "định nghĩa" in text:    score += 8
            if "khái niệm" in text:     score += 6
            if chunk_type == "theory":  score += 4
            if source_folder == "raw":  score += 3
            if _FORMULA_RE.search(doc) and not re.search(r'\d+[\.,]\d*', doc):
                score += 5

        elif query_type == "exercise":
            if chunk_type == "exercise":                              score += 6
            if "giải" in text or "lời giải" in text:                 score += 4
            if "công thức" in text:                                   score += 3
            if "tính" in text:                                        score += 3
            if meta.get("type") == "web" and ("bước" in text or "step" in text):
                score += 4

        elif query_type == "mixed":
            if chunk_type == "theory":    score += 3
            if chunk_type == "exercise":  score += 3
            if meta.get("type") == "web": score += 2

        if overlap == 0 and phrase_score == 0:
            score -= 5

        scored.append((score, doc, meta, chunk_type))

    scored.sort(reverse=True, key=lambda x: x[0])

    return (
        [x[1] for x in scored],
        [x[2] for x in scored],
        [x[3] for x in scored],
        [x[0] for x in scored],
    )


# ══════════════════════════════════════════════════════════════════
# MERGE NEIGHBOR CHUNKS
# ══════════════════════════════════════════════════════════════════

def merge_neighbor_chunks(collection, meta: dict, query_type="mixed") -> str:
    source        = meta.get("source")
    idx           = meta.get("chunk_index", 0)
    source_type   = meta.get("type", "")
    source_folder = meta.get("source_folder", "raw")

    # Web chunks: ID format là "web_{source}_text_{idx}"
    # Raw/exercises: ID format là "{source}_{idx}"
    if source_type == "web" or source_folder == "web":
        id_prefix = f"web_{source}_text"
    else:
        id_prefix = source

    offsets = [-1, 0, 1] if query_type == "exercise" else [0]
    merged = []

    for offset in offsets:
        chunk_id = f"{id_prefix}_{idx + offset}"
        try:
            result = collection.get(ids=[chunk_id])
            if result["documents"]:
                text = result["documents"][0]
                if len(text) > MAX_CHUNK_CHARS:
                    text = text[:MAX_CHUNK_CHARS] + "..."
                merged.append(text)
        except Exception:
            continue

    combined = "\n".join(merged)
    if len(combined) > MAX_CHUNK_CHARS * 2:
        combined = combined[:MAX_CHUNK_CHARS * 2] + "..."
    return combined


# ══════════════════════════════════════════════════════════════════
# TOKEN BUDGET
# ══════════════════════════════════════════════════════════════════

def estimate_tokens(text: str) -> int:
    return int(len(text) / 2.5)


def trim_context_to_budget(chunks: list, question: str, system_prompt: str) -> list:
    # Chỉ tính question + overhead, KHÔNG tính system_prompt vào budget context
    # vì system_prompt nằm ngoài context window của RAG
    base = estimate_tokens(question) + 300
    budget = TOKEN_BUDGET - base

    if budget <= 0:
        return []

    result, used = [], 0

    for chunk in chunks:
        if len(chunk) > MAX_CHUNK_CHARS:
            chunk = chunk[:MAX_CHUNK_CHARS] + "..."

        t = estimate_tokens(chunk)

        if used + t > budget:
            remaining = budget - used
            if remaining > 80:
                char_limit = int(remaining * 2.5)
                result.append(chunk[:char_limit] + "...")
            break

        result.append(chunk)
        used += t

    return result


# ══════════════════════════════════════════════════════════════════
# RETRIEVE CONTEXT — main entry point
# RAG luôn chạy bình thường, KHÔNG check out-of-scope ở đây.
# Scope check do app.py xử lý trước khi gọi hàm này.
# ══════════════════════════════════════════════════════════════════

def retrieve_context(
    question: str,
    system_prompt: str = "",
    verbose: bool = True,
    groq_client=None,
) -> str:
    try:
        model = _get_model()
        collection = _get_collection()

        query_type = classify_query(question)
        question_expanded = expand_query(question)

        if verbose and question_expanded != question:
            print(f"   [expand] '{question}' → '{question_expanded[:80]}'")

        if query_type == "exercise" and groq_client is not None:
            question_for_embed = rewrite_query_for_exercise(question_expanded, groq_client)
            if verbose:
                lines = question_for_embed.split("\n")
                print(f"   [rewrite] {len(lines)} dòng, {len(question_for_embed)} ký tự")
        else:
            question_for_embed = question_expanded

        embedding = model.encode(question_for_embed).tolist()

        results = collection.query(
            query_embeddings=[embedding],
            n_results=TOP_K,
            include=["documents", "distances", "metadatas"]
        )

        raw_docs  = results["documents"][0]
        raw_dists = results["distances"][0]
        raw_metas = results["metadatas"][0]

        if verbose:
            print(f"\n{'='*60}")
            print(f"RAG | type='{query_type}' | '{question[:50]}'")
            for i, (doc, dist, meta) in enumerate(zip(raw_docs, raw_dists, raw_metas)):
                ctype  = classify_chunk(doc)
                flag   = "OK" if dist <= MAX_DISTANCE else "XX"
                icon   = {"theory": "[LT]", "exercise": "[BT]", "image": "[HI]"}.get(ctype, "[?]")
                folder = meta.get("source_folder", "raw")
                print(f"   {flag}{icon}[{i+1}] dist={dist:.3f} src={folder} | {doc[:50]}...")

        filtered = [
            (doc, meta, dist)
            for doc, dist, meta in zip(raw_docs, raw_dists, raw_metas)
            if dist <= MAX_DISTANCE
        ]

        if not filtered:
            if verbose:
                print("   Không có chunk pass MAX_DISTANCE\n")
            return ""

        f_docs, f_metas, f_dists = zip(*filtered)

        r_docs, r_metas, r_types, r_scores = rerank_chunks(
            question,
            list(f_docs),
            list(f_metas),
            list(f_dists),
            query_type
        )

        d_docs, d_metas = deduplicate_chunks(r_docs, r_metas)

        d_docs  = d_docs[:FINAL_TOP_K]
        d_metas = d_metas[:FINAL_TOP_K]

        if verbose:
            print(f"\n   → Chọn top-{FINAL_TOP_K} (type={query_type}):")
            for i, (doc, meta) in enumerate(zip(d_docs, d_metas)):
                ctype  = classify_chunk(doc)
                icon   = {"theory": "[LT]", "exercise": "[BT]", "image": "[HI]"}.get(ctype, "[?]")
                folder = meta.get("source_folder", "raw")
                print(f"   {icon}[{i+1}] src={folder} | {doc[:50]}...")

        expanded = []
        for meta in d_metas:
            merged = merge_neighbor_chunks(collection, meta, query_type)
            if merged:
                expanded.append(merged)

        expanded = trim_context_to_budget(expanded, question, system_prompt)

        if verbose:
            total_tok = sum(estimate_tokens(c) for c in expanded)
            print(f"   Context: {len(expanded)} chunks ~{total_tok} tokens (budget={TOKEN_BUDGET})")
            print(f"{'='*60}\n")

        return "\n\n---\n\n".join(expanded)

    except Exception as e:
        print(f"[RAG ERROR] {e}")
        return ""


# ══════════════════════════════════════════════════════════════════
# BUILD RAG PROMPT
# ══════════════════════════════════════════════════════════════════

CALC_EXAMPLE = """
Vi du cach trinh bay:
  De: v = 72 km/h, t = 5 phut. Tinh s.
  -> Doi: v = 20 m/s, t = 300 s
  -> CT: s = v x t = 20 x 300 = 6000 m
"""


def build_rag_prompt(question: str, context: str) -> str:
    query_type = classify_query(question)

    if not context:
        if query_type == "exercise":
            return f"{CALC_EXAMPLE}\nCau hoi: {question}"
        return f"Cau hoi: {question}"

    header = {
        "theory":   "Tai lieu ly thuyet SGK (raw):",
        "exercise": "Cong thuc SGK + bai tap mau:",
        "mixed":    "Tai lieu ly thuyet + vi du:",
    }.get(query_type, "Tai lieu:")

    instruction = {
        "theory": (
            "Tra loi LY THUYET suc tich, KHONG tinh so, KHONG giai bai tap. "
            "Chi dung kien thuc tu tai lieu, khong them bai toan vi du."
        ),
        "exercise": (
            "Giai day du 3 buoc: cong thuc -> thay so -> ket qua. "
            "BAT BUOC hoan thanh den Buoc ba voi ket qua so cu the."
        ),
        "mixed": "Giai thich ngan (1-2 cau), roi vi du neu can.",
    }.get(query_type, "Tra loi trong tam.")

    example_block = CALC_EXAMPLE if query_type == "exercise" else ""

    return f"""{header}
{context}

{example_block}

Yeu cau: {instruction}
CHI DUOC tra loi dua tren tai lieu tren.
Neu tai lieu khong lien quan, phai noi:
"Khong tim thay thong tin phu hop trong tai lieu."

Cau hoi: {question}"""
