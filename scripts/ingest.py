import os
import glob
import re
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
import PyPDF2
from pdf2image import convert_from_path
from dotenv import load_dotenv
import pytesseract
import pdfplumber

load_dotenv()

PDF_DIRS      = ["data/raw", "data/exercises"]
PROCESSED_DIR = "data/processed"
DB_DIR        = "data/chroma_db"
COLLECTION_NAME = "physbot_sgk"
POPPLER_PATH    = r"C:\poppler-25.12.0\Library\bin"
TESSERACT_PATH  = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)


# ══════════════════════════════════════════════════════════════════
# CLEAN TEXT — fix lỗi encoding + layout từ PDF extract
# ══════════════════════════════════════════════════════════════════

_HEADER_RE = re.compile(
    r"Lớp học lý Hai Nguyen.*?(?=\n)",
    re.IGNORECASE
)

_ENCODING_MAP = {
    "\x00": "",
    "\x0c": "\n",
    "\uf061": "α", "\uf062": "β", "\uf063": "χ", "\uf064": "δ",
    "\uf065": "ε", "\uf066": "φ", "\uf067": "γ", "\uf068": "η",
    "\uf069": "ι", "\uf06a": "ϕ", "\uf06b": "κ", "\uf06c": "λ",
    "\uf06d": "μ", "\uf06e": "ν", "\uf06f": "ο", "\uf070": "π",
    "\uf071": "θ", "\uf072": "ρ", "\uf073": "σ", "\uf074": "τ",
    "\uf075": "υ", "\uf076": "ω", "\uf077": "ω", "\uf078": "ξ",
    "\uf079": "ψ", "\uf07a": "ζ",
    "\uf041": "Α", "\uf042": "Β", "\uf043": "Χ", "\uf044": "Δ",
    "\uf045": "Ε", "\uf046": "Φ", "\uf047": "Γ", "\uf048": "Η",
    "\uf049": "Ι", "\uf04b": "Κ", "\uf04c": "Λ", "\uf04d": "Μ",
    "\uf04e": "Ν", "\uf04f": "Ω", "\uf050": "Π", "\uf051": "Θ",
    "\uf052": "Ρ", "\uf053": "Σ", "\uf054": "Τ", "\uf055": "Υ",
    "\uf056": "ς", "\uf057": "Ω", "\uf058": "Ξ", "\uf059": "Ψ",
    "\uf05a": "Ζ",
    "\uf02b": "+", "\uf02d": "−", "\uf03d": "=", "\uf03c": "<",
    "\uf03e": ">", "\uf0a3": "≤", "\uf0b3": "≥", "\uf0b9": "≠",
    "\uf0ab": "↔", "\uf0ae": "→", "\uf0ac": "←", "\uf0ad": "↑",
    "\uf0af": "↓", "\uf0db": "↕", "\uf0dc": "⇒", "\uf0de": "⇔",
    "\uf0e0": "→", "\uf0e1": "↑", "\uf0e3": "↔", "\uf0b1": "±",
    "\uf0b4": "×", "\uf0b8": "÷", "\uf0b7": "·", "\uf0b0": "°",
    "\uf0b2": "′", "\uf0a2": "″", "\uf0d6": "√", "\uf0a5": "∞",
    "\uf0a7": "∂", "\uf0d1": "∇", "\uf0c5": "∅", "\uf0ce": "∈",
    "\uf0cf": "∉", "\uf0cc": "⊂", "\uf0c9": "∪", "\uf0c7": "∩",
    "\uf0d2": "®", "\uf0a9": "©", "\uf0e4": "◊",
    "\uf0a4": "∴", "\uf0c0": "≅", "\uf0bb": "≈", "\uf0be": "…",
    "\uf0b6": "∝", "\uf0a8": "°", "\uf0d0": "−",
    "\uf0f2": "∫", "\uf0f3": "∮", "\uf0e5": "∑", "\uf0d5": "∏",
    "\uf0e6": "(", "\uf0f6": ")", "\uf0e9": "{", "\uf0fd": "}",
    "\uf0eb": "|",
    "\uf028": "(", "\uf029": ")", "\uf02f": "/", "\uf05c": "\\",
    "\x80": "€", "\x82": "‚", "\x83": "ƒ", "\x84": "„", "\x85": "…",
    "\x86": "†", "\x87": "‡", "\x88": "ˆ", "\x89": "‰", "\x8a": "Š",
    "\x8b": "‹", "\x8c": "Œ", "\x8e": "Ž", "\x91": "'", "\x92": "'",
    "\x93": '"', "\x94": '"', "\x95": "•", "\x96": "–", "\x97": "—",
    "\x98": "~", "\x99": "™", "\x9a": "š", "\x9b": "›", "\x9c": "œ",
    "\x9e": "ž", "\x9f": "Ÿ",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "--", "\u2022": "·", "\u2026": "...",
    "\u00b0": "°", "\u00b1": "±", "\u00b2": "^2", "\u00b3": "^3",
    "\u00b5": "μ", "\u00d7": "×", "\u00f7": "÷", "\u2212": "-",
    "\u221a": "√", "\u221e": "∞", "\u2248": "≈", "\u2260": "≠",
    "\u2264": "≤", "\u2265": "≥",
    "\u03b1": "α", "\u03b2": "β", "\u03b3": "γ", "\u03b4": "δ",
    "\u03b5": "ε", "\u03b7": "η", "\u03b8": "θ", "\u03bb": "λ",
    "\u03bc": "μ", "\u03bd": "ν", "\u03be": "ξ", "\u03c0": "π",
    "\u03c1": "ρ", "\u03c3": "σ", "\u03c4": "τ", "\u03c6": "φ",
    "\u03c7": "χ", "\u03c8": "ψ", "\u03c9": "ω",
    "\u0393": "Γ", "\u0394": "Δ", "\u0398": "Θ", "\u039b": "Λ",
    "\u039e": "Ξ", "\u03a0": "Π", "\u03a3": "Σ", "\u03a6": "Φ",
    "\u03a8": "Ψ", "\u03a9": "Ω",
    "\u200b": "", "\u200c": "", "\u200d": "", "\u00ad": "",
    "\ufeff": "", "\ufffd": "",
}

_COMBINING_VEC_RE = re.compile(
    r"([A-Za-zαβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ])"
    r"[\u20d7\u20d6\u20e1\u0305\u20d2]"
)

_MATH_BOLD_VECTORS = {
    "\U0001d41a": "vec(a)", "\U0001d41b": "vec(b)", "\U0001d41c": "vec(c)",
    "\U0001d41d": "vec(d)", "\U0001d41e": "vec(e)", "\U0001d41f": "vec(f)",
    "\U0001d420": "vec(g)", "\U0001d421": "vec(h)", "\U0001d422": "vec(i)",
    "\U0001d423": "vec(j)", "\U0001d424": "vec(k)", "\U0001d425": "vec(l)",
    "\U0001d426": "vec(m)", "\U0001d427": "vec(n)", "\U0001d428": "vec(o)",
    "\U0001d429": "vec(p)", "\U0001d42a": "vec(q)", "\U0001d42b": "vec(r)",
    "\U0001d42c": "vec(s)", "\U0001d42d": "vec(t)", "\U0001d42e": "vec(u)",
    "\U0001d42f": "vec(v)", "\U0001d430": "vec(w)", "\U0001d431": "vec(x)",
    "\U0001d432": "vec(y)", "\U0001d433": "vec(z)",
    "\U0001d400": "vec(A)", "\U0001d401": "vec(B)", "\U0001d402": "vec(C)",
    "\U0001d403": "vec(D)", "\U0001d404": "vec(E)", "\U0001d405": "vec(F)",
    "\U0001d406": "vec(G)", "\U0001d407": "vec(H)", "\U0001d408": "vec(I)",
    "\U0001d409": "vec(J)", "\U0001d40a": "vec(K)", "\U0001d40b": "vec(L)",
    "\U0001d40c": "vec(M)", "\U0001d40d": "vec(N)", "\U0001d40e": "vec(O)",
    "\U0001d40f": "vec(P)", "\U0001d410": "vec(Q)", "\U0001d411": "vec(R)",
    "\U0001d412": "vec(S)", "\U0001d413": "vec(T)", "\U0001d414": "vec(U)",
    "\U0001d415": "vec(V)", "\U0001d416": "vec(W)", "\U0001d417": "vec(X)",
    "\U0001d418": "vec(Y)", "\U0001d419": "vec(Z)",
}

_MATH_ITALIC_BOLD_VECTORS = {
    "\U0001d468": "vec(A)", "\U0001d469": "vec(B)", "\U0001d46a": "vec(C)",
    "\U0001d46b": "vec(D)", "\U0001d46c": "vec(E)", "\U0001d46d": "vec(F)",
    "\U0001d46e": "vec(G)", "\U0001d46f": "vec(H)", "\U0001d470": "vec(I)",
    "\U0001d471": "vec(J)", "\U0001d472": "vec(K)", "\U0001d473": "vec(L)",
    "\U0001d474": "vec(M)", "\U0001d475": "vec(N)", "\U0001d476": "vec(O)",
    "\U0001d477": "vec(P)", "\U0001d478": "vec(Q)", "\U0001d479": "vec(R)",
    "\U0001d47a": "vec(S)", "\U0001d47b": "vec(T)", "\U0001d47c": "vec(U)",
    "\U0001d47d": "vec(V)", "\U0001d47e": "vec(W)", "\U0001d47f": "vec(X)",
    "\U0001d480": "vec(Y)", "\U0001d481": "vec(Z)",
    "\U0001d482": "vec(a)", "\U0001d483": "vec(b)", "\U0001d484": "vec(c)",
    "\U0001d485": "vec(d)", "\U0001d486": "vec(e)", "\U0001d487": "vec(f)",
    "\U0001d488": "vec(g)", "\U0001d489": "vec(h)", "\U0001d48a": "vec(i)",
    "\U0001d48b": "vec(j)", "\U0001d48c": "vec(k)", "\U0001d48d": "vec(l)",
    "\U0001d48e": "vec(m)", "\U0001d48f": "vec(n)", "\U0001d490": "vec(o)",
    "\U0001d491": "vec(p)", "\U0001d492": "vec(q)", "\U0001d493": "vec(r)",
    "\U0001d494": "vec(s)", "\U0001d495": "vec(t)", "\U0001d496": "vec(u)",
    "\U0001d497": "vec(v)", "\U0001d498": "vec(w)", "\U0001d499": "vec(x)",
    "\U0001d49a": "vec(y)", "\U0001d49b": "vec(z)",
}


def normalize_vectors(text: str) -> str:
    text = _COMBINING_VEC_RE.sub(lambda m: f"vec({m.group(1)})", text)
    for ch, replacement in _MATH_BOLD_VECTORS.items():
        text = text.replace(ch, replacement)
    for ch, replacement in _MATH_ITALIC_BOLD_VECTORS.items():
        text = text.replace(ch, replacement)
    return text


_SUPERSCRIPT_MAP = str.maketrans({
    "⁰": "^0", "¹": "^1", "²": "^2", "³": "^3", "⁴": "^4",
    "⁵": "^5", "⁶": "^6", "⁷": "^7", "⁸": "^8", "⁹": "^9",
    "⁺": "^+", "⁻": "^-", "⁼": "^=",
})
_SUBSCRIPT_MAP = str.maketrans({
    "₀": "_0", "₁": "_1", "₂": "_2", "₃": "_3", "₄": "_4",
    "₅": "_5", "₆": "_6", "₇": "_7", "₈": "_8", "₉": "_9",
    "₊": "_+", "₋": "_-",
})


def clean_text(text: str) -> str:
    text = _HEADER_RE.sub("", text)
    text = re.sub(r"^\s*\d{1,3}\s*$", "", text, flags=re.MULTILINE)

    for wrong, correct in _ENCODING_MAP.items():
        text = text.replace(wrong, correct)

    text = normalize_vectors(text)
    text = text.translate(_SUPERSCRIPT_MAP)
    text = text.translate(_SUBSCRIPT_MAP)

    _FORMULA_FRAGMENT_RE = re.compile(
        r"[=+\-×÷/<>≤≥≠≈±^_]"
        r"|^\d[\d,\.\s]*[a-zA-Zα-ωΑ-Ω]"
    )

    lines = text.split("\n")
    merged = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if (0 < len(line) <= 12
                and _FORMULA_FRAGMENT_RE.search(line)
                and not line.endswith(".")
                and not line.endswith(":")
                and not re.match(r"^[A-ZĐÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂƠƯ][a-zđàáâãèéêìíòóôõùúýăơư]", line)
                and i + 1 < len(lines)
                and len(lines[i + 1].strip()) > 0):
            merged.append(line + " " + lines[i + 1].strip())
            i += 2
            continue
        merged.append(line)
        i += 1

    text = "\n".join(merged)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s+$", "", text, flags=re.MULTILINE)

    return text.strip()


# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

def is_vietnamese_text(text: str) -> bool:
    vietnamese_chars = set(
        "àáâãèéêìíòóôõùúýăđơư"
        "ạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỷỹỵ"
        "ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝĂĐƠƯ"
        "ẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỶỸỴ"
    )
    count = sum(1 for c in text if c in vietnamese_chars)
    return count / max(len(text), 1) >= 0.01


def semantic_chunk_text(text: str) -> list:
    parts = re.split(r"(Bài\s+\d+|Câu\s+\d+)", text)
    chunks = []
    current = ""

    for part in parts:
        if re.match(r"(Bài\s+\d+|Câu\s+\d+)", part):
            if current.strip() and len(current.strip()) >= 50:
                chunks.append(current.strip())
            current = part
        else:
            current += "\n" + part

    if current.strip() and len(current.strip()) >= 50:
        chunks.append(current.strip())

    return [c for c in chunks if len(c.strip()) >= 50]


def ocr_with_tesseract(image) -> str:
    return pytesseract.image_to_string(image, lang="vie")


# ══════════════════════════════════════════════════════════════════
# EXTRACT TEXT
# ══════════════════════════════════════════════════════════════════

def extract_text_from_pdf(pdf_path: str) -> str:
    filename   = Path(pdf_path).stem
    cache_path = Path(PROCESSED_DIR) / f"{filename}.txt"

    if cache_path.exists():
        print(f"  [cache] Dùng cache: {cache_path}", flush=True)
        raw = cache_path.read_text(encoding="utf-8")
        return clean_text(raw)

    text = ""

    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

        if len(text.strip()) > 100 and is_vietnamese_text(text):
            print("  [PyPDF2] OK, có dấu tiếng Việt.", flush=True)
            cache_path.write_text(text, encoding="utf-8")
            return clean_text(text)
        else:
            print("  [PyPDF2] Thiếu dấu → thử pdfplumber.", flush=True)
            text = ""
    except Exception as e:
        print(f"  [PyPDF2] Lỗi: {e}", flush=True)

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
                if page_text:
                    text += page_text + "\n"

        if len(text.strip()) > 100 and is_vietnamese_text(text):
            print("  [pdfplumber] OK, có ký hiệu toán học.", flush=True)
            cache_path.write_text(text, encoding="utf-8")
            return clean_text(text)
        else:
            print("  [pdfplumber] Vẫn thiếu → chuyển sang OCR.", flush=True)
            text = ""
    except Exception as e:
        print(f"  [pdfplumber] Lỗi: {e}", flush=True)

    print("  OCR bằng Tesseract (tiếng Việt)...", flush=True)

    checkpoint_dir = Path(PROCESSED_DIR) / f"{filename}_pages"
    checkpoint_dir.mkdir(exist_ok=True)

    print("  Đang convert PDF sang ảnh (có thể mất vài phút)...", flush=True)
    images = convert_from_path(pdf_path, dpi=200, poppler_path=POPPLER_PATH)

    total = len(images)
    print(f"  Convert xong! Tổng số trang: {total}", flush=True)

    for i, img in enumerate(images):
        page_cache = checkpoint_dir / f"page_{i:04d}.txt"
        if page_cache.exists():
            print(f"  Trang {i+1}/{total} [cache]", flush=True)
            continue
        print(f"  OCR trang {i+1}/{total}...", flush=True)
        page_text = ocr_with_tesseract(img)
        page_cache.write_text(page_text, encoding="utf-8")
        print(f"  Trang {i+1}/{total} xong", flush=True)

    all_pages = sorted(checkpoint_dir.glob("page_*.txt"))
    text = "\n".join(p.read_text(encoding="utf-8") for p in all_pages)
    cache_path.write_text(text, encoding="utf-8")

    return clean_text(text)


# ══════════════════════════════════════════════════════════════════
# INGEST MAIN
# ══════════════════════════════════════════════════════════════════

def ingest():
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    client = chromadb.PersistentClient(path=DB_DIR)

    try:
        collection = client.get_collection(COLLECTION_NAME)
        print(f"Collection '{COLLECTION_NAME}' đã tồn tại, tiếp tục thêm vào.")
    except Exception:
        collection = client.create_collection(
            COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Tạo mới collection '{COLLECTION_NAME}' với cosine distance.")

    # ── Lấy toàn bộ ID đã có trong ChromaDB (không load document/embedding) ──
    existing_ids = set(collection.get(include=[])["ids"])
    print(f"ChromaDB hiện có: {len(existing_ids)} chunks đã được ingest.\n")

    # ── Gom PDF từ tất cả folders, bỏ trùng theo tên file ──
    seen_stems = set()
    pdf_files  = []

    for pdf_dir in PDF_DIRS:
        if not os.path.exists(pdf_dir):
            print(f"Folder không tồn tại, bỏ qua: {pdf_dir}")
            continue
        for path in glob.glob(f"{pdf_dir}/**/*.pdf", recursive=True):
            stem = Path(path).stem
            if stem not in seen_stems:
                seen_stems.add(stem)
                pdf_files.append(path)
            else:
                print(f"  [skip trùng] {Path(path).name}")

    print(f"Tìm thấy {len(pdf_files)} PDF\n")

    all_chunks = []
    all_ids    = []
    all_meta   = []

    for pdf_path in pdf_files:
        print(f"Đang xử lý: {pdf_path}", flush=True)

        text     = extract_text_from_pdf(pdf_path)
        chunks   = semantic_chunk_text(text)
        filename = Path(pdf_path).stem
        source_folder = "exercises" if "exercises" in pdf_path else "raw"

        skipped = 0
        added   = 0

        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) < 50:
                continue

            chunk_id = f"{filename}_{i}"

            # ── KEY FIX: Skip nếu ID đã tồn tại trong ChromaDB ──
            if chunk_id in existing_ids:
                skipped += 1
                continue

            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_meta.append({
                "source":        filename,
                "chunk_index":   i,
                "source_folder": source_folder,
            })
            added += 1

        if skipped > 0:
            print(f"  → {added} chunks mới | {skipped} chunks đã có (skip)", flush=True)
        else:
            print(f"  → {added} chunks mới", flush=True)

    if not all_chunks:
        print("\nKhông có chunk mới nào cần ingest. Tất cả đã được xử lý rồi!")
        print(f"Collection tổng: {collection.count()} chunks")
        return

    print(f"\nĐang tạo embeddings cho {len(all_chunks)} chunks mới...", flush=True)
    embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()

    collection.add(
        documents=all_chunks,
        embeddings=embeddings,
        ids=all_ids,
        metadatas=all_meta
    )

    total = collection.count()
    print(f"\nIngest hoàn tất!")
    print(f"   Chunks vừa thêm : {len(all_chunks)}")
    print(f"   Collection tổng : {total} chunks")


# ══════════════════════════════════════════════════════════════════
# TEST clean_text — python ingest.py --test
# ══════════════════════════════════════════════════════════════════

def _test_clean():
    test_cases = [
        ("Lớp học lý Hai Nguyen – 01694232474 – CS1: 105 Láng Hạ\n\nv = v₀ + at", "Xóa header lặp lại"),
        ("a =\n\nv - v0\n\nt", "Gom công thức bị vỡ dòng"),
        ("v\uf06c f\nT", "Fix ký tự λ bị encode sai → λ"),
        ("F  =  m  a", "Normalize khoảng trắng OCR"),
        ("v² - v₀² = 2as\n\n\n\n\nchuyển động thẳng", "Reduce blank lines + superscript"),
    ]

    print("=" * 60)
    print("TEST clean_text()")
    print("=" * 60)
    for inp, desc in test_cases:
        result = clean_text(inp)
        print(f"\n[{desc}]")
        print(f"  Input : {repr(inp[:60])}")
        print(f"  Output: {repr(result[:80])}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        _test_clean()
    else:
        ingest()