"""
web_ingest.py
─────────────
Ingest lời giải bài tập vật lý từ web vào ChromaDB.

Hỗ trợ:
  - loigiaihay.com, hoc247.net, vietjack.com, tailieu.vn
  - Chunk theo bài/câu/lời giải — không bị quá lớn
  - Metadata đầy đủ: source_folder, type, url, chunk_index
  - Dedup thông minh (check thẳng ChromaDB, không phụ thuộc file log):
      * URL chưa ingest → ingest cả text lẫn hình
      * URL đã ingest text nhưng chưa có hình → chỉ ingest hình, bỏ qua text
      * URL đã ingest đủ cả hai → bỏ qua hoàn toàn
  - Hình ảnh: Selenium scroll để trigger lazy-load, download về local,
    embed bằng CLIP, lọc quảng cáo tự động
    (whitelist domain bài giải + blacklist domain quảng cáo + kích thước)
"""

from trafilatura import extract
from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer
from PIL import Image
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import re, os, hashlib, io, time

# ══════════════════════════════════════════════════════════════════
# CẤU HÌNH
# ══════════════════════════════════════════════════════════════════

DB_DIR          = "data/chroma_db"
COLLECTION_NAME = "physbot_sgk"
PROCESSED_DIR   = "data/processed/web"
IMAGE_DIR       = "data/processed/web/images"

MAX_CHUNK_CHARS = 600
MIN_CHUNK_CHARS = 60

MIN_IMG_WIDTH  = 200
MIN_IMG_HEIGHT = 100

# Domain ảnh bài giải tin cậy — WHITELIST, bỏ qua toàn bộ bộ lọc AD
CONTENT_IMG_DOMAINS = {
    "img.loigiaihay.com",
    "img.hoc247.net",
    "vietjack.com",
    "img.vietjack.com",
    "tailieu.vn",
    "img.tailieu.vn",
}

# Domain quảng cáo — ảnh từ những domain này bị bỏ qua
AD_DOMAINS = {
    "googleadservices.com", "doubleclick.net", "googlesyndication.com",
    "adservice.google.com", "ads.yahoo.com", "facebook.com",
    "amazon-adsystem.com", "adsrvr.org", "adnxs.com", "moatads.com",
    "scorecardresearch.com", "quantserve.com", "taboola.com", "outbrain.com",
    "zalo.me", "vnpayads.com", "admicro.vn", "eclick.vn", "adtrue.com",
    "tuyensinh247.com",
    "ladicdn.com",       # CDN quảng cáo khóa học
}

# Từ khoá trong URL/alt/class/id → khả năng cao là quảng cáo
AD_KEYWORDS = re.compile(
    r"(advert|adsense|banner|sponsor|promo|popup|tracking|pixel|beacon|themes"
    r"|logo|favicon|avatar|author|comment|footer|header|nav|sidebar"
    r"|share|social|zalo|facebook|youtube|tiktok|widget|thumbnail-author)",
    re.IGNORECASE
)

# ══════════════════════════════════════════════════════════════════
# SELENIUM — fetch HTML sau khi JS render + scroll lazy-load
# ══════════════════════════════════════════════════════════════════

def _fetch_rendered_html(url: str) -> str | None:
    """
    Dùng Selenium headless Chrome để:
      1. Load trang
      2. Scroll toàn bộ từng bước nhỏ → trigger lazy-load ảnh
      3. Trả về page_source sau khi JS render xong
    Yêu cầu: pip install selenium webdriver-manager
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
    except ImportError:
        print("[WEB] Thiếu thư viện. Chạy: pip install selenium webdriver-manager")
        return None

    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )

    driver = None
    try:
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
        driver.set_page_load_timeout(60)
        driver.set_script_timeout(60)
        driver.get(url)
        time.sleep(3)

        try:
            total_height = driver.execute_script("return document.body.scrollHeight")
        except Exception:
            total_height = 10000

        for y in range(0, total_height + 500, 500):
            try:
                driver.execute_script(f"window.scrollTo(0, {y});")
            except Exception:
                pass
            time.sleep(0.2)

        try:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        except Exception:
            pass
        time.sleep(2)

        html = driver.page_source
        print(f"[WEB] Selenium OK — {len(html):,} ký tự HTML")
        return html

    except Exception as e:
        print(f"[WEB] Selenium lỗi: {e}")
        return None
    finally:
        if driver:
            driver.quit()


# ══════════════════════════════════════════════════════════════════
# LOAD MODELS (lazy)
# ══════════════════════════════════════════════════════════════════

_text_model = None
_clip_model = None
_clip_processor = None


def _get_text_model():
    global _text_model
    if _text_model is None:
        _text_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    return _text_model


def _get_clip_model():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPProcessor, CLIPModel
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _clip_model, _clip_processor


# ══════════════════════════════════════════════════════════════════
# CHUNK TEXT
# ══════════════════════════════════════════════════════════════════

_SPLIT_RE = re.compile(
    r"(Bài\s+\d+|Câu\s+\d+[\.:]\s*|Lời giải[\s:]*|Hướng dẫn giải[\s:]*"
    r"|Ví dụ\s+\d+|Dạng\s+\d+|Phương pháp[\s:]*|Giải:)",
    re.IGNORECASE
)


def semantic_chunk_text(text: str) -> list:
    parts = _SPLIT_RE.split(text)
    chunks = []
    current = ""
    for part in parts:
        if _SPLIT_RE.match(part):
            if current.strip() and len(current.strip()) >= MIN_CHUNK_CHARS:
                chunks.extend(_hard_cap(current.strip()))
            current = part
        else:
            current += "\n" + part
    if current.strip() and len(current.strip()) >= MIN_CHUNK_CHARS:
        chunks.extend(_hard_cap(current.strip()))
    return chunks


def _hard_cap(text: str) -> list:
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    sentences = re.split(r'(?<=[.!?])\s+|\n{2,}', text)
    result, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= MAX_CHUNK_CHARS:
            current += (" " if current else "") + sent
        else:
            if current.strip():
                result.append(current.strip())
            if len(sent) > MAX_CHUNK_CHARS:
                for i in range(0, len(sent), MAX_CHUNK_CHARS):
                    result.append(sent[i:i + MAX_CHUNK_CHARS])
                current = ""
            else:
                current = sent
    if current.strip():
        result.append(current.strip())
    return [c for c in result if len(c) >= MIN_CHUNK_CHARS]


# ══════════════════════════════════════════════════════════════════
# LỌC HÌNH QUẢNG CÁO
# ══════════════════════════════════════════════════════════════════

def _is_ad_image(img_tag, img_url: str) -> tuple[bool, str]:
    try:
        domain = urlparse(img_url).netloc.lower()
        for trusted in CONTENT_IMG_DOMAINS:
            if trusted in domain:
                return False, ""
        for ad_domain in AD_DOMAINS:
            if ad_domain in domain:
                return True, f"ad domain: {domain}"
    except Exception:
        pass

    if AD_KEYWORDS.search(img_url):
        return True, "ad keyword in URL"

    parent = img_tag.find_parent(["div", "a", "section"])
    if parent:
        parent_text = parent.get_text(separator=" ", strip=True).upper()
        if any(kw in parent_text for kw in ["HỌC NGAY", "ĐĂNG KÝ NGAY", "MUA NGAY", "GV ", "GIÁO VIÊN"]):
            return True, "ad: parent contains course promo text"

    for attr in ("alt", "class", "id", "title"):
        val = img_tag.get(attr, "")
        if isinstance(val, list):
            val = " ".join(val)
        if val and AD_KEYWORDS.search(val):
            return True, f"ad keyword in {attr}"

    try:
        w = int(img_tag.get("width", 0))
        h = int(img_tag.get("height", 0))
        if (w > 0 and w < MIN_IMG_WIDTH) or (h > 0 and h < MIN_IMG_HEIGHT):
            return True, f"too small in HTML: {w}x{h}"
        if w > 0 and h > 0 and w / h > 8:
            return True, f"banner ratio: {w}x{h}"
    except (ValueError, TypeError):
        pass

    return False, ""


def _check_image_size(img_bytes: bytes) -> tuple[bool, str]:
    try:
        img = Image.open(io.BytesIO(img_bytes))
        w, h = img.size
        if w < MIN_IMG_WIDTH or h < MIN_IMG_HEIGHT:
            return False, f"too small: {w}x{h}"
        if w / h > 8:
            return False, f"banner ratio: {w}x{h}"
        return True, f"{w}x{h}"
    except Exception as e:
        return False, f"cannot open: {e}"


# ══════════════════════════════════════════════════════════════════
# EXTRACT IMAGES TỪ HTML
# ══════════════════════════════════════════════════════════════════

def extract_images_from_html(html: str, page_url: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    results = []
    seen_urls = set()

    if not soup.body:
        return []

    import copy
    content_area = copy.copy(soup.body)
    for tag in content_area.find_all(["nav", "header", "footer", "script", "style", "noscript"]):
        tag.decompose()
    for tag in content_area.find_all(True, class_=re.compile(
        r"(sidebar|breadcrumb|related|comment|social|share)", re.I
    )):
        tag.decompose()

    for img in content_area.find_all("img"):
        src = (
            img.get("src") or
            img.get("data-src") or
            img.get("data-lazy-src") or
            img.get("data-original") or
            img.get("data-url") or
            ""
        )
        if not src:
            continue

        img_url = urljoin(page_url, src)

        if img_url.startswith("data:") or img_url.endswith(".svg"):
            continue
        if img_url in seen_urls:
            continue
        seen_urls.add(img_url)

        is_ad, reason = _is_ad_image(img, img_url)
        if is_ad:
            print(f"  [IMG] Bỏ ({reason}): {img_url[:80]}")
            continue

        alt = img.get("alt", "").strip()
        caption = ""
        fig = img.find_parent("figure")
        if fig:
            cap = fig.find("figcaption")
            if cap:
                caption = cap.get_text(strip=True)
        parent_text = ""
        parent = img.find_parent(["p", "div", "td", "li"])
        if parent:
            parent_text = parent.get_text(separator=" ", strip=True)[:200]

        context = " | ".join(filter(None, [alt, caption, parent_text]))
        results.append({"url": img_url, "alt": alt, "context": context})

    return results


# ══════════════════════════════════════════════════════════════════
# CLIP EMBEDDING
# ══════════════════════════════════════════════════════════════════

def _embed_image(img_bytes: bytes) -> list[float] | None:
    try:
        import torch
        model, processor = _get_clip_model()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats[0].tolist()
    except Exception as e:
        print(f"  [IMG] Lỗi CLIP embed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════
# INGEST TEXT
# ══════════════════════════════════════════════════════════════════

def _ingest_text(url: str, save_name: str, html: str, collection,
                 existing_text_ids: set) -> int:
    """
    Ingest text chunks từ HTML vào collection.
    Chỉ add những chunk_id chưa tồn tại trong existing_text_ids.
    """
    text = extract(html, include_comments=False, include_tables=True, no_fallback=False)
    if not text or len(text.strip()) < MIN_CHUNK_CHARS:
        print("[TEXT] Không trích xuất được nội dung text")
        return 0

    cache_path = Path(PROCESSED_DIR) / f"{save_name}.txt"
    cache_path.write_text(text, encoding="utf-8")
    print(f"[TEXT] Cache: {cache_path} ({len(text)} ký tự)")

    chunks = semantic_chunk_text(text)
    print(f"[TEXT] {len(chunks)} chunks")
    if not chunks:
        return 0

    documents, ids, metadatas = [], [], []
    skip_count = 0

    for i, chunk in enumerate(chunks):
        chunk_id = f"web_{save_name}_text_{i}"

        # ── SKIP nếu đã có trong ChromaDB ──
        if chunk_id in existing_text_ids:
            skip_count += 1
            continue

        documents.append(chunk)
        ids.append(chunk_id)
        metadatas.append({
            "source":        save_name,
            "source_folder": "web",
            "type":          "web",
            "chunk_index":   i,
            "url":           url,
        })

    print(f"[TEXT] {len(documents)} chunks mới | {skip_count} chunks skip (đã ingest)")

    if not documents:
        return 0

    embeddings = _get_text_model().encode(documents).tolist()
    collection.add(documents=documents, embeddings=embeddings, ids=ids, metadatas=metadatas)
    print(f"[TEXT] Đã ingest {len(documents)} chunks")
    return len(documents)


# ══════════════════════════════════════════════════════════════════
# INGEST IMAGES
# ══════════════════════════════════════════════════════════════════

def _ingest_images(url: str, save_name: str, html: str, collection,
                   existing_img_ids: set, force_img: bool = False) -> int:
    """
    Ingest hình ảnh từ HTML vào collection.
    Dùng existing_img_ids (lấy từ ChromaDB) để dedup — không cần file log.
    """
    img_save_dir = Path(IMAGE_DIR) / save_name
    img_save_dir.mkdir(parents=True, exist_ok=True)

    candidates = extract_images_from_html(html, url)
    print(f"[IMG] Tìm được {len(candidates)} ảnh ứng viên sau lọc HTML")

    count = 0
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Referer": url,
    }

    for img_info in candidates:
        img_url = img_info["url"]

        # Tính doc_id trước để check ChromaDB
        img_hash = hashlib.md5(img_url.encode()).hexdigest()[:10]
        doc_id   = f"web_{save_name}_img_{img_hash}"

        # ── SKIP nếu đã có trong ChromaDB (không cần file log) ──
        if doc_id in existing_img_ids and not force_img:
            print(f"  [IMG] Đã ingest, bỏ qua: {img_url[:80]}")
            continue

        try:
            resp = requests.get(img_url, headers=headers, timeout=15)
            resp.raise_for_status()
            img_bytes = resp.content
        except Exception as e:
            print(f"  [IMG] Không tải được ({e}): {img_url[:80]}")
            continue

        ok, size_info = _check_image_size(img_bytes)
        if not ok:
            print(f"  [IMG] Bỏ (kích thước — {size_info}): {img_url[:80]}")
            continue

        embedding = _embed_image(img_bytes)
        if embedding is None:
            continue

        ext        = Path(urlparse(img_url).path).suffix or ".jpg"
        local_path = img_save_dir / f"{img_hash}{ext}"
        local_path.write_bytes(img_bytes)

        # Xóa doc cũ nếu force
        if force_img and doc_id in existing_img_ids:
            try:
                collection.delete(ids=[doc_id])
            except Exception:
                pass

        doc_text = img_info["context"] or img_info["alt"] or f"[Hình từ {save_name}]"

        collection.add(
            documents=[doc_text],
            embeddings=[embedding],
            ids=[doc_id],
            metadatas=[{
                "source":        save_name,
                "source_folder": "web",
                "type":          "image",
                "url":           url,
                "img_url":       img_url,
                "local_path":    str(local_path),
                "alt":           img_info["alt"],
                "img_size":      size_info,
            }]
        )

        print(f"  [IMG] ✓ {size_info} → {local_path.name} | {img_info['alt'][:50]}")
        count += 1

    print(f"[IMG] Đã ingest {count} hình ảnh")
    return count


# ══════════════════════════════════════════════════════════════════
# MAIN INGEST FUNCTION
# ══════════════════════════════════════════════════════════════════

def ingest_web(url: str, save_name: str, force: bool = False):
    """
    Ingest 1 URL vào ChromaDB (text + hình ảnh).

    Dedup check thẳng vào ChromaDB — không phụ thuộc file log.
    Nếu DB bị xóa rồi chạy lại → ingest lại đúng, không bị skip oan.

    Logic:
      - Chưa ingest gì     → ingest text + hình
      - Text ✓, hình ✗     → chỉ ingest hình
      - Text ✓, hình ✓     → bỏ qua (hoặc ingest lại nếu force=True)

    Args:
        url:       URL trang lời giải
        save_name: Tên ngắn dùng làm ID (ví dụ: "loigiai_co_nang_1")
        force:     True = ingest lại hoàn toàn
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(IMAGE_DIR, exist_ok=True)

    client = chromadb.PersistentClient(path=DB_DIR)

    def _get_or_create(name):
        try:
            return client.get_collection(name)
        except Exception:
            return client.create_collection(name, metadata={"hnsw:space": "cosine"})

    text_collection  = _get_or_create(COLLECTION_NAME)
    image_collection = _get_or_create(COLLECTION_NAME + "_img")

    # ── CHECK THẲNG VÀO CHROMADB — source of truth duy nhất ─────
    # include=[] → chỉ lấy IDs, không fetch vectors/documents (nhanh)
    try:
        existing_text_ids = set(
            text_collection.get(where={"url": url}, include=[])["ids"]
        )
    except Exception:
        existing_text_ids = set()

    try:
        existing_img_ids = set(
            image_collection.get(where={"url": url}, include=[])["ids"]
        )
    except Exception:
        existing_img_ids = set()

    text_done = len(existing_text_ids) > 0
    img_done  = len(existing_img_ids) > 0

    if text_done and img_done and not force:
        print(f"[WEB] Đã ingest đủ text ({len(existing_text_ids)} chunks) "
              f"+ hình ({len(existing_img_ids)} ảnh), bỏ qua: {url}")
        return {"text": 0, "images": 0}

    if text_done and not img_done and not force:
        print(f"[WEB] Text đã có ({len(existing_text_ids)} chunks), chỉ ingest hình: {url}")
    elif not text_done:
        print(f"[WEB] Ingest mới: {url}")
    else:
        print(f"[WEB] Force re-ingest: {url}")

    # Fetch HTML 1 lần, dùng cho cả text lẫn hình
    print(f"[WEB] Đang tải trang (Selenium)...")
    html = _fetch_rendered_html(url)
    if not html:
        print("[WEB] Không tải được trang")
        return {"text": 0, "images": 0}

    n_text = 0
    n_img  = 0

    # ── Ingest TEXT ──────────────────────────────────────────────
    if not text_done or force:
        if force and text_done:
            try:
                old_ids = list(existing_text_ids)
                if old_ids:
                    text_collection.delete(ids=old_ids)
                    print(f"[TEXT] Đã xóa {len(old_ids)} chunks cũ")
                existing_text_ids = set()   # reset sau khi xóa
            except Exception as e:
                print(f"[TEXT] Lỗi khi xóa chunks cũ: {e}")

        n_text = _ingest_text(url, save_name, html, text_collection, existing_text_ids)

    # ── Ingest IMAGES ─────────────────────────────────────────────
    if force and img_done:
        # Xóa ảnh cũ trước khi re-ingest
        try:
            old_img_ids = list(existing_img_ids)
            if old_img_ids:
                image_collection.delete(ids=old_img_ids)
                print(f"[IMG] Đã xóa {len(old_img_ids)} ảnh cũ")
            existing_img_ids = set()   # reset sau khi xóa
        except Exception as e:
            print(f"[IMG] Lỗi khi xóa ảnh cũ: {e}")

    n_img = _ingest_images(
        url, save_name, html, image_collection,
        existing_img_ids, force_img=force
    )

    print(f"\n[WEB] Tổng: {n_text} text chunks + {n_img} hình từ: {url}")
    return {"text": n_text, "images": n_img}


# ══════════════════════════════════════════════════════════════════
# BATCH INGEST
# ══════════════════════════════════════════════════════════════════

def ingest_web_batch(url_dict: dict, force: bool = False):
    """
    Ingest nhiều URL cùng lúc.

    Args:
        url_dict: {"save_name": "https://..."}
        force:    True = ingest lại hết

    Ví dụ:
        ingest_web_batch({
            "loigiai_co_nang_1": "https://loigiaihay.com/...",
            "hoc247_dien_truong": "https://hoc247.net/...",
        })
    """
    total_text = total_img = 0
    for save_name, url in url_dict.items():
        result = ingest_web(url, save_name, force=force)
        total_text += result["text"]
        total_img  += result["images"]
    print(f"\n[WEB BATCH] Tổng: {total_text} text chunks + {total_img} hình "
          f"từ {len(url_dict)} URL")


# ══════════════════════════════════════════════════════════════════
# CLI — python web_ingest.py <url> <save_name> [--force]
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        ingest_web(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4 and sys.argv[3] == "--force":
        ingest_web(sys.argv[1], sys.argv[2], force=True)
    else:
        print("Dùng: python web_ingest.py <url> <save_name> [--force]")
        print("\nVí dụ:")
        print('  python web_ingest.py "https://loigiaihay.com/bai-tap" "loigiai_co_nang"')
        print('  python web_ingest.py "https://hoc247.net/..." "hoc247_dao_dong" --force')
        print("\nBatch ingest trong Python:")
        print("""
  from web_ingest import ingest_web_batch
  ingest_web_batch({
      "loigiai_co_nang": "https://loigiaihay.com/...",
      "hoc247_dao_dong":  "https://hoc247.net/...",
  })
""")