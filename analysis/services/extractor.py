import httpx
from bs4 import BeautifulSoup
import os
import re
import tempfile
from urllib.parse import urlparse
import numpy as np
import cv2

os.environ.setdefault("PADDLE_PDX_CACHE_HOME", r"C:\Users\jhjh1\.codex\memories\paddlex_cache")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    _EASYOCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    _PADDLEOCR_AVAILABLE = True
except ImportError:
    _PADDLEOCR_AVAILABLE = False

_easy_reader = None
_paddle_reader = None
OCR_PRIMARY_MIN_SCORE = 0.45
OCR_PRIMARY_MIN_TEXT_LEN = 12
OCR_ENGINE = os.getenv("OCR_ENGINE", "easy").strip().lower()
OCR_MAX_SIDE = int(os.getenv("OCR_MAX_SIDE", "2200"))
OCR_ROW_MAX_SIDE = int(os.getenv("OCR_ROW_MAX_SIDE", "1800"))
OCR_PRIMARY_VARIANT_LIMIT = max(1, int(os.getenv("OCR_PRIMARY_VARIANT_LIMIT", "4")))
OCR_FALLBACK_VARIANT_LIMIT = max(1, int(os.getenv("OCR_FALLBACK_VARIANT_LIMIT", "2")))
OCR_ROW_CROP_MIN_SCORE = float(os.getenv("OCR_ROW_CROP_MIN_SCORE", "0.38"))
OCR_EARLY_EXIT_SCORE = float(os.getenv("OCR_EARLY_EXIT_SCORE", "0.72"))
OCR_EARLY_EXIT_TEXT_LEN = int(os.getenv("OCR_EARLY_EXIT_TEXT_LEN", "24"))
OCR_CORRECTIONS = {
    "피부릍": "피부를",
    "피부륻": "피부를",
    "피부률": "피부를",
    "주릅": "주름",
    "주름개선": "주름 개선",
    "효과보장입나다": "효과 보장입니다",
    "효과보장": "효과 보장",
    "아토피를료": "아토피 치료",
    "논코메도제나": "논코메도제닉",
    "논코메도제닉야": "논코메도제닉",
    "안티에이징 ": "안티에이징 ",
    "피부치밀도도": "피부치밀도",
    "인체적용시힘 완료": "인체적용시험 완료",
    "인체적용시험 완로": "인체적용시험 완료",
    "인체적용시혐": "인체적용시험",
    "기미잡리": "기미잡티",
    "기미 잡리": "기미 잡티",
    "잠티": "잡티",
    "탄력저하": "탄력 저하",
    "지쳐보이틀": "지쳐 보이는",
    "지쳐보이는": "지쳐 보이는",
    "지처보이는": "지쳐 보이는",
    "피부결": "피부 결",
    "로리셋": "로 리셋",
    "리셋하세오": "리셋하세요",
    "리셋하서요": "리셋하세요",
    "쎄럼": "세럼",
    "앰풀": "앰플",
    "크림입나다": "크림입니다",
}


def _get_easy_reader():
    global _easy_reader
    if _easy_reader is None:
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except Exception:
            use_gpu = False
        _easy_reader = easyocr.Reader(["ko", "en"], gpu=use_gpu)
    return _easy_reader


def _get_paddle_reader():
    global _paddle_reader
    if _paddle_reader is None:
        _paddle_reader = PaddleOCR(
            lang="korean",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    return _paddle_reader


def _normalize_ocr_text(text: str) -> str:
    cleaned = text.replace("\r", "\n")
    cleaned = re.sub(r"[|¦]+", " ", cleaned)
    cleaned = re.sub(r"[`´‘’“”]+", "", cleaned)
    cleaned = cleaned.replace(" / ", "/")
    cleaned = cleaned.replace(" mm", "mm")
    cleaned = re.sub(r"(?<=\d)\.\s+(?=\d)", ".", cleaned)
    cleaned = re.sub(r"([가-힣A-Za-z0-9])([,.:;!?])", r"\1\2 ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    for src, dst in OCR_CORRECTIONS.items():
        cleaned = cleaned.replace(src, dst)
    cleaned = re.sub(r"(?<=\d)\s*/\s*(?=\d)", "/", cleaned)
    cleaned = re.sub(r"(?<=\d)\s*mm", "mm", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?<=[가-힣])\s*/\s*(?=[가-힣])", "/", cleaned)
    cleaned = re.sub(r"0\.\s*5mm/15mm", "0.5mm/1.5mm", cleaned)
    cleaned = re.sub(r"0\.5mm/15mm", "0.5mm/1.5mm", cleaned)
    cleaned = re.sub(r"0\.5mm/1Smm", "0.5mm/1.5mm", cleaned)
    cleaned = re.sub(r"인체 적용 시험 완료", "인체적용시험 완료", cleaned)
    lines = [line.strip() for line in cleaned.split("\n") if len(line.strip()) > 1]
    return "\n".join(lines).strip()


def _ocr_quality_score(text: str, confidences: list[float]) -> float:
    if not text:
        return 0.0
    avg_conf = sum(confidences) / max(len(confidences), 1)
    text_len = len(text)
    korean_chars = sum("가" <= ch <= "힣" for ch in text)
    alpha_num = sum(ch.isalnum() for ch in text)
    special_chars = text_len - alpha_num - text.count(" ")
    korean_ratio = korean_chars / max(text_len, 1)
    alpha_ratio = alpha_num / max(text_len, 1)
    special_penalty = min(special_chars / max(text_len, 1), 0.35)
    length_bonus = min(text_len / 80.0, 1.0) * 0.15
    return round(
        (avg_conf * 0.55) + (korean_ratio * 0.15) + (alpha_ratio * 0.15) + length_bonus - special_penalty,
        4,
    )


def _decode_image(image_bytes: bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _resize_to_max_side(image, max_side: int):
    if image is None or max_side <= 0:
        return image
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    return cv2.resize(
        image,
        (max(1, int(w * scale)), max(1, int(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _encode_png(image) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    return buf.tobytes() if ok else b""


def _extract_text_rows(image_bytes: bytes) -> list[bytes]:
    """
    체크리스트/배너형 이미지에서 텍스트 줄 단위로 분리한다.
    """
    rows: list[bytes] = []
    try:
        img = _decode_image(image_bytes)
        if img is None:
            return rows
        img = _resize_to_max_side(img, OCR_ROW_MAX_SIDE)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if min(gray.shape[:2]) < 700:
            scale = min(2.2, 800 / max(1, min(gray.shape[:2])))
            gray = cv2.resize(
                gray,
                (int(gray.shape[1] * scale), int(gray.shape[0] * scale)),
                interpolation=cv2.INTER_CUBIC,
            )

        binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        projection = np.sum(binary > 0, axis=1)
        min_pixels = max(20, int(binary.shape[1] * 0.03))

        bands: list[tuple[int, int]] = []
        in_band = False
        start = 0
        for idx, value in enumerate(projection):
            if value >= min_pixels and not in_band:
                start = idx
                in_band = True
            elif value < min_pixels and in_band:
                end = idx
                if end - start >= 18:
                    bands.append((start, end))
                in_band = False
        if in_band:
            end = len(projection) - 1
            if end - start >= 18:
                bands.append((start, end))

        merged: list[tuple[int, int]] = []
        for start, end in bands:
            if not merged or start - merged[-1][1] > 18:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], end)

        for start, end in merged:
            pad = 10
            top = max(start - pad, 0)
            bottom = min(end + pad, gray.shape[0])
            crop = gray[top:bottom, :]
            row_binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            encoded = _encode_png(row_binary)
            if encoded:
                rows.append(encoded)
    except Exception:
        return rows

    return rows


def _build_ocr_variants(image_bytes: bytes) -> list[tuple[str, bytes]]:
    """
    원본 + 여러 전처리 버전을 만들어 OCR 성공률을 높인다.
    """
    variants: list[tuple[str, bytes]] = [("original", image_bytes)]
    try:
        img = _decode_image(image_bytes)
        if img is None:
            return variants
        img = _resize_to_max_side(img, OCR_MAX_SIDE)

        h, w = img.shape[:2]
        min_side = min(h, w)
        scale = 1.0
        if min_side < 700:
            scale = min(2.0, 700 / max(1, min_side))
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)
        variants.append(("grayscale_clahe", _encode_png(contrast)))

        blurred = cv2.bilateralFilter(contrast, d=7, sigmaColor=75, sigmaSpace=75)
        adaptive = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9
        )
        variants.append(("adaptive_threshold", _encode_png(adaptive)))

        adaptive_inv = cv2.bitwise_not(adaptive)
        variants.append(("adaptive_threshold_inverted", _encode_png(adaptive_inv)))

        sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        sharpened = cv2.filter2D(contrast, -1, sharpen_kernel)
        variants.append(("sharpened", _encode_png(sharpened)))

        otsu = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        variants.append(("otsu_threshold", _encode_png(otsu)))

        otsu_inv = cv2.bitwise_not(otsu)
        variants.append(("otsu_threshold_inverted", _encode_png(otsu_inv)))

        # Small, low-contrast ad text benefits from stronger enlargement.
        super_res = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        super_res = cv2.GaussianBlur(super_res, (0, 0), 0.6)
        super_res = cv2.addWeighted(super_res, 1.6, cv2.GaussianBlur(super_res, (0, 0), 2.0), -0.6, 0)
        variants.append(("super_res_sharpened", _encode_png(super_res)))
    except Exception:
        return variants

    return [(name, data) for name, data in variants if data]


def _select_ocr_variants(variants: list[tuple[str, bytes]], limit: int) -> list[tuple[str, bytes]]:
    preferred_order = {
        "original": 0,
        "grayscale_clahe": 1,
        "sharpened": 2,
        "super_res_sharpened": 3,
        "adaptive_threshold": 4,
        "otsu_threshold": 5,
        "adaptive_threshold_inverted": 6,
        "otsu_threshold_inverted": 7,
    }
    ordered = sorted(variants, key=lambda item: preferred_order.get(item[0], 99))
    return ordered[: max(1, limit)]


def _is_strong_ocr_candidate(candidate: dict) -> bool:
    return (
        candidate.get("score", 0.0) >= OCR_EARLY_EXIT_SCORE
        and len(candidate.get("text", "")) >= OCR_EARLY_EXIT_TEXT_LEN
    )


def _run_easyocr_once(
    image_bytes: bytes,
    *,
    text_threshold: float,
    low_text: float,
    link_threshold: float,
    min_confidence: float,
    mag_ratio: float,
) -> dict:
    reader = _get_easy_reader()
    result = reader.readtext(
        image_bytes,
        detail=1,
        paragraph=False,
        text_threshold=text_threshold,
        low_text=low_text,
        link_threshold=link_threshold,
        mag_ratio=mag_ratio,
        contrast_ths=0.05,
        adjust_contrast=0.7,
        width_ths=0.9,
    )
    if not result:
        return {"text": "", "score": 0.0, "confidences": [], "raw_count": 0}

    result.sort(key=lambda x: (x[0][0][1], x[0][0][0]))
    lines = [item[1] for item in result if item[2] >= min_confidence]
    confidences = [float(item[2]) for item in result if item[2] >= min_confidence]
    text = _normalize_ocr_text("\n".join(lines))
    return {
        "text": text,
        "score": _ocr_quality_score(text, confidences),
        "confidences": confidences,
        "raw_count": len(result),
    }


def _run_paddleocr_once(image_bytes: bytes, *, min_confidence: float) -> dict:
    reader = _get_paddle_reader()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        result = reader.predict(tmp_path)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if not result:
        return {"text": "", "score": 0.0, "confidences": [], "raw_count": 0}

    page = result[0] if isinstance(result, list) else result
    texts = list(page.get("rec_texts") or [])
    scores = [float(score) for score in (page.get("rec_scores") or [])]
    dt_polys = list(page.get("dt_polys") or [])

    rows = []
    for idx, text in enumerate(texts):
        score = scores[idx] if idx < len(scores) else 0.0
        if score < min_confidence:
            continue
        poly = dt_polys[idx] if idx < len(dt_polys) else None
        if poly is not None:
            try:
                poly_arr = np.asarray(poly)
                y = int(np.min(poly_arr[:, 1]))
                x = int(np.min(poly_arr[:, 0]))
            except Exception:
                y = idx
                x = 0
        else:
            y = idx
            x = 0
        rows.append((y, x, str(text), score))

    if not rows:
        return {"text": "", "score": 0.0, "confidences": [], "raw_count": len(texts)}

    rows.sort(key=lambda item: (item[0], item[1]))
    lines = [item[2] for item in rows]
    confidences = [item[3] for item in rows]
    text = _normalize_ocr_text("\n".join(lines))
    return {
        "text": text,
        "score": _ocr_quality_score(text, confidences),
        "confidences": confidences,
        "raw_count": len(texts),
    }


def _extract_with_easyocr_variants(image_bytes: bytes) -> str:
    variants = _build_ocr_variants(image_bytes)
    best = {"text": "", "score": 0.0}

    primary_settings = {
        "text_threshold": 0.4,
        "low_text": 0.3,
        "link_threshold": 0.3,
        "min_confidence": 0.35,
        "mag_ratio": 1.7,
    }
    primary_variants = _select_ocr_variants(variants, OCR_PRIMARY_VARIANT_LIMIT)
    for name, variant_bytes in primary_variants:
        candidate = _run_easyocr_once(variant_bytes, **primary_settings)
        if candidate["score"] > best["score"]:
            best = {**candidate, "variant": name}
        if _is_strong_ocr_candidate(best):
            return best["text"]

    if best["score"] < OCR_ROW_CROP_MIN_SCORE:
        row_texts = []
        row_scores = []
        for row_bytes in _extract_text_rows(image_bytes):
            candidate = _run_easyocr_once(row_bytes, **primary_settings)
            if candidate["text"]:
                row_texts.append(candidate["text"])
                row_scores.append(candidate["score"])
        if row_texts:
            row_joined = _normalize_ocr_text("\n".join(row_texts))
            row_score = (sum(row_scores) / max(len(row_scores), 1)) + min(len(row_texts) / 20.0, 0.2)
            if row_score > best["score"] and len(row_joined) >= len(best["text"]) * 0.7:
                best = {"text": row_joined, "score": round(row_score, 4), "variant": "row_crops"}

    if best["score"] >= OCR_PRIMARY_MIN_SCORE and len(best["text"]) >= OCR_PRIMARY_MIN_TEXT_LEN:
        return best["text"]

    # Fallback: relax thresholds and retry on the top variants.
    fallback_settings = {
        "text_threshold": 0.22,
        "low_text": 0.12,
        "link_threshold": 0.18,
        "min_confidence": 0.22,
        "mag_ratio": 2.0,
    }
    fallback_variants = _select_ocr_variants(variants, OCR_FALLBACK_VARIANT_LIMIT)
    for name, variant_bytes in fallback_variants:
        candidate = _run_easyocr_once(variant_bytes, **fallback_settings)
        if candidate["score"] > best["score"]:
            best = {**candidate, "variant": name}
        if _is_strong_ocr_candidate(best):
            return best["text"]

    return best["text"]


def _extract_with_paddleocr_variants(image_bytes: bytes) -> str:
    variants = _build_ocr_variants(image_bytes)
    best = {"text": "", "score": 0.0}

    primary_min_confidence = 0.45
    primary_variants = _select_ocr_variants(variants, OCR_PRIMARY_VARIANT_LIMIT)
    for name, variant_bytes in primary_variants:
        candidate = _run_paddleocr_once(variant_bytes, min_confidence=primary_min_confidence)
        if candidate["score"] > best["score"]:
            best = {**candidate, "variant": name}
        if _is_strong_ocr_candidate(best):
            return best["text"]

    if best["score"] >= OCR_PRIMARY_MIN_SCORE and len(best["text"]) >= OCR_PRIMARY_MIN_TEXT_LEN:
        return best["text"]

    if best["score"] < OCR_ROW_CROP_MIN_SCORE:
        row_texts = []
        row_scores = []
        for row_bytes in _extract_text_rows(image_bytes):
            candidate = _run_paddleocr_once(row_bytes, min_confidence=primary_min_confidence)
            if candidate["text"]:
                row_texts.append(candidate["text"])
                row_scores.append(candidate["score"])
        if row_texts:
            row_joined = _normalize_ocr_text("\n".join(row_texts))
            row_score = (sum(row_scores) / max(len(row_scores), 1)) + min(len(row_texts) / 20.0, 0.2)
            if row_score > best["score"] and len(row_joined) >= len(best["text"]) * 0.7:
                best = {"text": row_joined, "score": round(row_score, 4), "variant": "row_crops"}

    fallback_min_confidence = 0.25
    fallback_variants = _select_ocr_variants(variants, OCR_FALLBACK_VARIANT_LIMIT)
    for name, variant_bytes in fallback_variants:
        candidate = _run_paddleocr_once(variant_bytes, min_confidence=fallback_min_confidence)
        if candidate["score"] > best["score"]:
            best = {**candidate, "variant": name}
        if _is_strong_ocr_candidate(best):
            return best["text"]

    return best["text"]



try:
    import kss
    _KSS_AVAILABLE = True
except ImportError:
    _KSS_AVAILABLE = False

try:
    from playwright.async_api import async_playwright
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False

# ── 로그인 필요 사이트 (Playwright로도 불가) ──────────────────
LOGIN_REQUIRED_DOMAINS = {
    "youtube.com":   "유튜브는 크롤링이 차단되어 있습니다. 영상 설명란 텍스트를 복사해서 텍스트 분석을 이용해주세요.",
    "youtu.be":      "유튜브는 크롤링이 차단되어 있습니다. 영상 설명란 텍스트를 복사해서 텍스트 분석을 이용해주세요.",
    "instagram.com": "인스타그램은 로그인이 필요하여 크롤링이 불가합니다. 광고 문구를 복사해서 텍스트 분석을 이용해주세요.",
    "facebook.com":  "페이스북은 로그인이 필요하여 크롤링이 불가합니다. 광고 문구를 복사해서 텍스트 분석을 이용해주세요.",
    "kakao.com":     "카카오는 로그인이 필요하여 크롤링이 불가합니다. 광고 문구를 복사해서 텍스트 분석을 이용해주세요.",
}

# ── Playwright로 재시도할 사이트 (JS 렌더링 필요) ──────────────
PLAYWRIGHT_DOMAINS = {
    "coupang.com",
    "smartstore.naver.com",
    "oliveyoung.co.kr",
    "musinsa.com",
    "kurly.com",
}

# ── 사이트별 본문 추출 CSS 선택자 ────────────────────────────
SITE_SELECTORS = {
    "blog.naver.com":   ["div.se-main-container", "div#postViewArea", "div.post-view"],
    "m.blog.naver.com": ["div.se-main-container", "div#postViewArea", "div.post-view", "div.blog_content"],
    "tistory.com":      ["div.entry-content", "article", "div.article-view"],
    "brunch.co.kr":     ["div.wrap_body", "article"],
    "oliveyoung.co.kr": ["div.prd-detail", "div#artcInfo", "div.prd_detail_box"],
    "default":          ["article", "main", "div.content", "div.product-detail",
                         "div#content", "div.detail", "section.content"],
}


def _get_domain(url: str) -> str:
    """URL에서 도메인 추출"""
    parsed = urlparse(url)
    domain = parsed.netloc.replace("www.", "")
    return domain


def _normalize_url(url: str) -> str:
    """네이버 블로그 등 본문 접근이 어려운 URL을 크롤링 친화적으로 변환"""
    if "blog.naver.com" in url:
        import re as _re
        m = _re.search(r'blogId=([^&]+)&logNo=(\d+)', url)
        if m:
            return f"https://m.blog.naver.com/{m.group(1)}/{m.group(2)}"
        url = url.replace("blog.naver.com", "m.blog.naver.com")
    return url


def _check_login_required(url: str) -> str | None:
    """로그인 필요 사이트면 안내 메시지 반환, 아니면 None"""
    domain = _get_domain(url)
    for blocked, message in LOGIN_REQUIRED_DOMAINS.items():
        if blocked in domain:
            return message
    return None


def _needs_playwright(url: str) -> bool:
    """Playwright 재시도가 필요한 사이트인지 확인"""
    domain = _get_domain(url)
    return any(d in domain for d in PLAYWRIGHT_DOMAINS)


def _extract_main_text(soup: BeautifulSoup, domain: str) -> str:
    """사이트별 선택자로 본문 추출, 없으면 전체 텍스트 추출"""
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "iframe", "noscript", "svg",
                     "button", "input", "select", "form"]):
        tag.decompose()

    for selector in [
        ".blog_menu", ".gnb", ".lnb", ".snb",
        ".post_menu", ".post_toolbar", ".post_share",
        "#header", "#footer", "#gnb", "#lnb",
        ".comment_area", ".related_posts",
        "[class*='menu']", "[class*='toolbar']",
        "[class*='navigation']", "[id*='menu']",
    ]:
        for el in soup.select(selector):
            el.decompose()

    selectors = SITE_SELECTORS.get(domain, []) + SITE_SELECTORS["default"]
    for selector in selectors:
        element = soup.select_one(selector)
        if element:
            text = element.get_text(separator="\n")
            if len(text.strip()) > 100:
                return text.strip()

    return soup.get_text(separator="\n").strip()


def _clean_text(text: str) -> str:
    """추출된 텍스트 정리"""
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\t+", " ", text)
    lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 1]
    return "\n".join(lines)


async def extract_from_url(url: str) -> str:
    """URL에서 광고 본문 텍스트 추출
    1단계: httpx + BeautifulSoup (빠름, 최대 8초)
    2단계: Playwright 재시도 필요한 사이트만 (JS 렌더링, 최대 15초)
    """
    login_msg = _check_login_required(url)
    if login_msg:
        raise ValueError(login_msg)

    if _needs_playwright(url) and _PLAYWRIGHT_AVAILABLE:
        text = await _extract_with_playwright(url)
    else:
        text = await _extract_with_httpx(url)

        if len(text) < 200 and _PLAYWRIGHT_AVAILABLE:
            playwright_text = await _extract_with_playwright(url)
            if len(playwright_text) > len(text):
                text = playwright_text

    if not text or len(text) < 20:
        raise ValueError(
            "페이지에서 분석할 텍스트를 찾지 못했습니다. "
            "광고 문구를 직접 복사해서 텍스트 분석을 이용해주세요."
        )

    return text


async def _extract_with_httpx(url: str) -> str:
    """httpx + BeautifulSoup으로 텍스트 추출 (1단계)"""
    url = _normalize_url(url)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }

    try:
        async with httpx.AsyncClient(
            timeout=8.0,
            headers=headers,
            follow_redirects=True,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403:
            return ""
        elif e.response.status_code == 404:
            raise ValueError("페이지를 찾을 수 없습니다. URL을 다시 확인해주세요.")
        else:
            return ""
    except httpx.TimeoutException:
        raise ValueError("페이지 로딩 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.")
    except Exception:
        return ""

    domain = _get_domain(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = _extract_main_text(soup, domain)
    return _clean_text(text)


async def _extract_with_playwright(url: str) -> str:
    """Playwright로 JS 렌더링 후 텍스트 추출 (2단계)"""
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            )
            page = await context.new_page()

            await page.route(
                "**/*.{png,jpg,jpeg,gif,svg,woff,woff2,ttf,otf,mp4,mp3,webp}",
                lambda r: r.abort()
            )

            await page.goto(url, timeout=12000, wait_until="domcontentloaded")

            try:
                await page.wait_for_load_state("networkidle", timeout=3000)
            except Exception:
                pass

            html = await page.content()
            await browser.close()

        domain = _get_domain(url)
        soup = BeautifulSoup(html, "html.parser")
        text = _extract_main_text(soup, domain)
        return _clean_text(text)

    except Exception:
        return ""


def extract_from_image(image_bytes: bytes) -> str:
    """
    이미지에서 텍스트 추출.
    - 여러 전처리 버전을 만든 뒤 EasyOCR로 점수화
    - 1차 결과 품질이 낮으면 완화된 threshold로 fallback 재시도
    - 결과는 간단한 OCR 오타 교정을 거쳐 반환
    """
    engine = OCR_ENGINE
    if engine not in {"auto", "paddle", "easy"}:
        engine = "auto"

    if engine == "paddle":
        if not _PADDLEOCR_AVAILABLE:
            raise RuntimeError("PaddleOCR가 설치되어 있지 않습니다. pip install paddleocr 를 실행하세요.")
        try:
            return _extract_with_paddleocr_variants(image_bytes)
        except Exception as e:
            raise RuntimeError(f"PaddleOCR 텍스트 추출 실패: {e}")

    if engine == "easy":
        if not _EASYOCR_AVAILABLE:
            raise RuntimeError("EasyOCR가 설치되어 있지 않습니다. pip install easyocr 를 실행하세요.")
        try:
            return _extract_with_easyocr_variants(image_bytes)
        except Exception as e:
            raise RuntimeError(f"EasyOCR 텍스트 추출 실패: {e}")

    if _EASYOCR_AVAILABLE:
        try:
            text = _extract_with_easyocr_variants(image_bytes)
            if len(text.strip()) >= OCR_PRIMARY_MIN_TEXT_LEN:
                return text
        except Exception:
            pass

    if _PADDLEOCR_AVAILABLE:
        try:
            text = _extract_with_paddleocr_variants(image_bytes)
            if len(text.strip()) >= OCR_PRIMARY_MIN_TEXT_LEN:
                return text
        except Exception:
            pass

    if not _EASYOCR_AVAILABLE:
        raise RuntimeError("사용 가능한 OCR 엔진이 없습니다. paddleocr 또는 easyocr를 설치하세요.")
    try:
        return _extract_with_easyocr_variants(image_bytes)
    except Exception as e:
        raise RuntimeError(f"EasyOCR 텍스트 추출 실패: {e}")


def split_sentences(text: str) -> list[str]:
    """텍스트를 문장 단위로 분리 (kss 우선, 없으면 정규식 폴백)"""
    if _KSS_AVAILABLE:
        return _merge_sentence_fragments(_split_with_kss(text))
    return _merge_sentence_fragments(_split_with_regex(text))


def _split_with_kss(text: str) -> list[str]:
    """kss 라이브러리를 사용한 한국어 문장 분리"""
    try:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        sentences = []
        for para in paragraphs:
            split = kss.split_sentences(para, backend="punct")
            sentences.extend(split)
        return [s.strip() for s in sentences if len(s.strip()) > 3]
    except Exception:
        return _split_with_regex(text)


def _split_with_regex(text: str) -> list[str]:
    """정규식 기반 문장 분리 (폴백용)"""
    sentences = re.split(r"(?<=[.!?])\s+|[\n]+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 3]


_SENTENCE_END_RE = re.compile(
    r"([.!?。！？]$|"
    r"(습니다|합니다|됩니다|했습니다|드립니다|입니다|합니다|하세요|돼요|해요|나요|어요|아요|다|요)$)"
)


def _looks_complete_sentence(text: str) -> bool:
    return bool(_SENTENCE_END_RE.search(text.strip()))


def _is_short_ocr_fragment(text: str) -> bool:
    """
    OCR 광고 이미지는 카피가 줄 단위로 끊겨 들어오는 경우가 많다.
    완결 문장이 아닌 짧은 줄은 다음 줄과 합쳐 문맥을 보존한다.
    """
    s = text.strip()
    if _looks_complete_sentence(s):
        return False

    token_count = len(s.split())
    compact_len = len(re.sub(r"\s+", "", s))
    has_space = " " in s

    return (
        compact_len <= 18
        or (not has_space and compact_len <= 28)
        or (token_count <= 5 and compact_len <= 34)
    )


def _flush_fragment_buffer(buffer: list[str], merged: list[str]) -> None:
    if buffer:
        merged.append(" ".join(buffer).strip())
        buffer.clear()


def _merge_sentence_fragments(sentences: list[str]) -> list[str]:
    """
    줄바꿈 기반 OCR 조각을 의미 단위로 재결합한다.
    예: "기미잡티\\n탄력저하\\n...\\n리셋하세요" -> 한 카피 문장
    """
    merged: list[str] = []
    buffer: list[str] = []
    max_merged_len = 110

    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue

        if not buffer:
            if _is_short_ocr_fragment(s):
                buffer.append(s)
            else:
                merged.append(s)
            continue

        candidate_len = len(" ".join(buffer + [s]))
        if candidate_len <= max_merged_len and (
            _is_short_ocr_fragment(s) or not _looks_complete_sentence(buffer[-1])
        ):
            buffer.append(s)
            if _looks_complete_sentence(s) or candidate_len >= max_merged_len:
                _flush_fragment_buffer(buffer, merged)
            continue

        _flush_fragment_buffer(buffer, merged)
        if _is_short_ocr_fragment(s):
            buffer.append(s)
        else:
            merged.append(s)

    _flush_fragment_buffer(buffer, merged)
    return merged
