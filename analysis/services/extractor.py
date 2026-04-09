import httpx
from bs4 import BeautifulSoup
from PIL import Image
import io
import re
import os
from urllib.parse import urlparse

try:
    import easyocr
    _EASYOCR_AVAILABLE = True
except ImportError:
    _EASYOCR_AVAILABLE = False

_easyocr_reader = None


def _get_easyocr_reader() -> "easyocr.Reader":
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import torch
            use_gpu = torch.cuda.is_available()
        except ImportError:
            use_gpu = False
        _easyocr_reader = easyocr.Reader(["ko", "en"], gpu=use_gpu)
    return _easyocr_reader


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
    """이미지에서 EasyOCR로 텍스트 추출 (한국어 + 영어)"""
    if not _EASYOCR_AVAILABLE:
        raise RuntimeError("EasyOCR가 설치되어 있지 않습니다. pip install easyocr 를 실행하세요.")
    try:
        reader = _get_easyocr_reader()
        result = reader.readtext(image_bytes)
        if not result:
            return ""
        # 위→아래, 왼→오른쪽 순으로 정렬 (bbox 좌상단 기준)
        result.sort(key=lambda x: (x[0][0][1], x[0][0][0]))
        lines = [item[1] for item in result if item[2] > 0.3]
        return "\n".join(lines).strip()
    except Exception as e:
        raise RuntimeError(f"이미지 텍스트 추출 실패: {e}")


def split_sentences(text: str) -> list[str]:
    """텍스트를 문장 단위로 분리 (kss 우선, 없으면 정규식 폴백)"""
    if _KSS_AVAILABLE:
        return _split_with_kss(text)
    return _split_with_regex(text)


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