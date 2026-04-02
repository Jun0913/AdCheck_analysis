import httpx
from bs4 import BeautifulSoup
from PIL import Image
import pytesseract
import io
import re
import os
from urllib.parse import urlparse

# Windows Tesseract 경로 직접 지정
if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

try:
    import kss
    _KSS_AVAILABLE = True
except ImportError:
    _KSS_AVAILABLE = False

# ── 크롤링 차단 사이트 목록 ──────────────────────────────────
# 로그인 필요 또는 강력한 봇 차단으로 크롤링 불가한 사이트
BLOCKED_DOMAINS = {
    "coupang.com":              "쿠팡은 크롤링이 차단되어 있습니다. 광고 문구를 직접 복사해서 텍스트 분석을 이용해주세요.",
    "youtube.com":              "유튜브는 크롤링이 차단되어 있습니다. 영상 설명란 텍스트를 복사해서 텍스트 분석을 이용해주세요.",
    "youtu.be":                 "유튜브는 크롤링이 차단되어 있습니다. 영상 설명란 텍스트를 복사해서 텍스트 분석을 이용해주세요.",
    "instagram.com":            "인스타그램은 로그인이 필요하여 크롤링이 불가합니다. 광고 문구를 복사해서 텍스트 분석을 이용해주세요.",
    "smartstore.naver.com":     "네이버 스마트스토어는 크롤링이 차단되어 있습니다. 상품 설명을 복사해서 텍스트 분석을 이용해주세요.",
    "oliveyoung.co.kr":         "올리브영은 크롤링이 차단되어 있습니다. 상품 설명을 복사해서 텍스트 분석을 이용해주세요.",
    "kakao.com":                "카카오는 로그인이 필요하여 크롤링이 불가합니다. 광고 문구를 복사해서 텍스트 분석을 이용해주세요.",
    "facebook.com":             "페이스북은 로그인이 필요하여 크롤링이 불가합니다. 광고 문구를 복사해서 텍스트 분석을 이용해주세요.",
}

# ── 사이트별 본문 추출 CSS 선택자 ────────────────────────────
SITE_SELECTORS = {
    "blog.naver.com":   ["div.se-main-container", "div#postViewArea", "div.post-view"],
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


def _check_blocked(url: str) -> str | None:
    """차단된 사이트면 안내 메시지 반환, 아니면 None"""
    domain = _get_domain(url)
    for blocked, message in BLOCKED_DOMAINS.items():
        if blocked in domain:
            return message
    return None


def _extract_main_text(soup: BeautifulSoup, domain: str) -> str:
    """사이트별 선택자로 본문 추출, 없으면 전체 텍스트 추출"""
    # 공통 제거 태그
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "iframe", "noscript", "svg"]):
        tag.decompose()

    # 사이트별 선택자 시도
    selectors = SITE_SELECTORS.get(domain, []) + SITE_SELECTORS["default"]
    for selector in selectors:
        element = soup.select_one(selector)
        if element:
            text = element.get_text(separator="\n")
            if len(text.strip()) > 100:  # 너무 짧으면 다음 선택자 시도
                return text.strip()

    # 선택자 실패 → 전체 텍스트
    return soup.get_text(separator="\n").strip()


def _clean_text(text: str) -> str:
    """추출된 텍스트 정리"""
    text = re.sub(r"\n{3,}", "\n\n", text)   # 연속 빈줄 제거
    text = re.sub(r" {2,}", " ", text)         # 연속 공백 제거
    text = re.sub(r"\t+", " ", text)           # 탭 제거
    # 너무 짧은 줄 제거 (1~2글자짜리 잡음 제거)
    lines = [line.strip() for line in text.split("\n") if len(line.strip()) > 2]
    return "\n".join(lines)


async def extract_from_url(url: str) -> str:
    """URL에서 광고 본문 텍스트 추출"""

    # 차단 사이트 체크
    blocked_msg = _check_blocked(url)
    if blocked_msg:
        raise ValueError(blocked_msg)

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
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Cache-Control": "max-age=0",
    }

    try:
        async with httpx.AsyncClient(
            timeout=20.0,
            headers=headers,
            follow_redirects=True,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403:
            raise ValueError(
                "해당 사이트는 크롤링이 차단되어 있습니다. "
                "광고 문구를 직접 복사해서 텍스트 분석을 이용해주세요."
            )
        elif e.response.status_code == 404:
            raise ValueError("페이지를 찾을 수 없습니다. URL을 다시 확인해주세요.")
        else:
            raise ValueError(f"페이지 접근 실패 ({e.response.status_code}). URL을 다시 확인해주세요.")
    except httpx.TimeoutException:
        raise ValueError("페이지 로딩 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.")
    except Exception as e:
        raise ValueError(f"URL에서 텍스트를 가져오지 못했습니다: {str(e)}")

    domain = _get_domain(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = _extract_main_text(soup, domain)
    text = _clean_text(text)

    if not text or len(text) < 20:
        raise ValueError(
            "페이지에서 분석할 텍스트를 찾지 못했습니다. "
            "광고 문구를 직접 복사해서 텍스트 분석을 이용해주세요."
        )

    return text


def extract_from_image(image_bytes: bytes) -> str:
    """이미지에서 OCR로 텍스트 추출 (한국어 + 영어)"""
    image = Image.open(io.BytesIO(image_bytes))
    text = pytesseract.image_to_string(image, lang="kor+eng")
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """텍스트를 문장 단위로 분리 (kss 우선, 없으면 정규식 폴백)"""
    if _KSS_AVAILABLE:
        return _split_with_kss(text)
    return _split_with_regex(text)


def _split_with_kss(text: str) -> list[str]:
    """kss 라이브러리를 사용한 한국어 문장 분리"""
    try:
        # 줄바꿈 기준으로 먼저 단락 분리 후 각 단락을 kss로 분리
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        sentences = []
        for para in paragraphs:
            split = kss.split_sentences(para, backend="punct")
            sentences.extend(split)
        # 너무 짧은 조각 제거
        return [s.strip() for s in sentences if len(s.strip()) > 5]
    except Exception:
        # kss 오류 시 정규식 폴백
        return _split_with_regex(text)


def _split_with_regex(text: str) -> list[str]:
    """정규식 기반 문장 분리 (폴백용)"""
    sentences = re.split(r"(?<=[.!?])\s+|[\n]+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 5]
