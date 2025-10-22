# Fact_check_bot.py  (웹문서 스크랩/정규화 지원)
# - URL 입력을 자동 분리/디코딩/중복제거
# - 각 URL에서 본문을 스크랩해 요약 텍스트를 컨텍스트로 주입
# - secrets.toml 없어도 안전 (환경변수/사이드바/실패시 경고)
# - OpenAI SDK v1 이상/이하 호환

from io import BytesIO
from pypdf import PdfReader
import json, os, re, time
from typing import Any, Dict, Optional, List, Tuple
from urllib.parse import urlparse, unquote, urlunparse

import requests
from bs4 import BeautifulSoup
from readability import Document
import streamlit as st

# (선택) .env 지원
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass


# ==========================
# secrets 안전 접근
# ==========================
def safe_secret(key: str, default: Optional[str] = None) -> Optional[str]:
    try:
        sec = getattr(st, "secrets", None)
        if sec is None:
            return default
        if hasattr(sec, "get"):
            return sec.get(key, default)  # type: ignore[attr-defined]
        return sec[key] if key in sec else default  # type: ignore[index]
    except Exception:
        return default


def get_openai_api_key(side_override: Optional[str] = None) -> Optional[str]:
    if side_override and side_override.strip():
        return side_override.strip()
    return os.getenv("OPENAI_API_KEY") or safe_secret("OPENAI_API_KEY")


# ==========================
# OpenAI 호환 래퍼
# ==========================
class OpenAICompat:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.mode: Optional[str] = None
        self.client = None
        try:
            from openai import OpenAI  # type: ignore
            self.client = OpenAI(api_key=api_key)
            self.mode = "new"
        except Exception:
            try:
                import openai  # type: ignore
                openai.api_key = api_key
                self.client = openai
                self.mode = "legacy"
            except Exception as e:
                raise RuntimeError("OpenAI SDK 초기화 실패") from e

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: Optional[int] = None,
    ) -> str:
        if self.mode == "new":
            resp = self.client.chat.completions.create(  # type: ignore[attr-defined]
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            return resp.choices[0].message.content or ""
        else:
            resp = self.client.ChatCompletion.create(  # type: ignore[attr-defined]
                model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
            )
            return resp["choices"][0]["message"]["content"]  # type: ignore[index]


# ==========================
# PDF에서(title, text) 추출
# ==========================
def is_pdf_url(url: str) -> bool:
    """URL이 .pdf로 끝나거나 HEAD 요청의 Content-Type이 application/pdf면 PDF로 간주"""
    try:
        p = urlparse(url)
        if p.path.lower().endswith(".pdf"):
            return True
        # 네트워크 비용을 줄이고 싶으면 이 HEAD 요청은 생략해도 됩니다.
        h = requests.head(url, headers=HEADERS, allow_redirects=True, timeout=8)
        ctype = h.headers.get("Content-Type", "")
        return "application/pdf" in ctype.lower()
    except Exception:
        return False


def fetch_pdf_text(url: str, timeout: int = 20, max_pages: int = 8) -> tuple[str, str]:
    """
    PDF에서 (title, text) 추출.
    - title: URL 파일명 기준
    - text : 앞쪽 max_pages 페이지만 추출 (길이·토큰 폭주 방지)
    """
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        data = BytesIO(r.content)
        reader = PdfReader(data)
    except Exception:
        return "", ""

    # 제목은 파일명으로 대체
    title = os.path.basename(urlparse(url).path) or "document.pdf"

    texts = []
    n = min(len(reader.pages), max_pages)
    for i in range(n):
        try:
            pg = reader.pages[i]
            txt = pg.extract_text() or ""
            # 줄바꿈 정리
            txt = re.sub(r"\s+\n", "\n", txt)
            txt = re.sub(r"\n{3,}", "\n\n", txt)
            texts.append(txt.strip())
        except Exception:
            continue

    body = "\n\n".join([t for t in texts if t])
    # 너무 길면 앞부분만
    if len(body) > 9000:
        body = body[:9000] + "\n…(truncated)…"
    return title, body


# ==========================
# URL 전처리 & 스크랩
# ==========================
_VALID_SCHEMES = {"http", "https"}

def normalize_urls_text(urls_text: str) -> List[str]:
    """줄바꿈/공백/쉼표 기준 분리 → 디코딩 → 쿼리 트래킹 제거(가벼움) → 중복 제거"""
    if not urls_text.strip():
        return []
    cand = re.split(r"[\n\r,\s]+", urls_text.strip())
    cleaned: List[str] = []
    seen = set()
    for u in cand:
        if not u:
            continue
        u = unquote(u.strip())
        try:
            p = urlparse(u)
            if p.scheme.lower() not in _VALID_SCHEMES or not p.netloc:
                continue
            # 너무 긴 tracking query는 잘라줌 (선택)
            q = p.query
            if len(q) > 512:
                q = ""
            norm = urlunparse((p.scheme, p.netloc, p.path, p.params, q, ""))  # fragment 제거
            if norm not in seen:
                seen.add(norm)
                cleaned.append(norm)
        except Exception:
            continue
    return cleaned


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
}

def fetch_and_extract(url: str, timeout: int = 12, max_pdf_pages: int = 8) -> Tuple[str, str]:
    """
    URL에서 (title, text) 추출.
    - PDF면 pypdf로, 그 외는 readability/BeautifulSoup로 처리
    """
    # (A) PDF 분기
    if is_pdf_url(url):
        return fetch_pdf_text(url, timeout=max(timeout, 20), max_pages=max_pdf_pages)

    # (B) HTML 처리 (기존 코드 그대로 유지)
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        html = r.text
    except Exception:
        return "", ""

    # title
    try:
        title = Document(html).short_title()
    except Exception:
        try:
            soup = BeautifulSoup(html, "lxml")
            title = (soup.title.string or "").strip() if soup.title else ""
        except Exception:
            title = ""

    # body
    text = ""
    try:
        summary_html = Document(html).summary()
        soup = BeautifulSoup(summary_html, "lxml")
        text = soup.get_text(separator="\n", strip=True)
    except Exception:
        try:
            soup = BeautifulSoup(html, "lxml")
            article = soup.find("article")
            if article:
                text = article.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)
        except Exception:
            text = ""

    if len(text) > 6000:
        text = text[:6000] + "\n…(truncated)…"
    return title, text


# ==========================
# Streamlit UI
# ==========================
st.set_page_config(page_title="Fact Check Bot", page_icon="🕵️", layout="wide")
st.title("🕵️ Fact Check Bot (웹 스크랩 지원)")

with st.sidebar:
    st.subheader("⚙️ 설정")
    side_api_key = st.text_input("🔑 OPENAI_API_KEY (선택 입력: 우선 적용)", type="password")
    model = st.selectbox("모델 선택", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"], index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.slider("Max tokens (응답 상한)", 256, 4096, 1024, 64)
    max_pdf_pages = st.slider("PDF에서 읽을 최대 페이지 수", 1, 20, 8, 1)
    st.caption("키 우선순위: 사이드바 > 환경변수 > secrets.toml")

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    claim = st.text_area("검증할 주장(Claim)", placeholder="예: '하루에 와인 한 잔은 건강에 도움 된다.'", height=140)
    context = st.text_area("추가 컨텍스트(선택)", placeholder="인용문/배경 텍스트 등", height=120)
    urls_text = st.text_area("참고 URL들(선택, 줄바꿈/쉼표로 구분)", height=110)

with col2:
    st.markdown("**출력 포맷**")
    st.code(
        """{
  "verdict": "SUPPORTED | REFUTED | INSUFFICIENT",
  "confidence": 0.0~1.0,
  "evidence": ["핵심 근거 1", "핵심 근거 2"],
  "notes": "설명",
  "citations": [{"url": "...", "title": "..."}]
}""",
        "json",
    )
    run = st.button("✅ 사실검증 실행", type="primary", use_container_width=True)

# ==========================
# 실행
# ==========================
if run:
    api_key = get_openai_api_key(side_override=side_api_key)
    if not api_key:
        st.error("OPENAI_API_KEY가 필요합니다. (사이드바/환경변수/secrets.toml)")
        st.stop()
    if not claim.strip():
        st.warning("검증할 주장을 입력하세요.")
        st.stop()

    try:
        client = OpenAICompat(api_key)
    except Exception as e:
        st.error(f"OpenAI 초기화 실패: {e}")
        st.stop()

    # 1) URL 정규화
    urls = normalize_urls_text(urls_text)
    fetched_items: List[Dict[str, str]] = []
    if urls:
        st.info(f"URL {len(urls)}개에서 본문 스크랩 중…")
        progress = st.progress(0.0)
        for i, u in enumerate(urls, 1):
            title, text = fetch_and_extract(u)
            if text.strip():
                fetched_items.append({"url": u, "title": title or u, "text": text})
            progress.progress(i / len(urls))
            time.sleep(0.05)

    # 2) 모델 프롬프트
    sys_prompt = (
        "당신은 근거 기반 사실검증 어시스턴트입니다. 사용자가 제공한 주장과(선택) 컨텍스트·웹 페이지 텍스트를 바탕으로 "
        "판정이 SUPPORTED(지지됨), REFUTED(반박됨), INSUFFICIENT(근거 불충분) 중 무엇인지 판단하세요. "
        "제공된 텍스트만 1차 근거로 사용하며, 근거가 부족하거나 상충되면 INSUFFICIENT를 선택하세요. "
        "반드시 **JSON만** 반환합니다. 키는 verdict(SUPPORTED|REFUTED|INSUFFICIENT), "
        "confidence(0..1), evidence(간결한 불릿 리스트), notes(문장형 설명), "
        "citations({url,title} 배열)입니다. "
        "모든 자연어 콘텐츠(evidence, notes, citations.title)는 **한국어로** 작성하세요. "
        "단, verdict 값 자체는 영문 코드(SUPPORTED/REFUTED/INSUFFICIENT)를 유지하세요."
)

    user_payload: Dict[str, Any] = {"claim": claim.strip()}
    if context.strip():
        user_payload["extra_context"] = context.strip()

    if fetched_items:
        # 너무 길면 요약을 요구하는 추가 지시 포함
        user_payload["sources"] = [
            {"url": it["url"], "title": it["title"], "text": it["text"][:4000]}
            for it in fetched_items
        ]

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]

    with st.spinner("사실 검증 중…"):
        try:
            raw = client.chat(messages, model=model, temperature=temperature, max_tokens=max_tokens)
        except Exception as e:
            st.error(f"모델 호출 실패: {e}")
            st.stop()

    # 3) JSON 파싱
    parsed: Dict[str, Any] = {}
    try:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].lstrip()
        parsed = json.loads(cleaned)
    except Exception:
        st.warning("응답 JSON 파싱 실패. 원문 표시합니다.")
        st.code(raw, "json")
        st.stop()

    # 4) 렌더링
    st.subheader("🧾 결과")
    verdict = str(parsed.get("verdict", "")).upper()
    confidence = float(parsed.get("confidence", 0.0))
    evidence_list = parsed.get("evidence", [])
    notes = parsed.get("notes", "")
    citations = parsed.get("citations", [])

    icon = {"SUPPORTED": "✅", "REFUTED": "❌", "INSUFFICIENT": "⚠️"}.get(verdict, "❓")
    st.markdown(f"### {icon} Verdict: **{verdict or 'UNKNOWN'}**")
    st.progress(min(max(confidence, 0.0), 1.0))
    st.caption(f"Confidence: {confidence:.2f}")

    if evidence_list:
        st.markdown("**Evidence**")
        for ev in evidence_list:
            st.markdown(f"- {ev}")

    if notes:
        st.markdown("**Notes**")
        st.write(notes)

    if citations:
        st.markdown("**Citations**")
        for c in citations:
            url = c.get("url", "")
            title = c.get("title", url)
            st.markdown(f"- [{title}]({url})")

    if fetched_items:
        with st.expander("🔎 스크랩된 원문 요약(모델 입력에 사용됨)"):
            for it in fetched_items:
                st.markdown(f"**{it['title']}**  \n{it['url']}")
                st.text(it["text"][:1200] + ("…" if len(it["text"]) > 1200 else ""))


with st.expander("ℹ️ 도움말"):
    st.markdown(
        """
**입력 팁**
- URL은 검색결과 링크 말고 **기사/보고서 원문 URL**을 넣어주세요.
- 퍼센트 인코딩된 주소도 자동 복원합니다. 여러 개는 줄바꿈/쉼표로 구분.

**설치**
```bash
pip install readability-lxml beautifulsoup4 lxml
        """)