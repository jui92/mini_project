# -*- coding: utf-8 -*-
import os, re, json, textwrap, time, urllib.parse, html
from typing import Optional, Tuple, Dict, List

import requests
from bs4 import BeautifulSoup
import html2text
import streamlit as st

# ==============================
# Page config
# ==============================
st.set_page_config(page_title="지원 회사 특화 취업 준비 코치 (v1 - 원문 확보 집중)", page_icon="🧭", layout="wide")

# ==============================
# Helpers: secrets/env
# ==============================
def get_secret(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    if v:
        return v
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default

NAVER_CLIENT_ID = get_secret("NAVER_CLIENT_ID")
NAVER_CLIENT_SECRET = get_secret("NAVER_CLIENT_SECRET")

# ==============================
# Core 1: Robust text fetchers
# - Jina Reader (SPA/더보기 우선)
# - WebBaseLoader 유사(정적 HTML + html2text)
# - BS4 fallback
# ==============================
def normalize_url(u: str) -> Optional[str]:
    if not u:
        return None
    u = u.strip()
    if not re.match(r"^https?://", u):
        u = "https://" + u
    # remove fragments
    u = urllib.parse.urlsplit(u)
    u = urllib.parse.urlunsplit((u.scheme, u.netloc, u.path, u.query, ""))  # drop fragment
    return u

def http_get(url: str, timeout: int = 12) -> Optional[requests.Response]:
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept-Language": "ko, en;q=0.9",
        }
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code == 200 and "text/html" in r.headers.get("content-type", ""):
            return r
    except Exception:
        pass
    return None

def fetch_jina(url: str, timeout: int = 15) -> str:
    """
    Jina Reader: 동적 렌더링/더보기 텍스트까지 프리렌더한 결과를 text로 반환.
    """
    try:
        # jina reader: https://r.jina.ai/http://example.com
        # (https도 http 접두 사용 권장. 원본이 https여도 상관 없음)
        prox = f"https://r.jina.ai/http://{urllib.parse.urlsplit(url).netloc}{urllib.parse.urlsplit(url).path}"
        if urllib.parse.urlsplit(url).query:
            prox += f"?{urllib.parse.urlsplit(url).query}"
        r = http_get(prox, timeout=timeout)
        if r and r.text:
            return r.text.strip()
    except Exception:
        pass
    return ""

def html_to_text(html_str: str) -> str:
    try:
        conv = html2text.HTML2Text()
        conv.ignore_links = True
        conv.ignore_images = True
        conv.body_width = 0
        txt = conv.handle(html_str)
        txt = re.sub(r"\n{3,}", "\n\n", txt)
        return txt.strip()
    except Exception:
        return ""

def fetch_static(url: str, timeout: int = 12) -> str:
    """
    WebBaseLoader 유사: 정적 HTML을 받아 html2text로 변환
    """
    r = http_get(url, timeout=timeout)
    if not r:
        return ""
    return html_to_text(r.text)

def fetch_bs4_blocks(url: str, timeout: int = 12) -> str:
    r = http_get(url, timeout=timeout)
    if not r:
        return ""
    soup = BeautifulSoup(r.text, "lxml")

    # 긴 본문 위주로 합치기
    blocks = []
    for sel in ["article", "section", "main", "div", "ul", "ol"]:
        for el in soup.select(sel):
            txt = el.get_text(" ", strip=True)
            if txt and len(txt) > 300:
                blocks.append(txt)
    if not blocks:
        # 전체에서라도
        txt = soup.get_text(" ", strip=True)
        return txt[:150000]
    # 중복 제거 + 합치기
    seen = set(); out=[]
    for b in blocks:
        b = re.sub(r"\s+", " ", b)
        if b not in seen:
            seen.add(b); out.append(b)
    return "\n\n".join(out)[:150000]

def fetch_job_text_all(url: str) -> Tuple[str, Dict]:
    """
    하나의 URL에서 3단계로 텍스트를 최대한 확보.
    반환: (결과 텍스트, 디버그메타)
    """
    url = normalize_url(url)
    if not url:
        return "", {"error": "invalid_url"}

    jina_txt = fetch_jina(url)
    webbase_txt = fetch_static(url) if not jina_txt else ""
    bs4_txt = fetch_bs4_blocks(url) if (not jina_txt and not webbase_txt) else ""

    # 특수: 원티드 JSON-LD 보강
    enrich = ""
    try:
        if "wanted.co.kr" in url:
            r = http_get(url)
            if r:
                soup = BeautifulSoup(r.text, "lxml")
                for s in soup.find_all("script", {"type": "application/ld+json"}):
                    try:
                        data = json.loads(s.string or "{}")
                        if isinstance(data, dict) and data.get("@type") == "JobPosting":
                            desc = data.get("description") or ""
                            desc = BeautifulSoup(desc, "lxml").get_text(" ", strip=True)
                            if len(desc) > 200:
                                enrich = desc
                                break
                        elif isinstance(data, list):
                            for obj in data:
                                if isinstance(obj, dict) and obj.get("@type") == "JobPosting":
                                    desc = BeautifulSoup(obj.get("description",""), "lxml").get_text(" ", strip=True)
                                    if len(desc) > 200:
                                        enrich = desc
                                        break
                    except Exception:
                        continue
    except Exception:
        pass

    # 조합 (우선순위: Jina > WebBase > BS4 > Enrich)
    base = jina_txt or webbase_txt or bs4_txt or ""
    if enrich and enrich not in base:
        base = base + "\n\n" + enrich

    lens = {
        "jina": len(jina_txt),
        "webbase": len(webbase_txt),
        "bs4": len(bs4_txt),
        "enrich": len(enrich),
    }
    return base.strip(), {"url_final": url, "lens": lens}

# ==============================
# Core 2: 채용 URL 탐색
# - Naver OpenAPI (키 있으면)
# - DuckDuckGo HTML 폴백
# - 목록 → 상세화
# ==============================
JOB_SITES = [
    "wanted.co.kr", "saramin.co.kr", "jobkorea.co.kr",
    "rocketpunch.com", "indeed.com", "linkedin.com"
]

def naver_search_web(query: str, display: int = 10) -> List[str]:
    if not (NAVER_CLIENT_ID and NAVER_CLIENT_SECRET):
        return []
    try:
        url = "https://openapi.naver.com/v1/search/webkr.json"
        headers = {
            "X-Naver-Client-Id": NAVER_CLIENT_ID,
            "X-Naver-Client-Secret": NAVER_CLIENT_SECRET,
        }
        params = {"query": query, "display": display, "sort": "date"}
        r = requests.get(url, headers=headers, params=params, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json()
        out = []
        for it in data.get("items", []):
            link = it.get("link")
            if link: out.append(link)
        return out
    except Exception:
        return []

def ddg_search(query: str, limit: int = 10) -> List[str]:
    try:
        url = f"https://duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        r = http_get(url, timeout=10)
        if not r: return []
        soup = BeautifulSoup(r.text, "lxml")
        out=[]
        for a in soup.select("a.result__a, a.result__url, a[href]"):
            href = a.get("href")
            if not href: continue
            # duckduckgo redirect decode
            if href.startswith("/l/?kh=") and "uddg=" in href:
                href = urllib.parse.unquote(href.split("uddg=")[-1])
            if re.match(r"^https?://", href):
                out.append(href)
            if len(out) >= limit: break
        return out
    except Exception:
        return []

def first_detail_from_list(url: str) -> str:
    """
    목록 URL이면 첫 상세 공고 링크로 한 번 더 진입
    """
    try:
        r = http_get(url, timeout=8)
        if not r: return url
        soup = BeautifulSoup(r.text, "lxml")
        # 사이트별 대충 맞는 셀렉터들
        cand = []
        # 원티드
        cand += [a["href"] for a in soup.select("a.JobCard, a[href*='/wd/']") if a.has_attr("href")]
        # 사람인/잡코리아
        cand += [a["href"] for a in soup.select("a[href*='view.asp'], a[href*='/Recruit/']:not([href*='Search'])") if a.has_attr("href")]
        # 로켓펀치/기타
        cand += [a["href"] for a in soup.select("a[href*='/companies/'], a[href*='/jobs/']") if a.has_attr("href")]

        for h in cand:
            if not re.match(r"^https?://", h):
                h = urllib.parse.urljoin(url, h)
            d = urllib.parse.urlsplit(h).netloc
            if any(s in d for s in JOB_SITES):
                return normalize_url(h)
    except Exception:
        pass
    return url

def discover_job_urls(company: str, role: str, limit: int = 8) -> List[str]:
    queries = []
    if role:
        queries.append(f"{company} {role} 채용")
        queries.append(f"{company} {role} 공고")
    queries.append(f"{company} 채용 공고")
    queries.append(f"{company} hiring jobs")

    urls = []
    # 1) NAVER
    for q in queries:
        for lk in naver_search_web(q, display=10):
            d = urllib.parse.urlsplit(lk).netloc
            if any(s in d for s in JOB_SITES) and lk not in urls:
                urls.append(lk)
            if len(urls) >= limit: break
        if len(urls) >= limit: break

    # 2) DuckDuckGo
    if len(urls) < 2:
        for q in queries:
            for lk in ddg_search(q, limit=12):
                d = urllib.parse.urlsplit(lk).netloc
                if any(s in d for s in JOB_SITES) and lk not in urls:
                    urls.append(lk)
                if len(urls) >= limit: break
            if len(urls) >= limit: break

    # 목록 → 상세화
    detail = []
    for u in urls[:limit]:
        detail.append(first_detail_from_list(u))
    # 중복 제거
    seen=set(); out=[]
    for u in detail:
        if u not in seen:
            seen.add(u); out.append(u)
    return out[:limit]

def pick_best_fetchable(urls: List[str], min_len: int = 800) -> Tuple[Optional[str], Dict]:
    """
    여러 URL 중 텍스트를 실제로 길게 뽑을 수 있는 '가장 좋은' 후보 1개 선택
    """
    best = None
    best_meta = {}
    best_score = -1
    for u in urls:
        txt, meta = fetch_job_text_all(u)
        score = len(txt)
        if score > best_score:
            best = u; best_meta = meta; best_score = score
        if score >= min_len:
            break
    return best, {"tried": urls, "chosen_meta": best_meta}

# ==============================
# UI — 사이드바: 설정/도움
# ==============================
with st.sidebar:
    st.title("⚙️ 설정/도움")
    st.markdown("- NAVER 키가 있으면 검색 품질↑ (선택)")
    st.json({
        "NAVER_CLIENT_ID": bool(NAVER_CLIENT_ID),
        "NAVER_CLIENT_SECRET": bool(NAVER_CLIENT_SECRET),
    })
    st.caption("원문 확보가 우선 과제입니다. 이후 자소서/RAG/질문/채점은 이 원문을 근거로 연결하면 됩니다.")

st.title("지원 회사 특화 취업 준비 코치 · 원문 확보 집중판")

# ==============================
# 단위 A) 원문 테스트(직접 URL)
# ==============================
st.header("단위 A) 원문 테스트 (직접 URL)")
test_url = st.text_input("테스트할 채용 상세 URL을 입력하세요", placeholder="https://www.wanted.co.kr/wd/123456")
if st.button("📄 텍스트 실행", type="secondary"):
    if not test_url.strip():
        st.warning("URL을 입력하세요.")
    else:
        with st.spinner("원문 수집 중..."):
            txt, meta = fetch_job_text_all(test_url)
        st.json({"url_final": meta.get("url_final"), "lens": meta.get("lens")})
        st.write(f"텍스트 길이: {len(txt)}")
        st.text_area("미리보기(앞 3000자)", value=txt[:3000], height=220)
        st.download_button("원문 다운로드", data=txt.encode("utf-8"), file_name="raw_job_text.txt", mime="text/plain")

st.divider()

# ==============================
# 단위 B) 회사명 + 직무로 채용 URL 자동 탐색
# ==============================
st.header("단위 B) 회사명+직무 → 채용 URL 자동탐색")
col = st.columns(3)
with col[0]:
    company = st.text_input("회사 이름", placeholder="예: 화해글로벌 / NAVER / 카카오 등")
with col[1]:
    role = st.text_input("지원 직무명", placeholder="예: Data Engineer / Data Analyst ...")
with col[2]:
    min_len = st.number_input("최소 길이(문자)", value=800, min_value=0, step=100)

auto_state = st.empty()
if st.button("🔎 채용 URL 찾기", type="primary"):
    if not company.strip():
        st.warning("회사 이름을 입력하세요.")
    else:
        with st.spinner("검색 → 후보URL 모으는 중..."):
            urls = discover_job_urls(company.strip(), role.strip(), limit=8)
        st.write("후보 URL:", urls if urls else "(없음)")

        chosen, choose_meta = (None, {})
        if urls:
            with st.spinner("후보 URL 실제로 텍스트 뽑히는지 검사중..."):
                chosen, choose_meta = pick_best_fetchable(urls, min_len=min_len)

        if chosen:
            st.success("상세 URL 선택 완료")
            st.code(chosen, language="text")
            st.json(choose_meta)
            st.session_state["chosen_job_url"] = chosen
        else:
            st.error("텍스트를 충분히 뽑을 수 있는 URL을 찾지 못했습니다. (로그인/차단/SPA 가능)")

st.divider()

# ==============================
# 단위 C) 회사 요약 / 채용 요건 — '원문 그대로' 출력
# ==============================
st.header("단위 C) 회사 요약 / 채용 요건 (원문 그대로 출력)")

col2 = st.columns(2)
with col2[0]:
    st.subheader("회사 요약 (원문 전체)")
    raw_company = ""
    # 회사 소개는 공식 홈페이지 or 뉴스가 필요하지만
    # 여기선 '채용 공고 원문'을 우선 표준 소스로 사용 (후속 단계에서 별도 보강)
    # 사용자가 직접 회사 홈페이지 URL을 넣으면 그 원문도 함께 보여줌
    home_url = st.text_input("(선택) 회사 홈페이지 또는 회사 소개가 있는 URL")
    if st.button("🏠 회사 소개 URL 원문 불러오기", key="btn_home"):
        if not home_url.strip():
            st.warning("회사 소개 URL을 입력하세요.")
        else:
            with st.spinner("회사 소개 원문 수집..."):
                raw_company, meta = fetch_job_text_all(home_url.strip())
            st.json(meta)
            st.write(f"회사소개 텍스트 길이: {len(raw_company)}")
            st.text_area("회사 소개 원문", value=raw_company, height=360)
            if raw_company:
                st.download_button("회사 소개 원문 다운로드", data=raw_company.encode("utf-8"),
                                   file_name="raw_company_text.txt", mime="text/plain")

with col2[1]:
    st.subheader("채용 요건 (원문 전체)")
    job_url = st.text_input("(선택) 채용 공고 상세 URL", value=st.session_state.get("chosen_job_url",""))
    if st.button("🧾 채용 공고 원문 불러오기", key="btn_job"):
        if not job_url.strip():
            st.warning("채용 공고 URL을 입력하거나, 위에서 자동탐색 먼저 실행하세요.")
        else:
            with st.spinner("채용 공고 원문 수집..."):
                raw_job, meta = fetch_job_text_all(job_url.strip())
            st.json(meta)
            st.write(f"채용요건 텍스트 길이: {len(raw_job)}")
            st.text_area("채용 요건 원문", value=raw_job, height=360)
            if raw_job:
                st.download_button("채용 요건 원문 다운로드", data=raw_job.encode("utf-8"),
                                   file_name="raw_job_text.txt", mime="text/plain")

st.info("※ 지금 단계는 '원문 확보'에 집중합니다. 후속으로 이 원문을 RAG에 넣어 자소서/질문/채점으로 확장하면 됩니다.")
