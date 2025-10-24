# -*- coding: utf-8 -*-
# interview_coach_bot.py
# v3.1 — 채용 포털 상세 페이지의 "모든 텍스트"를 ② 회사 요약 / 채용 요건에 그대로 출력
#        (WebBaseLoader 우선, 실패시 BeautifulSoup get_text() 폴백)
#        나머지 기능(질문 생성, 채점/레이다/CSV)은 그대로 유지

import os, io, re, json, textwrap, urllib.parse, difflib, random, time
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import streamlit as st

# ---------- Optional deps ----------
try:
    import pypdf
except Exception:
    pypdf = None

try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# --- WebBaseLoader (원문 전체 로드용)
try:
    from langchain_community.document_loaders import WebBaseLoader
    WEBBASE_OK = True
except Exception:
    WEBBASE_OK = False

import requests
from bs4 import BeautifulSoup

# ---------- Page config ----------
st.set_page_config(page_title="회사 특화 가상 면접 코치", page_icon="🎯", layout="wide")

# =========================================================
# Secrets / API keys
# =========================================================
def _secrets_file_exists() -> bool:
    candidates = [
        os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml"),
        os.path.join(os.getcwd(), ".streamlit", "secrets.toml"),
    ]
    return any(os.path.exists(p) for p in candidates)

def load_api_key_from_env_or_secrets() -> Optional[str]:
    key = os.getenv("OPENAI_API_KEY")
    if key: return key
    try:
        if _secrets_file_exists() or hasattr(st, "secrets"):
            return st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        pass
    return None

def load_naver_keys():
    cid = os.getenv("NAVER_CLIENT_ID")
    csec = os.getenv("NAVER_CLIENT_SECRET")
    try:
        if hasattr(st, "secrets"):
            cid = cid or st.secrets.get("NAVER_CLIENT_ID", None)
            csec = csec or st.secrets.get("NAVER_CLIENT_SECRET", None)
    except Exception:
        pass
    return cid, csec

NAVER_ID, NAVER_SECRET = load_naver_keys()
UA = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}

# =========================================================
# Utils
# =========================================================
def _clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()

def _snippetize(text: str, maxlen: int = 240) -> str:
    t = _clean_text(text)
    return t if len(t) <= maxlen else t[: maxlen - 1] + "…"

def _domain(u: str|None) -> str|None:
    if not u: return None
    try:
        if not u.startswith("http"): u = "https://" + u
        return urllib.parse.urlparse(u).netloc.lower().replace("www.","")
    except Exception:
        return None

def read_file_to_text(uploaded) -> str:
    name = uploaded.name.lower()
    data = uploaded.read()
    if name.endswith((".txt", ".md")):
        for enc in ("utf-8", "cp949", "euc-kr"):
            try: return data.decode(enc)
            except Exception: continue
        return data.decode("utf-8", errors="ignore")
    elif name.endswith(".pdf"):
        if pypdf is None:
            st.warning("pypdf가 필요합니다. requirements.txt에 pypdf 추가.")
            return ""
        try:
            reader = pypdf.PdfReader(io.BytesIO(data))
            return "\n\n".join([(reader.pages[i].extract_text() or "") for i in range(len(reader.pages))])
        except Exception as e:
            st.warning(f"PDF 파싱 실패({uploaded.name}): {e}")
            return ""
    return ""

def chunk_text(text: str, size: int = 900, overlap: int = 150):
    text = re.sub(r"\s+", " ", text).strip()
    if not text: return []
    out, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        out.append(text[start:end])
        if end == len(text): break
        start = max(0, end - overlap)
    return out

# =========================================================
# NAVER Open API (뉴스/웹검색)
# =========================================================
def _naver_api_get(api: str, params: dict, cid: str, csec: str):
    url = f"https://openapi.naver.com/v1/search/{api}.json"
    headers = {"X-Naver-Client-Id": cid, "X-Naver-Client-Secret": csec, **UA}
    r = requests.get(url, headers=headers, params=params, timeout=8)
    if r.status_code != 200:
        return None
    return r.json()

@st.cache_data(show_spinner=False, ttl=1800)
def naver_search_news(query: str, display: int = 8, sort: str = "date") -> list[dict]:
    cid, csec = load_naver_keys()
    if not (cid and csec): return []
    js = _naver_api_get("news", {"query": query, "display": display, "sort": sort}, cid, csec)
    if not js: return []
    out = []
    for it in js.get("items", []):
        title = _clean_text(re.sub(r"</?b>|&quot;|&apos;|&amp;|&lt;|&gt;", "", it.get("title","")))
        out.append({"title": title, "link": it.get("link"), "pubDate": it.get("pubDate")})
    return out

@st.cache_data(show_spinner=False, ttl=3600)
def naver_search_web(query: str, display: int = 10, sort: str = "date") -> list[str]:
    cid, csec = load_naver_keys()
    if not (cid and csec): return []
    js = _naver_api_get("webkr", {"query": query, "display": display, "sort": sort}, cid, csec)
    if not js: return []
    links = []
    for it in js.get("items", []):
        link = it.get("link")
        if link and link not in links:
            links.append(link)
    return links

@st.cache_data(ttl=3600)
def fetch_news(company_name: str, max_items: int = 6) -> list[dict]:
    news = naver_search_news(company_name, display=max_items, sort="date")
    if news:
        return news
    # Google RSS fallback
    q = urllib.parse.quote(company_name)
    url = f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"
    items = []
    try:
        r = requests.get(url, timeout=8, headers=UA)
        if r.status_code != 200: return []
        soup = BeautifulSoup(r.text, "xml")
        for it in soup.find_all("item")[:max_items]:
            title = _clean_text(it.title.get_text()) if it.title else ""
            link  = it.link.get_text() if it.link else ""
            pub   = it.pubDate.get_text() if it.pubDate else ""
            items.append({"title": title, "link": link, "pubDate": pub})
    except Exception:
        return []
    return items

# =========================================================
# 채용 링크 탐색 + 상세 공고
# =========================================================
CAREER_HINTS = ["careers","recruit","jobs","career","채용","인재영입","recruitment","join"]
SEARCH_ENGINES = ["https://duckduckgo.com/html/?q={query}"]
JOB_SITES = ["wanted.co.kr","saramin.co.kr","jobkorea.co.kr","rocketpunch.com","linkedin.com","indeed.com"]

def _first_detail_from_list(url: str, role_hint: str="") -> Optional[str]:
    try:
        r = requests.get(url, timeout=8, headers=UA)
        if r.status_code != 200 or "text/html" not in r.headers.get("content-type",""): return None
        soup = BeautifulSoup(r.text, "html.parser")
        dom = _domain(url) or ""
        if "wanted.co.kr" in dom:
            for a in soup.select("a[href*='/wd/']"):
                href = urllib.parse.urljoin(url, a.get("href"))
                title = (a.get_text() or "").strip()
                if role_hint and (role_hint not in title):
                    continue
                return href
        if "saramin.co.kr" in dom:
            for a in soup.select("a[href*='view?idx=']"):
                return urllib.parse.urljoin(url, a.get("href"))
        if "jobkorea.co.kr" in dom:
            for a in soup.select("a[href*='/Recruit/GI_Read/']"):
                return urllib.parse.urljoin(url, a.get("href"))
        # 일반 패턴
        for a in soup.find_all("a", href=True):
            href = urllib.parse.urljoin(url, a.get("href"))
            if re.search(r"/(wd|jobs|job|view|read|detail|posting)/", href, re.I):
                return href
    except Exception:
        return None
    return None

@st.cache_data(show_spinner=False, ttl=3600)
def discover_job_from_homepage(homepage: str, limit: int = 5) -> list[str]:
    if not homepage: return []
    try:
        if not homepage.startswith("http"): homepage = "https://" + homepage
        r = requests.get(homepage, timeout=8, headers=UA)
        if r.status_code != 200 or "text/html" not in r.headers.get("content-type",""):
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        links=[]
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = (a.get_text() or "").lower()
            if any(k in href.lower() or k in text for k in CAREER_HINTS):
                links.append(urllib.parse.urljoin(homepage, href))
        out=[]; seen=set()
        for lk in links:
            d = _domain(lk)
            if lk not in seen and d:
                seen.add(lk); out.append(lk)
                if len(out) >= limit: break
        return out[:limit]
    except Exception:
        return []

@st.cache_data(show_spinner=False, ttl=3600)
def discover_job_posting_urls(company: str, role: str, homepage: str|None, limit: int=5) -> list[str]:
    urls=[]
    if homepage:
        urls += discover_job_from_homepage(homepage, limit=limit)
    resolved=[]
    for u in urls:
        if re.search(r"/(wd|view|read|detail|posting)/", u, re.I):
            resolved.append(u)
        else:
            detail = _first_detail_from_list(u, role_hint=role or "")
            if detail: resolved.append(detail)
    urls = resolved[:]
    if urls: return urls[:limit]

    # 포털 검색
    if NAVER_ID and NAVER_SECRET:
        for dom in JOB_SITES:
            if len(urls)>=limit: break
            q = f"{company} {role} site:{dom}" if role else f"{company} 채용 site:{dom}"
            links = naver_search_web(q, display=7, sort="date")
            for lk in links:
                if _domain(lk) and dom in _domain(lk) and lk not in urls:
                    if not re.search(r"/(wd|view|read|detail|posting)/", lk, re.I):
                        lk2 = _first_detail_from_list(lk, role_hint=role or "")
                        if lk2: lk = lk2
                    urls.append(lk)
                if len(urls)>=limit: break
    return urls[:limit]

# =========================================================
# 원문 전체 텍스트 로더 (핵심)
# =========================================================
# 필요한 import
import requests, re, urllib.parse
from bs4 import BeautifulSoup

# WebBaseLoader (langchain-community) 시도
try:
    from langchain_community.document_loaders import WebBaseLoader
    WEBBASE_OK = True
except Exception:
    WEBBASE_OK = False

# Playwright URL Loader (선택) - 설치 필요
USE_PLAYWRIGHT = False
try:
    from langchain.document_loaders import PlaywrightURLLoader 
    #from langchain.document_loaders import PlaywrightURLLoader
    PLAYWRIGHT_OK = True
except Exception:
    PLAYWRIGHT_OK = False

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124 Safari/537.36",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7"
}

def webbase_dump_all_text(url: str) -> str:
    if not WEBBASE_OK or not url:
        return ""
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        return "\n\n".join([getattr(d, "page_content", "") for d in docs if getattr(d, "page_content", "")])
    except Exception:
        return ""

def bs4_dump_all_text(url: str) -> str:
    try:
        r = requests.get(url, timeout=12, headers=UA)
        if r.status_code != 200 or "text/html" not in r.headers.get("content-type",""):
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script","style","noscript"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        return re.sub(r"\n{3,}", "\n\n", text)
    except Exception:
        return ""

def jina_reader_text(url: str) -> str:
    try:
        # r.jina.ai proxy (프리렌더) — public 서비스, 도메인/리미트 주의
        proxied = "https://r.jina.ai/http/" + url
        r = requests.get(proxied, timeout=18, headers=UA)
        if r.status_code != 200:
            return ""
        txt = r.text.strip()
        # 일부 페이지는 아주 짧은 텍스트만 반환하므로 길이 제한 완화
        return txt if len(txt) > 10 else txt
    except Exception:
        return ""

def playwright_dump_all_text(url: str) -> str:
    # Playwright 환경이 준비되어 있어야 함. Streamlit Cloud는 지원 여부 확인 필요.
    if not PLAYWRIGHT_OK:
        return ""
    try:
        loader = PlaywrightURLLoader(urls=[url], browser_type="chromium", headless=True, wait_until="networkidle")
        docs = loader.load()
        return "\n\n".join([getattr(d, "page_content", "") for d in docs if getattr(d, "page_content", "")])
    except Exception:
        return ""

def get_full_page_text(url: str) -> str:
    # 정규화(간단)
    if not url: return ""
    if url.startswith("//"): url = "https:" + url
    # 1) WebBaseLoader
    txt = webbase_dump_all_text(url)
    if txt and len(txt.strip()) > 50:
        return txt
    # 2) BeautifulSoup
    txt = bs4_dump_all_text(url)
    if txt and len(txt.strip()) > 50:
        return txt
    # 3) Jina 프리렌더
    txt = jina_reader_text(url)
    if txt and len(txt.strip()) > 10:
        return txt
    # 4) Playwright (가장 강력하지만 리소스/설치 필요)
    txt = playwright_dump_all_text(url)
    return txt or ""


# =========================================================
# OpenAI
# =========================================================
with st.sidebar:
    st.title("⚙️ 설정")
    API_KEY = load_api_key_from_env_or_secrets()
    if not API_KEY:
        st.info("환경변수/Secrets에서 키를 못 찾았습니다. 아래에 입력 후 엔터.")
        API_KEY = st.text_input("OPENAI_API_KEY", type="password")
    MODEL = st.selectbox("챗 모델", ["gpt-4o-mini","gpt-4o","gpt-4.1-mini"], index=0)
    EMBED_MODEL = st.selectbox("임베딩 모델", ["text-embedding-3-small","text-embedding-3-large"], index=0)

    with st.expander("디버그"):
        try:
            import openai as _openai_pkg; _openai_ver = getattr(_openai_pkg, "__version__", None)
        except Exception: _openai_ver=None
        try:
            import httpx as _httpx_pkg; _httpx_ver = getattr(_httpx_pkg, "__version__", None)
        except Exception: _httpx_ver=None
        st.write({
            "api_key_provided": bool(API_KEY),
            "naver_keys": bool(NAVER_ID and NAVER_SECRET),
            "openai_version": _openai_ver,
            "httpx_version": _httpx_ver,
            "webbaseloader_available": WEBBASE_OK
        })

if not OpenAI or not API_KEY:
    st.error("OpenAI API Key가 필요합니다.")
    st.stop()
client = OpenAI(api_key=API_KEY, timeout=30.0)

# =========================================================
# ① 회사/직무 입력
# =========================================================
st.subheader("① 회사/직무 입력")
c1, c2 = st.columns(2)
with c1:
    company_name_input = st.text_input("회사 이름", placeholder="예: 네이버 / Kakao / 삼성SDS")
with c2:
    role_title = st.text_input("지원 직무명", placeholder="데이터 애널리스트 / ML 엔지니어 ...")
job_url_input  = st.text_input("채용 공고 URL(선택) — 없다면 자동 탐색")
homepage_input = st.text_input("공식 홈페이지 URL(선택)", placeholder="https://...")

for k,v in [("company", None), ("answer_text",""), ("history",[]), ("current_question","")]:
    if k not in st.session_state: st.session_state[k]=v

def build_company_obj(name: str, homepage: str|None, role: str|None, job_url: str|None) -> dict:
    # 뉴스는 유지(다른 섹션 용)
    news = fetch_news(name, max_items=6)
    # 공고 URL 결정
    if job_url and job_url.strip():
        urls = [job_url.strip()]
    else:
        urls = discover_job_posting_urls(name, role or "", homepage, limit=6)
    chosen_url = urls[0] if urls else None
    # 핵심: 원문 전체 텍스트
    full_text = get_full_page_text(chosen_url) if chosen_url else ""
    return {
        "company_name": name.strip() or "(회사명 미설정)",
        "homepage": homepage or None,
        "role": role or "",
        "job_url": chosen_url,
        "news": news,
        # ↓↓↓ 핵심: 페이지의 모든 텍스트(그대로) 저장
        "job_raw_text": full_text
    }

if st.button("회사/직무 정보 불러오기", type="primary"):
    if not company_name_input.strip():
        st.warning("회사 이름을 입력해 주세요.")
    else:
        st.session_state.history = []
        st.session_state.current_question = ""
        st.session_state.answer_text = ""
        with st.spinner("회사/직무/공고/뉴스 수집 및 원문 로드 중..."):
            st.session_state.company = build_company_obj(company_name_input, homepage_input or None, role_title or None, job_url_input or None)
        st.success("회사 정보 갱신 및 실행결과 초기화 완료!")

company = st.session_state.get("company")

# =========================================================
# ② 회사 요약 / 채용 요건  →  “페이지의 모든 텍스트” 그대로 출력
# =========================================================
st.subheader("② 회사 요약 / 채용 요건 (페이지 원문 전체)")

if company:
    cols = st.columns(2)
    with cols[0]:
        if company.get("job_url"): st.link_button("채용 공고 열기", company["job_url"])
    with cols[1]:
        if company.get("homepage"): st.link_button("홈페이지 열기", company["homepage"])
    if company.get("news"):
        st.markdown("**최근 뉴스:**")
        for n in company["news"][:3]:
            st.markdown(f"- [{_clean_text(n['title'])}]({n['link']})")
    st.markdown("---")

    raw_text = company.get("job_raw_text","").strip()
    if not raw_text:
        st.warning("원문 텍스트를 가져오지 못했습니다. 로그인/봇차단/동적 렌더링 여부를 확인하거나, 공고 URL을 직접 입력해 보세요.")
    # 요청사항: 두 섹션 모두 ‘모든 텍스트’를 그대로 출력
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 회사 요약 (원문 전체)")
        st.text_area("회사 요약 원문", value=raw_text, height=500)
        st.download_button("회사 요약 원문 다운로드", data=raw_text.encode("utf-8"), file_name="company_summary_raw.txt", mime="text/plain")
    with c2:
        st.markdown("#### 채용 요건 (원문 전체)")
        st.text_area("채용 요건 원문", value=raw_text, height=500)
        st.download_button("채용 요건 원문 다운로드", data=raw_text.encode("utf-8"), file_name="job_requirements_raw.txt", mime="text/plain")

else:
    st.info("위 입력을 완료하고 ‘회사/직무 정보 불러오기’를 눌러 표시하세요.")

# =========================================================
# ③ 질문 생성  (기존 유지)
# =========================================================
st.subheader("③ 질문 생성")

TYPE_INSTRUCTIONS = {
    "행동(STAR)": "S(상황)-T(과제)-A(행동)-R(성과) 실무사례 유도",
    "기술 심층": "설계/트레이드오프/성능-비용/품질 지표 심층",
    "핵심가치 적합성": "핵심가치·태도 검증 상황형",
    "역질문": "지원자가 회사를 평가하는 역질문"
}
q_type = st.selectbox("질문 유형", list(TYPE_INSTRUCTIONS.keys()))
level  = st.selectbox("난이도/연차", ["주니어","미들","시니어"])
hint   = st.text_input("질문 생성 힌트(선택)", placeholder="예: 전환 퍼널 / 모델 성능-비용 / 데이터 품질")

def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()
def pick_diverse(cands: list[str], hist: list[str], gamma: float = 0.35) -> str:
    if not cands: return ""
    if not hist:  return random.choice(cands)
    best=None; best_score=1e9
    for q in cands:
        sims=[_similarity(q,h) for h in hist] or [0.0]
        score=(sum(sims)/len(sims)) + gamma*np.std(sims)
        if score < best_score:
            best_score=score; best=q
    return best

if "history" not in st.session_state: st.session_state.history=[]
if "current_question" not in st.session_state: st.session_state.current_question=""

if st.button("새 질문 받기", use_container_width=True, type="primary"):
    st.session_state.answer_text = ""
    try:
        news_titles = ", ".join([_snippetize(n["title"], 70) for n in (company or {}).get("news", [])[:3]])
        ctx = textwrap.dedent(f"""
        [회사명] {(company or {}).get('company_name','')}
        [모집 분야] {(company or {}).get('role','')}
        [최근 이슈/뉴스] {news_titles}
        """).strip()
        seed = int(time.time()*1000) % 2_147_483_647
        sys = f"""너는 '{(company or {}).get('company_name','회사')}'의 면접관이다.
컨텍스트/이슈를 반영하여 **{q_type}** 유형({TYPE_INSTRUCTIONS[q_type]})의 질문 **6개**를 한국어로 생성하라.
서로 형태·관점·키워드가 달라야 하며 난이도는 {level}. 포맷: 1) ... 2) ... 3) ..."""
        user = f"[컨텍스트]\n{ctx}\n\n[힌트]\n{hint}\n[랜덤시드] {seed}"
        resp = client.chat.completions.create(
            model=MODEL, temperature=0.9,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}]
        )
        raw = resp.choices[0].message.content.strip()
        cands = [re.sub(r'^\s*\d+\)\s*','',line).strip() for line in raw.splitlines() if re.match(r'^\s*\d+\)', line)]
        if not cands:
            cands = [l.strip("- ").strip() for l in raw.splitlines() if len(l.strip())>0][:6]
        hist_qs = [h["question"] for h in st.session_state.history][-10:]
        st.session_state.current_question = pick_diverse(cands, hist_qs) or (cands[0] if cands else "질문 생성 실패")
    except Exception as e:
        st.error(f"질문 생성 오류: {e}")

st.text_area("질문", height=110, value=st.session_state.get("current_question",""))

# =========================================================
# ④ 나의 답변 / 채점 & 코칭 (기존 총점=기준 합산 0~100 유지)
# =========================================================
st.subheader("④ 나의 답변 / 코칭")
ans = st.text_area("여기에 답변을 작성하세요 (STAR 권장: 상황-과제-행동-성과)", height=200, key="answer_text")

CRITERIA = ["문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치"]

def evaluate_answer(question: str, answer: str) -> dict:
    schema = {
      "type":"object",
      "properties":{
        "factors":{
          "type":"object",
          "properties":{k:{"type":"object","properties":{
              "score":{"type":"integer","minimum":0,"maximum":20},
              "comment":{"type":"string"},
              "deduct":{"type":"string"},
              "improve":{"type":"string"}
          }, "required":["score"]} for k in CRITERIA},
          "additionalProperties": False
        },
        "revised":{"type":"string"},
        "strengths":{"type":"array","items":{"type":"string"}},
        "risks":{"type":"array","items":{"type":"string"}},
        "improvements":{"type":"array","items":{"type":"string"}}
      },
      "required":["factors","revised"]
    }

    sys = ("너는 톱티어 면접 코치다. 아래 스키마에 맞춘 **한국어 JSON만** 출력하라. "
           "각 기준(score 0~20)을 채점하고 comment/감점(deduct)/개선(improve)을 간단히 채워라. "
           f"스키마: {json.dumps(schema, ensure_ascii=False)}")
    user = f"[면접 질문]\n{question}\n\n[후보자 답변]\n{answer}"
    resp = client.chat.completions.create(
        model=MODEL, temperature=0.3,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
        response_format={"type":"json_object"}
    )
    data = json.loads(resp.choices[0].message.content)

    factors = data.get("factors", {})
    sum_score = sum(int(factors[k]["score"]) for k in CRITERIA if k in factors and isinstance(factors[k].get("score"), int))
    data["sum_score"] = max(0, min(100, sum_score))
    return data

if st.button("채점 & 코칭", type="primary", use_container_width=True):
    if not st.session_state.get("current_question"):
        st.warning("먼저 '새 질문 받기'로 질문을 생성하세요.")
    elif not st.session_state.answer_text.strip():
        st.warning("답변을 작성해 주세요.")
    else:
        with st.spinner("채점/코칭 중..."):
            data = evaluate_answer(st.session_state["current_question"], st.session_state.answer_text)
            row = {k: (data["factors"].get(k,{}).get("score") if data.get("factors") else None) for k in CRITERIA}
            st.session_state.history.append({
                "ts": pd.Timestamp.now(),
                "question": st.session_state["current_question"],
                "answer": st.session_state.answer_text,
                "sum_score": data.get("sum_score"),
                "factors": row,
                "comments": {k: data["factors"].get(k,{}).get("comment") for k in CRITERIA} if data.get("factors") else {},
                "deducts":  {k: data["factors"].get(k,{}).get("deduct") for k in CRITERIA} if data.get("factors") else {},
                "improves": {k: data["factors"].get(k,{}).get("improve") for k in CRITERIA} if data.get("factors") else {},
                "strengths": data.get("strengths",[]),
                "risks": data.get("risks",[]),
                "improvements": data.get("improvements",[]),
                "revised": data.get("revised",""),
                "raw": data
            })

# =========================================================
# ⑤ 결과/레이더/CSV — 총점 일원화, 최신 vs 평균, 누적합 표
# =========================================================
st.divider()
st.subheader("피드백 결과")
if st.session_state.history:
    last = st.session_state.history[-1]
    c1,c2 = st.columns([1,3])
    with c1: st.metric("총점(/100)", last.get("sum_score","—"))
    with c2:
        st.markdown(f"**총점: {last.get('sum_score','—')}/100**")
        st.markdown("**2. 기준별 근거(점수/감점/개선):**")
        rows=[]
        for k in CRITERIA:
            sc = last["factors"].get(k)
            comment = (last.get("comments") or {}).get(k,"")
            deduct  = (last.get("deducts") or {}).get(k,"")
            improve = (last.get("improves") or {}).get(k,"")
            rows.append((f"{k}({sc if sc is not None else '-'}/20)", f"강점: {comment or '-'} / 감점: {deduct or '-'} / 개선: {improve or '-'}"))
        st.dataframe(pd.DataFrame(rows, columns=["기준(점수)","코멘트"]), use_container_width=True, hide_index=True)

        if last.get("strengths"):
            st.markdown("**3. 강점:**\n" + "\n".join([f"- {x}" for x in last["strengths"]]))
        if last.get("risks"):
            st.markdown("**4. 리스크:**\n" + "\n".join([f"- {x}" for x in last["risks"]]))
        if last.get("improvements"):
            st.markdown("**5. 개선 포인트:**\n" + "\n".join([f"- {x}" for x in last["improvements"]]))
        if last.get("revised"):
            st.markdown("**6. 수정본 답변:**")
            st.markdown(last["revised"])
else:
    st.caption("아직 채점 결과가 없습니다.")

st.divider()
st.subheader("역량 레이더 (세션 누적, NA는 0으로 표시)")

def history_df(hist):
    if not hist: return None
    rows=[]
    for h in hist:
        rows.append([h["factors"].get(k) for k in CRITERIA])
    df = pd.DataFrame(rows, columns=CRITERIA)
    return df

cdf = history_df(st.session_state.history)
if cdf is not None and not cdf.empty:
    latest = cdf.iloc[-1].fillna(0.0).astype(float).tolist()
    avg    = cdf.astype(float).mean(skipna=True).fillna(0.0).tolist()

    if PLOTLY_OK:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=latest+[latest[0]], theta=CRITERIA+[CRITERIA[0]],
            fill='toself', name="최신", opacity=0.7))
        fig.add_trace(go.Scatterpolar(
            r=avg+[avg[0]], theta=CRITERIA+[CRITERIA[0]],
            fill='toself', name="세션 평균", opacity=0.4))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,20])), showlegend=True, height=440)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(pd.DataFrame({"latest": latest, "avg": avg}, index=CRITERIA))

    # 최신 점수 행 + 합계/시도
    table = pd.DataFrame([cdf.iloc[-1].tolist()], columns=CRITERIA)
    table["합계(0~100)"] = table[CRITERIA].sum(axis=1, numeric_only=True)
    attempts = len(cdf)
    st.dataframe(table, use_container_width=True)
    st.caption(f"시도 횟수: {attempts}회")
else:
    st.caption("아직 역량 점수가 파싱된 코칭 결과가 없습니다.")

st.divider()
st.subheader("세션 리포트 (CSV)")
def build_report(hist):
    rows=[]
    for h in hist:
        row={"timestamp":h.get("ts"),"question":h.get("question"),"answer":h.get("answer"),
             "sum_score":h.get("sum_score")}
        for k in CRITERIA:
            row[f"score_{k}"] = (h.get("factors") or {}).get(k)
            row[f"comment_{k}"] = (h.get("comments") or {}).get(k)
            row[f"deduct_{k}"]  = (h.get("deducts") or {}).get(k)
            row[f"improve_{k}"] = (h.get("improves") or {}).get(k)
        row["revised"] = h.get("revised","")
        rows.append(row)
    cols = ["timestamp","question","answer","sum_score"] + \
           [f"score_{k}" for k in CRITERIA] + \
           [f"comment_{k}" for k in CRITERIA] + \
           [f"deduct_{k}" for k in CRITERIA] + \
           [f"improve_{k}" for k in CRITERIA] + ["revised"]
    return pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=cols)

rep = build_report(st.session_state.history)
st.download_button("CSV 다운로드", data=rep.to_csv(index=False).encode("utf-8-sig"),
                   file_name="interview_session_report.csv", mime="text/csv")

st.caption("※ ② 섹션은 요청에 따라 공고 페이지의 텍스트를 **그대로** 노출합니다(WebBaseLoader 우선, 실패 시 BS4).")
