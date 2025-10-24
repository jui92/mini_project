# -*- coding: utf-8 -*-
# interview_coach_bot.py
# v3.0 — 회사/직무 검색→채용 상세 URL 탐색→WebBaseLoader 본문 수집→LLM 구조화 추출(주요업무/자격요건/우대사항)
#       원문 파서 폴백, 세로형 출력, 원문 보기, 총점 일원화, 레이더 최신/평균, 누적합 표, 캐시/병렬 최적화

import os, io, re, json, textwrap, urllib.parse, difflib, random, time, hashlib
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

# --- NEW: WebBaseLoader import & flag
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
# NAVER Open API
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
    if not (cid and csec):
        return []
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
    if not (cid and csec):
        return []
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
# 홈페이지 소개 후보
# =========================================================
VAL_KEYS = ["value","values","mission","vision","culture","고객","가치","문화","원칙","철학","혁신"]

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_site_intro(base_url: str|None) -> list[str]:
    if not base_url: return []
    url0 = base_url.strip()
    if not url0.startswith("http"): url0 = "https://" + url0
    cand_paths = ["", "/", "/about", "/company", "/about-us"]
    about_candidates = []
    for path in cand_paths:
        url = url0.rstrip("/") + path
        try:
            r = requests.get(url, timeout=6, headers=UA)
            if r.status_code != 200 or "text/html" not in r.headers.get("content-type",""):
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup.find_all(["p","div","li","section"]):
                txt = _clean_text(tag.get_text(" "))
                if 50 <= len(txt) <= 400 and any(k in txt.lower() for k in ["company","service","solution","platform","고객","서비스","제품","회사"]):
                    about_candidates.append(txt)
        except Exception:
            continue
    # dedup
    seen=set(); outs=[]
    for t in about_candidates:
        if t not in seen:
            seen.add(t); outs.append(t)
    return outs[:8]

# =========================================================
# 채용 링크 탐색 + 상세 공고
# =========================================================
CAREER_HINTS = ["careers","recruit","jobs","career","채용","인재영입","recruitment","join"]
SEARCH_ENGINES = ["https://duckduckgo.com/html/?q={query}"]
JOB_SITES = ["wanted.co.kr","saramin.co.kr","jobkorea.co.kr","rocketpunch.com","linkedin.com","indeed.com"]

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
# 공고 원문 파서 + LLM 구조화 + WebBaseLoader 덤프
# =========================================================
def _text_items_from_container(el) -> list[str]:
    if el is None: return []
    items = []
    for li in el.find_all(["li","p"]):
        t = _clean_text(li.get_text(" "))
        if len(t) >= 2: items.append(t)
    if not items:
        t = _clean_text(el.get_text(" "))
        parts = [x.strip(" •·▪︎-—") for x in re.split(r"[•·▪︎\-\n\r\t]+", t) if len(x.strip())>1]
        items = parts[:]
    seen=set(); out=[]
    for x in items:
        if x not in seen:
            seen.add(x); out.append(x[:300])
    return out

def _extract_by_headings(soup: BeautifulSoup, heads_regex: str) -> Optional[list[str]]:
    if soup is None: return None
    pat = re.compile(heads_regex, re.I)
    for h in soup.find_all(re.compile("^h[1-4]$")):
        title = _clean_text(h.get_text())
        if not pat.search(title): continue
        nxt = h.find_next_sibling()
        buf=[]; stop={"h1","h2","h3","h4"}
        while nxt and nxt.name not in stop:
            if nxt.name in {"div","section","article","ul","ol","p"}:
                buf.extend(_text_items_from_container(nxt))
            nxt = nxt.find_next_sibling()
        buf = [b for b in buf if len(b)>1]
        if buf: return buf[:24]
    return None

def _jsonld_job(soup: BeautifulSoup) -> dict:
    out = {"responsibilities": None, "qualifications": None, "preferences": None}
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(s.string or "")
            seq = data if isinstance(data, list) else [data]
            for obj in seq:
                typ = obj.get("@type") if isinstance(obj, dict) else None
                if (isinstance(typ, list) and "JobPosting" in typ) or typ == "JobPosting":
                    desc = _clean_text(obj.get("description",""))
                    if not desc: continue
                    parts = [p.strip(" -•·▪︎—") for p in re.split(r"[•\n\r\t]+", desc) if len(p.strip())>2]
                    resp, qual, pref = [], [], []
                    for p in parts:
                        if re.search(r"자격|요건|qual", p, re.I): qual.append(p)
                        elif re.search(r"우대|prefer|plus|nice", p, re.I): pref.append(p)
                        else: resp.append(p)
                    out = {"responsibilities":resp or None, "qualifications":qual or None, "preferences":pref or None}
                    return out
        except Exception:
            continue
    return out

def _whole_document_fallback(soup: BeautifulSoup) -> dict:
    text = _clean_text(soup.get_text(" "))
    patterns = {
        "responsibilities": r"(주요\s*업무|담당\s*업무|업무\s*내용|Responsibilities|Role|What\s+you('|’)ll\s+do)",
        "qualifications":   r"(자격\s*요건|지원\s*자격|Requirements|Qualifications|Must\s*have)",
        "preferences":      r"(우대\s*사항|우대|Preferred|Plus|Nice\s*to\s*have)",
    }
    result={"responsibilities":[], "qualifications":[], "preferences":[]}
    for key, pat in patterns.items():
        m = re.search(pat, text, re.I)
        if not m: continue
        start = m.end()
        next_pat = re.compile("|".join([p for k,p in patterns.items() if k!=key]), re.I)
        m2 = next_pat.search(text, start)
        chunk = text[start:(m2.start() if m2 else start+1800)]
        items = [x.strip(" -•·▪︎—") for x in re.split(r"[•\n\r\t]+", chunk)]
        items = [i for i in items if 2<len(i)<300]
        result[key] = items[:24]
    return result

# --- NEW: WebBaseLoader 전체 본문 덤프
@st.cache_data(ttl=1800, show_spinner=False)
def webbase_dump_all_text(url: str) -> str:
    if not WEBBASE_OK or not url:
        return ""
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        full = "\n\n".join([getattr(d, "page_content", "") for d in docs if getattr(d, "page_content", "")])
        return full
    except Exception as e:
        return f"[WebBaseLoader 에러] {e}"

# --- NEW: WebBaseLoader + LLM 구조화 추출
def extract_with_webbase_and_llm(url: str, client: OpenAI, model: str, force_summary: bool=False) -> dict:
    out = {"responsibilities": [], "qualifications": [], "preferences": []}
    if not WEBBASE_OK or not url:
        return out
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        body = "\n".join([d.page_content for d in docs if getattr(d, "page_content", "")])[:120_000]
        if not body.strip():
            return out
        sys = ("너는 채용공고 전문 요약기다. 아래 원문에서 '주요업무','자격요건','우대사항'을 **원문 문구 보존** 우선으로 "
               "3~12개 불릿씩 추출하고, JSON만 반환. 키: responsibilities, qualifications, preferences.")
        if force_summary:
            sys = sys.replace("보존", "핵심 요약")
        user = f"[원문]\n{body}"
        resp = client.chat.completions.create(
            model=model, temperature=0.2,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}]
        )
        text = resp.choices[0].message.content.strip()
        m = re.search(r"\{.*\}", text, re.S)
        if not m: 
            return out
        data = json.loads(m.group(0))
        for k in ["responsibilities","qualifications","preferences"]:
            lst = data.get(k) or []
            lst = [ _clean_text(x)[:300] for x in lst if _clean_text(x) ]
            out[k] = lst[:24]
        return out
    except Exception:
        return out

@st.cache_data(show_spinner=False, ttl=1800)
def parse_job_posting(url: str, client: OpenAI, model: str,
                      prefer_webbase: bool=True, force_summary: bool=False) -> dict:
    # WebBaseLoader+LLM 우선
    if prefer_webbase and WEBBASE_OK:
        via = extract_with_webbase_and_llm(url, client, model, force_summary=force_summary)
        if any(via[k] for k in ("responsibilities","qualifications","preferences")):
            return via

    # BeautifulSoup 기반 폴백
    out = {"responsibilities": [], "qualifications": [], "preferences": []}
    try:
        r = requests.get(url, timeout=12, headers=UA)
        if r.status_code != 200 or "text/html" not in r.headers.get("content-type",""): return out
        soup = BeautifulSoup(r.text, "html.parser")

        resp = _extract_by_headings(soup, r"주요\s*업무|담당\s*업무|업무|Responsibilities|Role|What\s+you('|’)ll\s+do")
        qual = _extract_by_headings(soup, r"자격\s*요건|지원\s*자격|Requirements|Qualifications|Must")
        pref = _extract_by_headings(soup, r"우대\s*사항|우대|Preferred|Plus|Nice\s*to\s*have")

        if not (resp and qual and pref):
            jd = _jsonld_job(soup)
            resp = resp or jd.get("responsibilities")
            qual = qual or jd.get("qualifications")
            pref = pref or jd.get("preferences")

        if not (resp or qual or pref):
            whole = _whole_document_fallback(soup)
            resp = whole.get("responsibilities")
            qual = whole.get("qualifications")
            pref = whole.get("preferences")

        def cut(lst): 
            return [_clean_text(x)[:300] for x in (lst or []) if _clean_text(x)] [:24]
        out["responsibilities"] = cut(resp)
        out["qualifications"]   = cut(qual)
        out["preferences"]      = cut(pref)
    except Exception:
        pass
    return out

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
# ① 회사/직무 입력 + 고급 옵션
# =========================================================
st.subheader("① 회사/직무 입력")
c1, c2 = st.columns(2)
with c1:
    company_name_input = st.text_input("회사 이름", placeholder="예: 네이버 / Kakao / 삼성SDS")
with c2:
    role_title = st.text_input("지원 직무명", placeholder="데이터 애널리스트 / ML 엔지니어 ...")
job_url_input  = st.text_input("채용 공고 URL(선택) — 없다면 자동 탐색")
homepage_input = st.text_input("공식 홈페이지 URL(선택)", placeholder="https://...")

with st.expander("고급 수집 옵션"):
    prefer_webbase = st.checkbox("WebBaseLoader로 공고 본문 수집 우선", value=True)
    force_summary  = st.checkbox("LLM 요약 강제(원문 파서 무시)", value=False)

for k,v in [("company", None), ("answer_text",""), ("history",[]), ("current_question","")]:
    if k not in st.session_state: st.session_state[k]=v

# =========================================================
# 회사 컨텍스트 구성
# =========================================================
def llm_summarize_intro(candidates: list[str], company: str) -> str:
    if not candidates: return ""
    sys = "너는 채용 담당자다. 회사 소개 문장 후보를 2~3문장으로 간결하게 요약하라. 과장/광고 문구는 제거."
    user = "회사명: {}\n\n후보 문장:\n- {}".format(company, "\n- ".join([_snippetize(t, 400) for t in candidates[:8]]))
    resp = client.chat.completions.create(
        model=MODEL, temperature=0.2,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}]
    )
    return resp.choices[0].message.content.strip()

def build_company_obj(name: str, homepage: str|None, role: str|None, job_url: str|None) -> dict:
    def _intro(): return fetch_site_intro(homepage or None)
    def _news():  return fetch_news(name, max_items=6)
    def _urls():
        if job_url: return [job_url]
        return discover_job_posting_urls(name, role or "", homepage, limit=6)

    with ThreadPoolExecutor(max_workers=3) as ex:
        fut = {ex.submit(fn): key for fn, key in [(_intro,"intro"),(_news,"news"),(_urls,"urls")]}
        ret = {}
        for f in as_completed(fut):
            try: ret[fut[f]] = f.result()
            except Exception: ret[fut[f]] = None

    intro_candidates = ret.get("intro") or []
    news = ret.get("news") or []
    urls = ret.get("urls") or []

    responsibilities, qualifications, preferences = [], [], []
    if urls:
        parsed = parse_job_posting(urls[0], client, MODEL, prefer_webbase=prefer_webbase, force_summary=force_summary)
        responsibilities = parsed["responsibilities"]
        qualifications   = parsed["qualifications"]
        preferences      = parsed["preferences"]

    intro_summary = ""
    try:
        intro_summary = llm_summarize_intro(intro_candidates, name) if intro_candidates else ""
    except Exception:
        intro_summary = ""

    return {
        "company_name": name.strip() or "(회사명 미설정)",
        "homepage": homepage or None,
        "role": role or "",
        "company_intro": intro_summary or "회사 소개를 요약할 수 있는 정보가 충분하지 않습니다.",
        "job_url": urls[0] if urls else (job_url or None),
        "responsibilities": responsibilities,
        "qualifications": qualifications,
        "preferences": preferences,
        "news": news
    }

def render_company_summary(c: dict):
    st.markdown(f"**회사명**  \n{c.get('company_name')}")
    st.markdown(f"**간단한 회사 소개(요약)**  \n{c.get('company_intro','')}")
    cols = st.columns(2)
    with cols[0]:
        if c.get("job_url"): st.link_button("채용 공고 열기", c["job_url"])
    with cols[1]:
        if c.get("homepage"): st.link_button("홈페이지 열기", c["homepage"])
    st.markdown("---")
    a,b,d = st.columns(3)
    def vlist(col, title, items):
        with col:
            st.markdown(f"### {title} (요약/원문 혼합)")
            if items:
                st.markdown("\n".join([f"- {x}" for x in items]))
            else:
                st.caption(f"{title} 추출 결과가 없습니다.")

    vlist(a, "주요업무", c.get("responsibilities", []))
    vlist(b, "자격요건", c.get("qualifications", []))
    vlist(d, "우대사항", c.get("preferences", []))

# 불러오기 버튼
if st.button("회사/직무 정보 불러오기", type="primary"):
    if not company_name_input.strip():
        st.warning("회사 이름을 입력해 주세요.")
    else:
        st.session_state.history = []
        st.session_state.current_question = ""
        st.session_state.answer_text = ""
        with st.spinner("회사/직무/공고/뉴스 수집 중..."):
            st.session_state.company = build_company_obj(company_name_input, homepage_input or None, role_title or None, job_url_input or None)
        st.success("회사 정보 갱신 및 실행결과 초기화 완료!")

company = st.session_state.get("company")

# =========================================================
# ② 회사 요약 / 채용 요건 + 원문 보기
# =========================================================
st.subheader("② 회사 요약 / 채용 요건")
if company:
    render_company_summary(company)

    # --- NEW: 원문 보기(WebBaseLoader) ---
    with st.expander("원문 보기 (WebBaseLoader로 페이지 텍스트 그대로 보기)"):
        if not WEBBASE_OK:
            st.info("requirements.txt에 `langchain-community>=0.2.0` 추가 후 배포하세요.")
        else:
            default_url = company.get("job_url") or ""
            raw_url = st.text_input("대상 URL", value=default_url, placeholder="https://... (공고 상세 URL 권장)")
            if st.button("원문 불러오기", use_container_width=True, key="btn_dump_webbase"):
                if not raw_url:
                    st.warning("URL을 입력하세요.")
                else:
                    with st.spinner("WebBaseLoader로 원문 수집 중..."):
                        fulltext = webbase_dump_all_text(raw_url)
                    if not fulltext:
                        st.error("텍스트를 가져오지 못했습니다. 로그인/차단 여부를 확인하세요.")
                    else:
                        st.success("원문을 불러왔습니다.")
                        st.text_area("페이지 텍스트(전부)", value=fulltext, height=420)
                        st.download_button(
                            "원문 텍스트 다운로드",
                            data=fulltext.encode("utf-8"),
                            file_name="job_posting_raw.txt",
                            mime="text/plain"
                        )
else:
    st.info("위 입력을 완료하고 ‘회사/직무 정보 불러오기’를 눌러 표시하세요.")

# =========================================================
# ③ 질문 생성
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
        [주요 업무] {", ".join((company or {}).get('responsibilities', [])[:6])}
        [자격 요건] {", ".join((company or {}).get('qualifications', [])[:6])}
        [우대 사항] {", ".join((company or {}).get('preferences', [])[:4])}
        [최근 이슈/뉴스] {news_titles}
        """).strip()
        seed = int(time.time()*1000) % 2_147_483_647
        sys = f"""너는 '{(company or {}).get('company_name','회사')}'의 면접관이다.
컨텍스트/채용 3요소/이슈를 반영하여 **{q_type}** 유형({TYPE_INSTRUCTIONS[q_type]})의 질문 **6개**를 한국어로 생성하라.
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
# ④ 나의 답변 / 채점 & 코칭 (총점=기준 합산 0~100)
# =========================================================
st.subheader("④ 나의 답변 / 코칭")
ans = st.text_area("여기에 답변을 작성하세요 (STAR 권장: 상황-과제-행동-성과)", height=200, key="answer_text")

CRITERIA = ["문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치"]
def evaluate_answer(company: dict, question: str, answer: str) -> dict:
    news = ", ".join([_snippetize(n["title"], 70) for n in (company or {}).get("news", [])[:3]])
    ctx = textwrap.dedent(f"""
    [회사명] {(company or {}).get('company_name','')}
    [모집 분야] {(company or {}).get('role','')}
    [주요 업무] {", ".join((company or {}).get('responsibilities', [])[:6])}
    [자격 요건] {", ".join((company or {}).get('qualifications', [])[:6])}
    [우대 사항] {", ".join((company or {}).get('preferences', [])[:4])}
    [최근 이슈/뉴스] {news}
    """).strip()

    schema = {
      "type":"object",
      "properties":{
        "overall":{"type":"integer","minimum":0,"maximum":100},
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
           "각 기준(score 0~20)은 질문/회사 맥락/채용 3요소 부합으로 채점하고 comment/감점(deduct)/개선(improve)을 간단히 채워라. "
           f"스키마: {json.dumps(schema, ensure_ascii=False)}")
    user = f"""[회사/직무 컨텍스트]\n{ctx}\n\n[면접 질문]\n{question}\n\n[후보자 답변]\n{answer}"""
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
            data = evaluate_answer(company or {}, st.session_state["current_question"], st.session_state.answer_text)
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

    # 최신 점수 행 + 누적합/시도
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

st.caption("검색→상세 공고 URL→WebBaseLoader 본문 수집→LLM 구조화 추출(주요업무/자격요건/우대사항). 실패 시 원문 파서 폴백.")
