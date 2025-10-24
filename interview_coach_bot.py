# interview_coach_bot.py
# v2.3 — WebBaseLoader 기반 공고 본문 수집/요약 추출 추가
#        (원문 파서와 병행, 실패 시 폴백) + 기존 채점/레이더/CSV 그대로 유지

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
except ImportError:
    st.error("`openai` 패키지가 필요합니다. requirements.txt에 openai를 추가했는지 확인하세요.")
    st.stop()

# NEW: WebBaseLoader (없으면 자동 폴백)
try:
    from langchain_community.document_loaders import WebBaseLoader
    WEBBASE_OK = True
except Exception:
    WEBBASE_OK = False

import requests
from bs4 import BeautifulSoup

# ---------- Page config ----------
st.set_page_config(page_title="회사 특화 가상 면접 코치", page_icon="🤖", layout="wide")

# ---------- Secrets ----------
def _secrets_file_exists() -> bool:
    cands = [
        os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml"),
        os.path.join(os.getcwd(), ".streamlit", "secrets.toml"),
    ]
    return any(os.path.exists(p) for p in cands)

def load_api_key_from_env_or_secrets() -> Optional[str]:
    k = os.getenv("OPENAI_API_KEY")
    if k: return k
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

# ---------- Utils ----------
def _clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()

def _snippetize(text: str, maxlen: int = 220) -> str:
    t = _clean_text(text)
    return t if len(t) <= maxlen else t[: maxlen - 1] + "…"

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

def _domain(u: str|None) -> str|None:
    if not u: return None
    try:
        if not u.startswith("http"): u = "https://" + u
        return urllib.parse.urlparse(u).netloc.lower().replace("www.","")
    except Exception:
        return None

# ---------- NAVER ----------
def _naver_api_get(api: str, params: dict, cid: str, csec: str):
    url = f"https://openapi.naver.com/v1/search/{api}.json"
    headers = {"X-Naver-Client-Id": cid, "X-Naver-Client-Secret": csec, **UA}
    r = requests.get(url, headers=headers, params=params, timeout=8)
    if r.status_code != 200:
        return None
    return r.json()

def naver_search_news(query: str, display: int = 10, sort: str = "date") -> list[dict]:
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

# ---------- 홈페이지 요약 후보 ----------
VAL_KEYS = ["핵심가치","가치","미션","비전","문화","원칙","철학","고객","데이터","혁신",
            "values","mission","vision","culture","principles","philosophy","customer","data","innovation"]

@st.cache_data(ttl=3600)
def fetch_site_snippets(base_url: str | None, company_name_hint: str | None = None) -> dict:
    if not base_url:
        return {"values": [], "recent": [], "site_name": None, "about": None}
    url0 = base_url.strip()
    if not url0.startswith("http"): url0 = "https://" + url0
    cand_paths = ["", "/", "/about", "/company", "/about-us", "/mission", "/values", "/culture"]
    values_found, recent_found = [], []
    site_name, about_para = None, None

    for path in cand_paths:
        url = url0.rstrip("/") + path
        try:
            r = requests.get(url, timeout=6, headers=UA)
            if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
                continue
            soup = BeautifulSoup(r.text, "html.parser")

            if site_name is None:
                og = soup.find("meta", {"property":"og:site_name"}) or soup.find("meta", {"name":"application-name"})
                if og and og.get("content"): site_name = _clean_text(og["content"])
                elif soup.title and soup.title.string: site_name = _clean_text(soup.title.string.split("|")[0])

            if about_para is None:
                hero = soup.find(["p","div"], class_=re.compile(r"(lead|hero|intro)", re.I)) if soup else None
                if hero: about_para = _snippetize(hero.get_text(" "))

            for tag in soup.find_all(["h1","h2","h3","p","li"]):
                txt = _clean_text(tag.get_text(separator=" "))
                if 10 <= len(txt) <= 240:
                    if any(k.lower() in txt.lower() for k in VAL_KEYS):
                        values_found.append(txt)
                    if any(k in txt for k in ["프로젝트","개발","출시","성과","project","launched","release","delivered","improved"]):
                        recent_found.append(txt)
        except Exception:
            continue

    def dedup(lst):
        seen=set(); out=[]
        for x in lst:
            if x not in seen: seen.add(x); out.append(x)
        return out
    values_found = dedup(values_found)[:6]
    recent_found = dedup(recent_found)[:6]
    return {"values": values_found, "recent": recent_found, "site_name": site_name, "about": about_para}

# ---------- 채용 링크 탐색 + 리스트→상세 공고 추적 ----------
CAREER_HINTS = ["careers","career","jobs","job","recruit","recruiting","join","hire","hiring",
                "채용","인재","입사지원","채용공고","인재영입","사람","커리어"]
SEARCH_ENGINES = ["https://duckduckgo.com/html/?q={query}"]
JOB_SITES = ["wanted.co.kr","saramin.co.kr","jobkorea.co.kr","rocketpunch.com",
             "indeed.com","linkedin.com","recruit.navercorp.com","kakao.recruit","naver"]

@st.cache_data(ttl=3600)
def discover_job_from_homepage(homepage: str, limit: int = 5) -> list[str]:
    if not homepage: return []
    try:
        if not homepage.startswith("http"): homepage = "https://" + homepage
        r = requests.get(homepage, timeout=8, headers=UA)
        if r.status_code != 200 or "text/html" not in r.headers.get("content-type",""):
            return []
        soup = BeautifulSoup(r.text, "html.parser")
        links=[]
        for path in ["careers","recruit","jobs","career","채용","인재영입","recruitment","join"]:
            links.append(urllib.parse.urljoin(homepage.rstrip("/") + "/", path))
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
                if role_hint and role_hint not in title:
                    continue
                return href
        if "saramin.co.kr" in dom:
            for a in soup.select("a[href*='view?idx=']"):
                return urllib.parse.urljoin(url, a.get("href"))
        if "jobkorea.co.kr" in dom:
            for a in soup.select("a[href*='/Recruit/GI_Read/']"):
                return urllib.parse.urljoin(url, a.get("href"))
        for a in soup.find_all("a", href=True):
            href = urllib.parse.urljoin(url, a.get("href"))
            if any(s in (_domain(href) or "") for s in JOB_SITES):
                if re.search(r"/(wd|jobs|job|view|read|detail|posting)/", href, re.I):
                    return href
    except Exception:
        return None
    return None

@st.cache_data(ttl=3600)
def discover_job_posting_urls(company_name: str, role: str, homepage: str|None, limit: int = 5) -> list[str]:
    urls = []
    urls += discover_job_from_homepage(homepage, limit=limit) if homepage else []
    resolved=[]
    for u in urls:
        if re.search(r"/(wd|view|read|detail|posting)/", u, re.I):
            resolved.append(u)
        else:
            detail = _first_detail_from_list(u, role_hint=role or "")
            if detail: resolved.append(detail)
    urls = resolved[:]
    if urls: return urls[:limit]

    if NAVER_ID and NAVER_SECRET:
        for dom in JOB_SITES:
            if len(urls) >= limit: break
            q = f"{company_name} {role} site:{dom}" if role else f"{company_name} 채용 site:{dom}"
            links = naver_search_web(q, display=7, sort="date")
            for lk in links:
                if _domain(lk) and dom in _domain(lk) and lk not in urls:
                    if not re.search(r"/(wd|view|read|detail|posting)/", lk, re.I):
                        lk2 = _first_detail_from_list(lk, role_hint=role or "")
                        if lk2: lk = lk2
                    urls.append(lk)
                if len(urls) >= limit: break
        if urls: return urls[:limit]

    site_part = " OR ".join([f'site:{d}' for d in JOB_SITES])
    q = f'{company_name} {role} ({site_part})' if role else f'{company_name} 채용 ({site_part})'
    for engine in SEARCH_ENGINES:
        url = engine.format(query=urllib.parse.quote(q))
        try:
            r = requests.get(url, timeout=8, headers=UA)
            if r.status_code != 200: continue
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("/l/?kh=-1&uddg="):
                    href = urllib.parse.unquote(href.split("/l/?kh=-1&uddg=")[-1])
                dom = _domain(href)
                if not dom: continue
                if any(d in dom for d in JOB_SITES):
                    if not re.search(r"/(wd|view|read|detail|posting)/", href, re.I):
                        d2 = _first_detail_from_list(href, role_hint=role or "")
                        if d2: href = d2
                    if href not in urls: urls.append(href)
                if len(urls) >= limit: break
        except Exception:
            continue
    return urls[:limit]

# ---------- 공고 파서(원문) ----------
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

def _extract_by_headings(soup: BeautifulSoup, heads: list[str]) -> Optional[list[str]]:
    if soup is None: return None
    pat = re.compile("|".join(heads), re.I)
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
        chunk = text[start:(m2.start() if m2 else start+1500)]
        items = [x.strip(" -•·▪︎—") for x in re.split(r"[•\n\r\t]+", chunk)]
        items = [i for i in items if 2<len(i)<300]
        result[key] = items[:24]
    return result

# ---------- NEW: WebBaseLoader로 공고 본문 로드 + LLM 구조화 ----------
def extract_with_webbase_and_llm(url: str, client: OpenAI, model: str, force_summary: bool=False) -> dict:
    """WebBaseLoader로 본문을 읽고 LLM으로 '주요업무/자격요건/우대사항'을 JSON으로 추출."""
    out = {"title": None, "responsibilities": [], "qualifications": [], "preferences": [], "company_intro": None}
    if not WEBBASE_OK:
        return out
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        body = "\n".join([d.page_content for d in docs if d.page_content])[:120_000]
        if not body.strip():
            return out

        sys = ("너는 채용공고 전문 요약기다. 아래 원문에서 '주요업무','자격요건','우대사항'을 **원문 문구를 최대한 보존**하는 "
               "불릿 리스트로 추출하고 JSON만 반환하라. 키는 responsibilities, qualifications, preferences. "
               "각 리스트는 3~12개 항목. 불필요한 말 금지.")
        if force_summary:
            sys = sys.replace("최대한 보존", "핵심 키워드 포함", "요약")

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

@st.cache_data(ttl=1800)
def parse_job_posting(url: str, client: OpenAI, model: str,
                      prefer_webbase: bool=True, force_summary: bool=False) -> dict:
    """prefer_webbase=True면 WebBaseLoader+LLM 우선, 부족하면 원문 파서 폴백."""
    # 0) WebBaseLoader + LLM
    if prefer_webbase and WEBBASE_OK:
        via_llm = extract_with_webbase_and_llm(url, client, model, force_summary=force_summary)
        if any(via_llm[k] for k in ("responsibilities","qualifications","preferences")):
            return via_llm

    # 1) BeautifulSoup 기반 원문 파싱
    out = {"title": None, "responsibilities": [], "qualifications": [], "preferences": [], "company_intro": None}
    try:
        r = requests.get(url, timeout=12, headers=UA)
        if r.status_code != 200 or "text/html" not in r.headers.get("content-type",""): return out
        soup = BeautifulSoup(r.text, "html.parser")
        resp = _extract_by_headings(soup, [r"주요\s*업무|담당\s*업무|업무|Responsibilities|Role|What\s+you('|’)ll\s+do"])
        qual = _extract_by_headings(soup, [r"자격\s*요건|지원\s*자격|Requirements|Qualifications|Must"])
        pref = _extract_by_headings(soup, [r"우대\s*사항|우대|Preferred|Plus|Nice\s*to\s*have"])

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

        meta_desc = soup.find("meta", {"name":"description"}) or soup.find("meta", {"property":"og:description"})
        if meta_desc and meta_desc.get("content"): out["company_intro"]=_snippetize(meta_desc["content"], 220)

        for k,vals in [("responsibilities",resp),("qualifications",qual),("preferences",pref)]:
            if vals:
                vals = [_clean_text(v)[:300] for v in vals if len(_clean_text(v))>1]
                out[k] = vals[:24]
    except Exception:
        pass
    return out

# ---------- OpenAI ----------
with st.sidebar:
    st.title("⚙️ 설정")
    API_KEY = load_api_key_from_env_or_secrets()
    if not API_KEY:
        st.info("환경변수/Secrets에서 키를 못 찾았습니다. 아래에 입력 후 엔터.")
        API_KEY = st.text_input("OPENAI_API_KEY", type="password")
    MODEL = st.selectbox("챗 모델", ["gpt-4o-mini","gpt-4o","gpt-4.1-mini"], index=0)
    EMBED_MODEL = st.selectbox("임베딩 모델", ["text-embedding-3-small","text-embedding-3-large"], index=0)

    try:
        import openai as _openai_pkg; _openai_ver = getattr(_openai_pkg, "__version__", None)
    except Exception: _openai_ver=None
    try:
        import httpx as _httpx_pkg; _httpx_ver = getattr(_httpx_pkg, "__version__", None)
    except Exception: _httpx_ver=None
    with st.expander("디버그: 시크릿/버전 상태"):
        st.write({
            "api_key_provided": bool(API_KEY),
            "naver_keys": bool(NAVER_ID and NAVER_SECRET),
            "openai_version": _openai_ver,
            "httpx_version": _httpx_ver,
            "webbaseloader_available": WEBBASE_OK
        })

if not API_KEY:
    st.error("OpenAI API Key가 필요합니다.")
    st.stop()
client = OpenAI(api_key=API_KEY, timeout=30.0)

# ==========================================================
# ① 회사/직무 입력 + 고급 수집 옵션
# ==========================================================
st.subheader("① 회사/직무 입력")
company_name_input = st.text_input("회사 이름", placeholder="예: 네이버 / Kakao / 삼성SDS")
role_title         = st.text_input("지원 직무명", placeholder="데이터 애널리스트 / ML 엔지니어 ...")
job_url_input      = st.text_input("채용 공고 URL(선택) — 없다면 자동 탐색")
homepage_input     = st.text_input("공식 홈페이지 URL(선택)", placeholder="https://...")

with st.expander("고급 수집 옵션"):
    prefer_webbase = st.checkbox("WebBaseLoader로 공고 본문 수집 **우선**", value=True, help="가능하면 WebBaseLoader로 본문을 읽어 LLM으로 구조화합니다.")
    force_summary  = st.checkbox("LLM 요약 강제(원문 파서 무시)", value=False, help="파서가 종종 실패할 때 강제로 요약 추출을 사용합니다.")

for k,v in [("company_state",{}),("answer_text",""),("history",[]),("current_question","")]:
    if k not in st.session_state: st.session_state[k]=v

def build_company_obj(name: str, homepage: str|None, role: str|None, job_url: str|None) -> dict:
    def _site(): return fetch_site_snippets(homepage or None, name)
    def _news(): return fetch_news(name, max_items=6)
    def _urls():
        if job_url: return [job_url]
        return discover_job_posting_urls(name, role or "", homepage, limit=6)

    with ThreadPoolExecutor(max_workers=3) as ex:
        fut = {ex.submit(fn): key for fn, key in [(_site,"site"),(_news,"news"),(_urls,"urls")]}
        ret = {}
        for f in as_completed(fut):
            try: ret[fut[f]] = f.result()
            except Exception: ret[fut[f]] = None

    site = ret.get("site") or {"values": [], "recent": [], "site_name": None, "about": None}
    urls = ret.get("urls") or []
    news = ret.get("news") or []

    jp = {"responsibilities": [], "qualifications": [], "preferences": [], "company_intro": None}
    if urls:
        jp = parse_job_posting(urls[0], client, MODEL, prefer_webbase=prefer_webbase, force_summary=force_summary)

    return {
        "company_name": name.strip() or "(회사명 미설정)",
        "homepage": homepage or None,
        "values": site.get("values", []),
        "recent_projects": site.get("recent", []),
        "company_intro_site": site.get("about"),
        "role": role or "",
        "responsibilities": jp["responsibilities"],
        "qualifications": jp["qualifications"],
        "preferences": jp["preferences"],
        "job_url": urls[0] if urls else (job_url or None),
        "news": news
    }

def generate_company_summary(c: dict) -> str:
    intro_src = c.get("company_intro_site") or ""
    news_titles = ", ".join([_snippetize(n["title"],70) for n in c.get("news", [])[:3]])
    sys = "너는 채용담당자다. 회사 소개만 2~3문장으로 간결히 요약(광고 표현 제거, 사실 위주)하라. 한국어."
    user = f"[회사명] {c.get('company_name','')}\n[회사 소개 원문 후보]\n{intro_src}\n[최근 뉴스 타이틀]\n{news_titles}"
    try:
        resp = client.chat.completions.create(
            model=MODEL, temperature=0.2,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}]
        )
        intro = resp.choices[0].message.content.strip()
    except Exception:
        intro = intro_src or "회사 소개 정보가 충분하지 않습니다."

    md = f"""**회사명**  
{c.get('company_name')}

**간단한 회사 소개(요약)**  
{intro}
"""
    return md

# 불러오기 버튼
if st.button("회사/직무 정보 불러오기", type="primary"):
    if not company_name_input.strip():
        st.warning("회사 이름을 입력해 주세요.")
    else:
        # 회사 변경 시 아래 실행결과 초기화
        st.session_state.history = []
        st.session_state.current_question = ""
        st.session_state.answer_text = ""
        with st.spinner("회사/직무/공고/뉴스 수집 중..."):
            cobj = build_company_obj(company_name_input, homepage_input or None, role_title or None, job_url_input or None)
            summary_md = generate_company_summary(cobj)
            st.session_state.company_state["company"] = cobj
            st.session_state.company_state["summary_md"] = summary_md
        st.success("회사 정보 갱신 완료")

company = st.session_state["company_state"].get("company", {
    "company_name": "(회사명 미설정)", "homepage": None, "values": [], "recent_projects": [],
    "company_intro_site": None, "role": "", "responsibilities": [], "qualifications": [], "preferences": [],
    "job_url": None, "news": []
})
summary_md = st.session_state["company_state"].get("summary_md", None)

# ==========================================================
# ② 회사 요약 / 채용 요건 (LLM 요약/원문 혼합)
# ==========================================================
st.subheader("② 회사 요약 / 채용 요건")
if summary_md:
    st.markdown(summary_md)
    cols = st.columns(2)
    with cols[0]:
        if company.get("job_url"): st.link_button("채용 공고 열기", company["job_url"])
    with cols[1]:
        if company.get("news"):
            st.write("최근 뉴스:")
            for n in company["news"][:3]:
                st.markdown(f"- [{_clean_text(n['title'])}]({n['link']})")

    st.markdown("---")
    c1,c2,c3 = st.columns(3)

    def _bullet(lst, title):
        st.markdown(f"**{title}(요약)**" if prefer_webbase or force_summary else f"**{title}(원문)**")
        if lst:
            for it in lst: st.markdown(f"- {it}")
        else:
            st.markdown(f"*공고에서 추출된 {title}이(가) 없습니다.*")

    with c1: _bullet(company.get("responsibilities") or [], "주요업무")
    with c2: _bullet(company.get("qualifications") or [], "자격요건")
    with c3: _bullet(company.get("preferences") or [], "우대사항")
else:
    st.info("위 입력을 완료하고 ‘회사/직무 정보 불러오기’를 누르면 표시됩니다.")

# ==========================================================
# ③ 질문 생성 (RAG 선택) — 이하 기존 그대로
# ==========================================================
st.subheader("③ 질문 생성")

@st.cache_data(ttl=3600)
def cached_embeddings(api_key: str, model: str, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 3), dtype=np.float32)
    _client = OpenAI(api_key=api_key)
    r = _client.embeddings.create(model=model, input=texts)
    return np.array([d.embedding for d in r.data], dtype=np.float32)

def embed_texts(client: OpenAI, embed_model: str, texts: list[str]) -> np.ndarray:
    return cached_embeddings(client.api_key, embed_model, texts)

with st.expander("RAG 옵션 (선택)"):
    rag_enabled = st.toggle("회사 문서 기반 질문/코칭 사용", value=True, key="rag_on")
    top_k = st.slider("검색 상위 K", 1, 8, 4, 1, key="topk")
    if "rag_store" not in st.session_state:
        st.session_state.rag_store = {"chunks": [], "embeds": None}
    docs = st.file_uploader("회사 문서 업로드 (TXT/MD/PDF, 여러 파일 가능)", type=["txt","md","pdf"], accept_multiple_files=True)
    chunk_size = st.slider("청크 길이(문자)", 400, 2000, 900, 100)
    chunk_ovlp = st.slider("오버랩(문자)", 0, 400, 150, 10)
    if docs:
        with st.spinner("문서 인덱싱 중..."):
            chunks=[]
            for up in docs:
                t = read_file_to_text(up)
                if t: chunks += chunk_text(t, chunk_size, chunk_ovlp)
            if chunks:
                embs = embed_texts(client, "text-embedding-3-small", chunks)
                st.session_state.rag_store["chunks"] += chunks
                if st.session_state.rag_store["embeds"] is None or st.session_state.rag_store["embeds"].size==0:
                    st.session_state.rag_store["embeds"] = embs
                else:
                    st.session_state.rag_store["embeds"] = np.vstack([st.session_state.rag_store["embeds"], embs])
                st.success(f"추가 청크 {len(chunks)}개")

def cosine_topk(matrix: np.ndarray, query: np.ndarray, k: int = 4):
    if matrix.size == 0:
        return np.array([]), np.array([], dtype=int)
    qn = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
    mn = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    sims = mn @ qn.T
    sims = sims.reshape(-1)
    idx = np.argsort(-sims)[:k]
    return sims[idx], idx

def retrieve_supports(qtext: str, k: int):
    store = st.session_state.rag_store
    chs, embs = store.get("chunks", []), store.get("embeds")
    if not st.session_state.get("rag_on") or embs is None or not chs:
        return []
    qv = embed_texts(client, "text-embedding-3-small", [qtext])
    scores, idxs = cosine_topk(embs, qv, k=k)
    return [("회사자료", float(s), chs[int(i)]) for s,i in zip(scores, idxs)]

TYPE_INSTRUCTIONS = {
    "행동(STAR)": "과거 실무 사례를 끌어내도록 S(상황)-T(과제)-A(행동)-R(성과)를 유도하는 질문",
    "기술 심층": "핵심 기술적 의사결정·트레이드오프·성능/비용/품질 지표를 파고드는 심층 질문",
    "핵심가치 적합성": "핵심가치와 태도를 검증하는, 상황기반 행동을 유도하는 질문",
    "역질문": "지원자가 회사를 평가할 수 있도록 통찰력 있는 역질문"
}
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

q_type = st.selectbox("질문 유형", list(TYPE_INSTRUCTIONS.keys()))
level  = st.selectbox("난이도/연차", ["주니어","미들","시니어"])
hint   = st.text_input("질문 생성 힌트(선택)", placeholder="예: 전환 퍼널 / 모델 성능-비용 / 데이터 품질")

if "history" not in st.session_state:
    st.session_state.history = []
if "current_question" not in st.session_state:
    st.session_state.current_question = ""

if st.button("새 질문 받기", use_container_width=True, type="primary"):
    st.session_state.answer_text = ""
    try:
        supports=[]
        if st.session_state.get("rag_on"):
            base_q = hint.strip() or f"{company.get('role','')} {' '.join(company.get('responsibilities', [])[:3])}"
            supports = retrieve_supports(base_q, st.session_state.get("topk",4))

        ctx = textwrap.dedent(f"""
        [회사명] {company.get('company_name','')}
        [회사 소개] {company.get('company_intro_site') or ''}
        [모집 분야] {company.get('role','')}
        [주요 업무] {", ".join(company.get('responsibilities', [])[:6])}
        [자격 요건] {", ".join(company.get('qualifications', [])[:6])}
        [우대 사항] {", ".join(company.get('preferences', [])[:6])}
        [핵심가치] {", ".join(company.get('values', [])[:6])}
        """).strip()
        rag_note = ""
        if supports:
            joined="\n".join([f"- ({s:.2f}) {txt[:200]}" for _,s,txt in supports[:3]])
            rag_note=f"\n[근거 발췌]\n{joined}"

        seed = int(time.time()*1000) % 2_147_483_647
        sys = f"""너는 '{company.get('company_name','')}'의 '{company.get('role','')}' 면접관이다.
회사/직무 컨텍스트(업무/자격/우대), (있다면) 근거 문서를 반영하여 **{q_type}** 유형({TYPE_INSTRUCTIONS[q_type]})의 질문 **6개 후보**를 한국어로 생성하라.
서로 **형태·관점·키워드**가 달라야 하며 난이도는 {level}. 지표/수치/기간/규모/리스크 요소를 섞어라.
포맷: 1) ... 2) ... 3) ... ... (한 줄씩)"""
        user = f"""[컨텍스트]\n{ctx}\n[힌트]\n{hint}{rag_note}\n[랜덤시드] {seed}"""

        resp = client.chat.completions.create(
            model=MODEL, temperature=0.9,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}]
        )
        raw = resp.choices[0].message.content.strip()
        cands = [re.sub(r'^\s*\d+\)\s*','',line).strip() for line in raw.splitlines() if re.match(r'^\s*\d+\)', line)]
        if not cands:
            cands = [l.strip("- ").strip() for l in raw.splitlines() if len(l.strip())>0][:6]
        hist_qs = [h["question"] for h in st.session_state.get("history", [])][-10:]
        selected = pick_diverse(cands, hist_qs)
        st.session_state.current_question = selected or (cands[0] if cands else "질문 생성 실패")
        st.session_state.last_supports_q = supports
    except Exception as e:
        st.error(f"질문 생성 오류: {e}")

st.text_area("질문", height=110, value=st.session_state.get("current_question",""))
if st.session_state.get("rag_on") and st.session_state.get("last_supports_q"):
    with st.expander("질문 생성에 사용된 근거 보기"):
        for i, (_, sc, txt) in enumerate(st.session_state.last_supports_q, 1):
            st.markdown(f"**[{i}] sim={sc:.3f}**\n\n{txt[:600]}{'...' if len(txt)>600 else ''}")
            st.markdown("---")

# ==========================================================
# ④ 나의 답변 / 코칭 — (기존 v2.2 채점/개선/수정본 답변 유지)
# ==========================================================
st.subheader("④ 나의 답변 / 코칭")

CRITERIA = [
    "문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치",
    "시스템설계","트레이드오프","성능/비용","품질/신뢰성","리스크관리",
    "보안/컴플라이언스","실험/검증","영향도","서술력"
]
QUESTION_MAP = [
    (re.compile(r"배치|스트리밍|카프카|플링크|스파크|파이프라인|아키텍처", re.I),
     {"시스템설계":0.3,"트레이드오프":0.25,"성능/비용":0.2,"품질/신뢰성":0.15,"데이터/지표":0.1}),
    (re.compile(r"지표|kpi|metric|측정|퍼널|분석", re.I),
     {"데이터/지표":0.4,"문제정의":0.2,"영향도":0.2,"실험/검증":0.2}),
    (re.compile(r"보안|security|침해|컴플라이언스|gdpr|hipaa|인증", re.I),
     {"보안/컴플라이언스":0.4,"리스크관리":0.3,"품질/신뢰성":0.15,"시스템설계":0.15}),
    (re.compile(r"협업|갈등|커뮤니케이션|협의|조율", re.I),
     {"협업/커뮤니케이션":0.5,"문제정의":0.2,"영향도":0.3}),
]
def detect_criteria_weights(question: str) -> Dict[str,float]:
    q = question or ""
    for pat, weights in QUESTION_MAP:
        if pat.search(q):
            return weights
    return {"문제정의":0.2,"데이터/지표":0.2,"실행력/주도성":0.3,"협업/커뮤니케이션":0.15,"고객가치":0.15}

def coach_answer(company: dict, question: str, answer: str, supports: list[Tuple[str,float,str]]) -> dict:
    q_trim = (question or "")[:500]
    a_trim = (answer or "")[:1400]
    ctx = textwrap.dedent(f"""
    [회사명] {company.get('company_name','')}
    [모집 분야] {company.get('role','')}
    [주요 업무] {", ".join(company.get('responsibilities', [])[:6])}
    [자격 요건] {", ".join(company.get('qualifications', [])[:6])}
    [우대 사항] {", ".join(company.get('preferences', [])[:6])}
    """).strip()

    weights = detect_criteria_weights(q_trim)
    crit_list = list(weights.keys())

    sys = (
        "너는 톱티어 면접 코치다. 아래 기준들에 대해서만 0~20 정수 점수와 '감점 요인', '개선 포인트'를 JSON으로 반환하고, "
        "'revised_answer' 키에 수정본 답변(STAR, 10~14문장)을 담아라.\n"
        "출력 예시: {\"문제정의\":{\"score\":14,\"penalty\":\"핵심 제약 누락\",\"improve\":\"문제 경계와 비즈니스 KPI를 먼저 명시\"}, ... , \"revised_answer\":\"...\"}\n"
        "존재하지 않는 기준은 넣지 말고, 한국어로 간결/구체적으로 작성하라."
    )
    user = f"""[컨텍스트]\n{ctx}\n\n[면접 질문]\n{q_trim}\n\n[후보자 답변]\n{a_trim}\n\n[채점 기준 목록]\n{', '.join(crit_list)}"""

    resp = client.chat_completions.create if hasattr(client, "chat_completions") else client.chat.completions.create
    out = resp(model=MODEL, temperature=0.25,
               messages=[{"role":"system","content":sys},{"role":"user","content":user}])
    content = out.choices[0].message.content.strip()

    data=None
    try:
        m = re.search(r"\{.*\}", content, re.S)
        if m: data = json.loads(m.group(0))
    except Exception:
        data=None

    scores = {c: None for c in CRITERIA}
    penalties, improves = {}, {}
    revised = ""
    if isinstance(data, dict):
        revised = data.get("revised_answer","")
        for c in crit_list:
            obj = data.get(c) or {}
            try:
                sc = int(obj.get("score"))
                sc = max(0, min(20, sc))
                scores[c] = sc
                penalties[c] = _clean_text(obj.get("penalty",""))
                improves[c]  = _clean_text(obj.get("improve",""))
            except Exception:
                pass

    used = [(c, s, weights[c]) for c,s in scores.items() if s is not None and c in weights]
    if used:
        num = sum(s * w for _,s,w in used)
        den = sum(w for _,_,w in used)
        final_score = int(round((num/den) * 5))  # 0~100
    else:
        final_score = 0

    lines = [f"총점: {final_score}/100", "", "2. 기준별 근거(점수/감점/개선):"]
    if used:
        for c,s,_ in used:
            p = penalties.get(c,"")
            im= improves.get(c,"")
            lines.append(f"- **{c}({s}/20)**  감점: {p or '—'} / 개선: {im or '—'}")
    else:
        lines.append("- (해당 기준 없음)")
    if revised:
        lines += ["", "5. 수정본 답변:", revised]

    comp_vector = [scores.get(c) if scores.get(c) is not None else 0 for c in CRITERIA]

    return {
        "raw": "\n".join(lines),
        "score": final_score,
        "competencies": comp_vector,
        "labels": CRITERIA,
        "scores_raw": scores,
    }

ans = st.text_area("여기에 답변을 작성하세요 (STAR 권장: 상황-과제-행동-성과)", height=180, key="answer_text")

if st.button("채점 & 코칭", type="primary", use_container_width=True):
    if not st.session_state.get("current_question"):
        st.warning("먼저 '새 질문 받기'로 질문을 생성하세요.")
    elif not st.session_state.answer_text.strip():
        st.warning("답변을 작성해 주세요.")
    else:
        with st.spinner("코칭 중..."):
            sups=[]
            if st.session_state.get("rag_on"):
                q_for_rag = (st.session_state["current_question"][:500]
                             + "\n" + st.session_state.answer_text[:800])
                sups = retrieve_supports(q_for_rag, st.session_state.get("topk",4))
            res = coach_answer(company, st.session_state["current_question"], st.session_state.answer_text, sups)
            st.session_state.history.append({
                "ts": pd.Timestamp.now(),
                "question": st.session_state["current_question"],
                "user_answer": st.session_state.answer_text,
                "score": res.get("score"),
                "feedback": res.get("raw"),
                "supports": sups,
                "competencies": res.get("competencies"),
                "labels": res.get("labels"),
                "scores_raw": res.get("scores_raw")
            })

# ==========================================================
# ⑤ 결과/레이더/CSV — 최신 vs 세션평균 (2중 폴라), 누적합/시도횟수 표시
# ==========================================================
st.divider()
st.subheader("피드백 결과")
if st.session_state.history:
    last = st.session_state.history[-1]
    c1,c2 = st.columns([1,3])
    with c1: st.metric("총점(/100)", last.get("score","—"))
    with c2: st.markdown(last.get("feedback",""))

    if st.session_state.get("rag_on") and last.get("supports"):
        with st.expander("코칭에 사용된 근거 보기"):
            for i,(_,sc,txt) in enumerate(last["supports"],1):
                st.markdown(f"**[{i}] sim={sc:.3f}**\n\n{txt[:800]}{'...' if len(txt)>800 else ''}")
                st.markdown("---")
else:
    st.info("아직 결과가 없습니다.")

st.divider()
st.subheader("역량 레이더 (세션 누적, NA는 0으로 표시)")

def comp_df(hist):
    rows=[]; labels=CRITERIA
    for h in hist:
        vec = h.get("competencies")
        if not vec: continue
        if len(vec) < len(labels): vec += [0]*(len(labels)-len(vec))
        rows.append(vec[:len(labels)])
    if not rows: return None, labels
    df = pd.DataFrame(rows, columns=labels)
    return df, labels

cdf, labels = comp_df(st.session_state.history)
if cdf is not None:
    latest = cdf.iloc[-1].tolist()
    avg    = cdf.mean().tolist()  # 세션 평균(누적 성향)
    if PLOTLY_OK:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=latest+[latest[0]], theta=labels+[labels[0]],
                                      fill='toself', name="최신", opacity=0.7))
        fig.add_trace(go.Scatterpolar(r=avg+[avg[0]], theta=labels+[labels[0]],
                                      fill='toself', name="세션 평균", opacity=0.4))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,20])),
                          showlegend=True, height=440)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(pd.DataFrame({"latest": latest, "avg": avg}, index=labels))

    attempts = len(cdf)
    sum_row  = (cdf.sum()).astype(int).to_dict()
    last = st.session_state.history[-1]
    raw = last.get("scores_raw", {})
    disp = []
    for c in labels:
        v = raw.get(c)
        disp.append("-" if v is None else v)
    disp_df = pd.DataFrame([disp], columns=labels)
    disp_df["누적합(각축)"] = int(sum(v for v in cdf.iloc[:, :].sum().tolist()))
    disp_df["시도횟수"] = attempts
    st.dataframe(disp_df, use_container_width=True)
    st.caption("파란색: 최신 / 초록색: 세션 평균. 표는 최신 점수(NA='-')와 세션 누적합·시도횟수를 보여줍니다.")
else:
    st.caption("아직 역량 점수가 파싱된 코칭 결과가 없습니다.")

st.divider()
st.subheader("세션 리포트 (CSV)")
def build_report(hist):
    rows=[]
    for h in hist:
        row={"timestamp":h.get("ts"),"question":h.get("question"),"user_answer":h.get("user_answer"),
             "score":h.get("score"),"feedback_raw":h.get("feedback")}
        comps=h.get("competencies")
        if comps:
            for k,v in zip(CRITERIA, comps): row[f"comp_{k}"]=v
            row["comp_sum"] = sum([int(v) for v in comps])
        sups=h.get("supports") or []
        row["supports_preview"]=" || ".join([s[2][:120].replace("\n"," ") for s in sups])
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["timestamp","question","user_answer","score","feedback_raw","supports_preview","comp_sum"])
rep = build_report(st.session_state.history)
st.download_button("CSV 다운로드", data=rep.to_csv(index=False).encode("utf-8-sig"),
                   file_name="interview_session_report.csv", mime="text/csv")

st.caption("검색→상세 공고 URL→(WebBaseLoader 본문 수집)→LLM 구조화 추출. 실패 시 원문 파서로 폴백합니다.")
