# -*- coding: utf-8 -*-
# ==========================================================
# 회사 특화 가상 면접 코치 (텍스트 전용 / RAG + 레이더 + CSV)
# - 직무 선택 & 채용공고 자동 수집(권장: URL 입력, 없으면 검색 시도)
# - 회사 뉴스/최근 이슈 반영 (Google News RSS)
# - 질문 다양성 강화: 후보 N개 생성 + 반중복 선택 + 무작위 포커스
# - 채용공고 기준 요약(회사/간단 소개/모집분야/주요 업무/자격 요건)
# - Streamlit Cloud 호환, Plotly/FAISS 선택적, 시크릿 안전 로더
# ==========================================================

import os, io, re, json, textwrap, urllib.parse, difflib, random, time
from typing import List, Dict, Tuple, Optional

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

import requests
from bs4 import BeautifulSoup
try:
    import wikipedia
    try:
        wikipedia.set_lang("ko")
    except Exception:
        pass
except Exception:
    wikipedia = None

# ---------- Page config ----------
st.set_page_config(page_title="회사 특화 가상 면접 코치", page_icon="🎯", layout="wide")

# ---------- Secrets loader ----------
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

# ---------- Text utils ----------
def _clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()

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

# ---------- Company/domain utils ----------
VAL_KEYWORDS = ["핵심가치","가치","미션","비전","문화","원칙","철학","고객","데이터","혁신",
                "values","mission","vision","culture","principles","philosophy","customer","data","innovation"]

def _domain(u: str|None) -> str|None:
    if not u: return None
    try:
        if not u.startswith("http"): u = "https://" + u
        return urllib.parse.urlparse(u).netloc.lower().replace("www.","")
    except Exception:
        return None

def _name_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

# ---------- Wikipedia helpers (후보 노출/선택) ----------
def wiki_search_candidates(company_name: str, k: int = 8) -> list[str]:
    if wikipedia is None or not company_name.strip():
        return []
    try:
        return wikipedia.search(company_name.strip(), results=k) or []
    except Exception:
        return []

def fetch_wikipedia_summary_exact(title: str) -> dict|None:
    if wikipedia is None or not title: return None
    try:
        page = wikipedia.page(title, auto_suggest=False, redirect=True)
        first = _clean_text((page.summary or "").split("\n")[0])
        return {"company_name": page.title, "wiki_summary": first}
    except Exception:
        return None

# ---------- 네이버 Open API 래퍼 ----------
def _naver_api_get(api: str, params: dict, cid: str, csec: str):
    url = f"https://openapi.naver.com/v1/search/{api}.json"
    headers = {
        "X-Naver-Client-Id": cid,
        "X-Naver-Client-Secret": csec,
        "User-Agent": "Mozilla/5.0",
    }
    r = requests.get(url, headers=headers, params=params, timeout=8)
    if r.status_code != 200:
        return None
    return r.json()

def naver_search_news(query: str, display: int = 10, sort: str = "date") -> list[dict]:
    cid, csec = load_naver_keys()
    if not (cid and csec):  # 키 없으면 빈 리스트(폴백 사용)
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

# ---------- 홈페이지 스니펫 ----------
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
            r = requests.get(url, timeout=6, headers={"User-Agent":"Mozilla/5.0"})
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
                    if any(k.lower() in txt.lower() for k in VAL_KEYWORDS):
                        values_found.append(txt)
                    if any(k in txt for k in ["프로젝트","개발","출시","성과","project","launched","release","delivered","improved"]):
                        recent_found.append(txt)
        except Exception:
            continue

    if company_name_hint and site_name and _name_similarity(company_name_hint, site_name) < 0.35:
        values_found, recent_found = [], []  # 오탐 방지

    def dedup(lst):
        seen=set(); out=[]
        for x in lst:
            if x not in seen: seen.add(x); out.append(x)
        return out
    values_found = dedup(values_found)[:5]
    recent_found = dedup(recent_found)[:5]

    trimmed=[]
    for v in values_found:
        v2 = v.split(":",1)[-1]
        if len(v2)>60 and "," in v2:
            trimmed += [p.strip() for p in v2.split(",") if 2<=len(p.strip())<=24][:6]
        else:
            trimmed.append(v2[:60])

    return {"values": trimmed[:6], "recent": recent_found, "site_name": site_name, "about": about_para}

# ---------- 뉴스: 네이버 우선, 폴백 구글RSS ----------
def fetch_news(company_name: str, max_items: int = 6) -> list[dict]:
    news = naver_search_news(company_name, display=max_items, sort="date")
    if news:
        return news
    # fallback: Google News RSS
    q = urllib.parse.quote(company_name)
    url = f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"
    items = []
    try:
        r = requests.get(url, timeout=8)
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

# ---------- 채용 공고: 네이버 웹검색 우선, 폴백 DuckDuckGo ----------
SEARCH_ENGINES = ["https://duckduckgo.com/html/?q={query}"]
JOB_SITES = ["wanted.co.kr","saramin.co.kr","jobkorea.co.kr","rocketpunch.com",
             "indeed.com","linkedin.com","recruit.navercorp.com","kakao.recruit","naver"]

def discover_job_posting_urls(company_name: str, role: str, limit: int = 5) -> list[str]:
    urls = []
    if NAVER_ID and NAVER_SECRET:
        for dom in JOB_SITES:
            if len(urls) >= limit: break
            q = f"{company_name} {role} site:{dom}"
            links = naver_search_web(q, display=5, sort="date")
            for lk in links:
                if _domain(lk) and dom in _domain(lk) and lk not in urls:
                    urls.append(lk)
                if len(urls) >= limit: break
        if urls:
            return urls[:limit]
    # fallback: DuckDuckGo
    site_part = " OR ".join([f'site:{d}' for d in JOB_SITES])
    q = f'{company_name} {role} ({site_part})'
    for engine in SEARCH_ENGINES:
        url = engine.format(query=urllib.parse.quote(q))
        try:
            r = requests.get(url, timeout=8, headers={"User-Agent":"Mozilla/5.0"})
            if r.status_code != 200:
                continue
            soup = BeautifulSoup(r.text, "html")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                if href.startswith("/l/?kh=-1&uddg="):
                    href = urllib.parse.unquote(href.split("/l/?kh=-1&uddg=")[-1])
                dom = _domain(href)
                if not dom: continue
                if any(d in dom for d in JOB_SITES):
                    if href not in urls:
                        urls.append(href)
                if len(urls) >= limit:
                    break
        except Exception:
            continue
    return urls[:limit]

# ---------- JobPosting 파서 ----------
def _extract_json_ld_job(soup: BeautifulSoup) -> Optional[dict]:
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(s.string or "")
            seq = data if isinstance(data, list) else [data]
            for obj in seq:
                typ = obj.get("@type") if isinstance(obj, dict) else None
                if (isinstance(typ, list) and "JobPosting" in typ) or typ == "JobPosting":
                    return obj
        except Exception:
            continue
    return None

def parse_job_posting(url: str) -> dict:
    out = {"title": None, "responsibilities": [], "qualifications": [], "company_intro": None}
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code != 200 or "text/html" not in r.headers.get("content-type",""): return out
        soup = BeautifulSoup(r.text, "html.parser")

        jp = _extract_json_ld_job(soup)
        if jp:
            out["title"] = jp.get("title")
            desc = _clean_text(jp.get("description", ""))
            if desc:
                bullets = re.split(r"[•\-\n•·▪️▶︎]+", desc)
                bullets = [b.strip(" -•·▪️▶︎") for b in bullets if len(b.strip()) > 3]
                for b in bullets:
                    if any(k in b for k in ["자격","요건","requirements","qualification","필수","우대"]):
                        out["qualifications"].append(b)
                    else:
                        out["responsibilities"].append(b)

        # 헤더 휴리스틱
        sections = {}
        for h in soup.find_all(re.compile("^h[1-4]$")):
            head = _clean_text(h.get_text())
            if not head: continue
            nxt=[]; sib=h.find_next_sibling(); stop={"h1","h2","h3","h4"}
            while sib and sib.name not in stop:
                if sib.name in {"p","li","ul","ol","div"}:
                    txt=_clean_text(sib.get_text(" "))
                    if len(txt)>5: nxt.append(txt)
                sib=sib.find_next_sibling()
            if nxt: sections[head]=" ".join(nxt)

        def pick(keys):
            for k in sections:
                if any(kk.lower() in k.lower() for kk in keys): return sections[k]
            return None

        resp = pick(["주요 업무","담당 업무","업무","Responsibilities","What you will do","Role"])
        qual = pick(["자격 요건","지원 자격","우대","Requirements","Qualifications","Must have","Preferred"])
        if resp and not out["responsibilities"]:
            out["responsibilities"]=[x for x in re.split(r"[•\-\n•·▪️▶︎]+", resp) if len(x.strip())>3][:12]
        if qual and not out["qualifications"]:
            out["qualifications"]=[x for x in re.split(r"[•\-\n•·▪️▶︎]+", qual) if len(x.strip())>3][:12]

        meta_desc = soup.find("meta", {"name":"description"}) or soup.find("meta", {"property":"og:description"})
        if meta_desc and meta_desc.get("content"): out["company_intro"]=_snippetize(meta_desc["content"], 220)
    except Exception:
        pass

    out["responsibilities"]=[_snippetize(x,140) for x in out["responsibilities"]][:12]
    out["qualifications"]=[_snippetize(x,140) for x in out["qualifications"]][:12]
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

    _openai_ver = None; _httpx_ver = None
    try:
        import openai as _openai_pkg; _openai_ver = getattr(_openai_pkg, "__version__", None)
    except Exception: pass
    try:
        import httpx as _httpx_pkg; _httpx_ver = getattr(_httpx_pkg, "__version__", None)
    except Exception: pass
    with st.expander("디버그: 시크릿/버전 상태"):
        st.write({
            "api_key_provided": bool(API_KEY),
            "naver_keys": bool(NAVER_ID and NAVER_SECRET),
            "openai_version": _openai_ver,
            "httpx_version": _httpx_ver,
        })

if not API_KEY:
    st.error("OpenAI API Key가 필요합니다. (Cloud: Settings → Secrets)")
    st.stop()
try:
    client = OpenAI(api_key=API_KEY, timeout=30.0)
except Exception as e:
    st.error(f"OpenAI 초기화 오류: {e}"); st.stop()

# ==========================================================
# MAIN LAYOUT — ① 회사 입력 | ② 직무 입력  → ③ 요약 → ④ 질문 → ⑤ 답변
# ==========================================================
col_company, col_role = st.columns([1,1], vertical_alignment="start")

# ① 회사 입력 ------------------------------------------------
with col_company:
    st.subheader("① 회사 입력")
    company_name_input = st.text_input("회사 이름", placeholder="예: 네이버 / Kakao / 삼성SDS")
    homepage_input     = st.text_input("공식 홈페이지 URL(선택)", placeholder="https://...")
    job_url_input      = st.text_input("채용 공고 URL(선택)")

    # 회사 후보 검색(오인식 방지)
    if company_name_input.strip():
        cands = wiki_search_candidates(company_name_input, k=8)
        cands = ["(입력 그대로 사용)"] + cands
        chosen = st.selectbox("회사 후보(정확한 이름 선택)", options=cands, index=0, help="위키 후보에서 정확한 회사를 선택하면 오인식이 크게 줄어듭니다.")
    else:
        chosen = "(입력 그대로 사용)"

    if "company_state" not in st.session_state:
        st.session_state.company_state = {}

    if st.button("회사 정보 불러오기"):
        # 위키 고정
        wiki = None
        if chosen != "(입력 그대로 사용)":
            wiki = fetch_wikipedia_summary_exact(chosen)
        elif company_name_input.strip():
            wiki = {"company_name": company_name_input.strip(), "wiki_summary": None}

        # 사이트 크롤링
        site = fetch_site_snippets(homepage_input or None, (wiki or {}).get("company_name") or company_name_input)

        # 뉴스 (네이버 우선)
        news_items = fetch_news((wiki or {}).get("company_name") or company_name_input)

        st.session_state.company_state["base"] = {
            "company_name": site.get("site_name") or (wiki or {}).get("company_name") or company_name_input.strip(),
            "wiki_summary": (wiki or {}).get("wiki_summary"),
            "homepage": homepage_input or None,
            "values": site.get("values", []),
            "recent_projects": site.get("recent", []),
            "company_intro": site.get("about") or (wiki or {}).get("wiki_summary"),
            "news": news_items
        }
        st.success("회사 정보 갱신 완료")

# ② 직무 입력 ------------------------------------------------
with col_role:
    st.subheader("② 직무 입력")
    role_title = st.text_input("지원 직무명", placeholder="데이터 애널리스트 / ML 엔지니어 ...")
    st.caption("채용 공고 URL을 모르면 ‘회사/직무명’으로 네이버 웹검색을 통해 자동 탐색합니다.")
    if st.button("채용 공고 불러오기/검색"):
        jp_data = {"title": None,"responsibilities":[],"qualifications":[],"company_intro":None}
        urls = []
        if job_url_input.strip():
            urls = [job_url_input.strip()]
        else:
            urls = discover_job_posting_urls(st.session_state.company_state.get("base",{}).get("company_name","") or role_title,
                                             role_title or "", limit=4)
        if urls:
            jp_data = parse_job_posting(urls[0])
        st.session_state.company_state["job"] = {
            "role": role_title,
            "job_url": urls[0] if urls else (job_url_input.strip() or None),
            "role_requirements": jp_data["responsibilities"],
            "role_qualifications": jp_data["qualifications"],
            "company_intro": jp_data["company_intro"] or st.session_state.company_state.get("base",{}).get("company_intro")
        }
        st.success("채용 공고 반영 완료")

# company 객체 조립 ------------------------------------------
base = st.session_state.get("company_state",{}).get("base",{})
job  = st.session_state.get("company_state",{}).get("job",{})
company = {
    "company_name": base.get("company_name") or "(회사명 미설정)",
    "homepage": base.get("homepage"),
    "wiki_summary": base.get("wiki_summary"),
    "values": base.get("values", []),
    "recent_projects": base.get("recent_projects", []),
    "company_intro": job.get("company_intro") or base.get("company_intro"),
    "news": base.get("news", []),
    "role": job.get("role"),
    "role_requirements": job.get("role_requirements", []),
    "role_qualifications": job.get("role_qualifications", []),
    "job_url": job.get("job_url")
}

# ③ 회사 요약 (채용공고 기준) --------------------------------
st.subheader("③ 회사 요약 (채용공고 기준)")
def build_company_summary_for_ui(c: dict) -> dict:
    return {
        "회사명": c.get("company_name"),
        "간단 소개": c.get("company_intro") or c.get("wiki_summary"),
        "모집 분야": c.get("role"),
        "주요 업무(요약)": c.get("role_requirements")[:6],
        "자격 요건(요약)": c.get("role_qualifications")[:6],
        "핵심가치(추정)": c.get("values")[:6],
        "홈페이지": c.get("homepage"),
        "채용 공고": c.get("job_url"),
        "최근 뉴스": [ n.get("title") for n in c.get("news", [])[:5] ],
    }
st.json(build_company_summary_for_ui(company), expanded=True)

# ④ 질문 생성 ------------------------------------------------
st.subheader("④ 질문 생성")

def embed_texts(client: OpenAI, embed_model: str, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    resp = client.embeddings.create(model=embed_model, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

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

def build_ctx(c: dict) -> str:
    news = ", ".join([_snippetize(n["title"], 70) for n in c.get("news", [])[:3]])
    return textwrap.dedent(f"""
    [회사명] {c.get('company_name','')}
    [회사 소개] {c.get('company_intro') or c.get('wiki_summary') or ''}
    [모집 분야] {c.get('role','')}
    [주요 업무] {", ".join(c.get('role_requirements', [])[:6])}
    [자격 요건] {", ".join(c.get('role_qualifications', [])[:6])}
    [핵심가치] {", ".join(c.get('values', [])[:6])}
    [최근 이슈/뉴스] {news}
    """).strip()

def build_focuses(c: dict, supports: list[Tuple[str,float,str]], k: int = 4) -> list[str]:
    pool=[]
    if c.get("role"): pool.append(c["role"])
    pool += c.get("role_requirements", [])[:6]
    pool += c.get("role_qualifications", [])[:6]
    pool += c.get("values", [])[:6]
    pool += [ _snippetize(n['title'], 60) for n in c.get("news", [])[:4] ]
    for _,_,txt in (supports or [])[:3]:
        pool += [t.strip() for t in re.split(r"[•\-\n\.]", txt) if 6 < len(t.strip()) < 100][:3]
    pool=[p for p in pool if p]; random.shuffle(pool)
    return pool[:k]

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

if st.button("새 질문 받기", use_container_width=True):
    try:
        supports=[]
        if st.session_state.get("rag_on"):
            base_q = hint.strip() or f"{company.get('role','')} {' '.join(company.get('role_requirements', [])[:3])}"
            supports = retrieve_supports(base_q, st.session_state.get("topk",4))

        ctx = build_ctx(company)
        focuses = build_focuses(company, supports, k=4)
        rag_note = ""
        if supports:
            joined="\n".join([f"- ({s:.2f}) {txt[:200]}" for _,s,txt in supports[:3]])
            rag_note=f"\n[근거 발췌]\n{joined}"

        seed = int(time.time()*1000) % 2_147_483_647
        sys = f"""너는 '{company.get('company_name','')}'의 '{company.get('role','')}' 면접관이다.
회사/직무 컨텍스트와 채용공고(주요업무/자격요건), 최근 이슈/뉴스, (있다면) 근거 문서를 반영하여 **{q_type}** 유형({TYPE_INSTRUCTIONS[q_type]})의 질문 **6개 후보**를 한국어로 생성하라.
서로 **형태·관점·키워드**가 달라야 하며 난이도는 {level}.
아래 '포커스' 중 최소 1개 키워드를 문장에 **명시적으로 포함**하고, 지표/수치/기간/규모/리스크 요소를 적절히 섞어라.
포맷: 1) ... 2) ... 3) ... ... (한 줄씩)"""
        user = f"""[회사/직무 컨텍스트]\n{ctx}\n[포커스]\n- {chr(10).join(focuses)}{rag_note}\n[랜덤시드] {seed}"""
        resp = client.chat.completions.create(model=MODEL, temperature=0.95,
                                              messages=[{"role":"system","content":sys},{"role":"user","content":user}])
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

# ⑤ 나의 답변 / 코칭 -----------------------------------------
st.subheader("⑤ 나의 답변 / 코칭")
def coach_answer(company: dict, question: str, answer: str, supports: list[Tuple[str,float,str]]) -> dict:
    comp = ["문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치"]
    news = ", ".join([_snippetize(n["title"], 70) for n in company.get("news", [])[:3]])
    ctx = textwrap.dedent(f"""
    [회사명] {company.get('company_name','')}
    [회사 소개] {company.get('company_intro') or company.get('wiki_summary') or ''}
    [모집 분야] {company.get('role','')}
    [주요 업무] {", ".join(company.get('role_requirements', [])[:6])}
    [자격 요건] {", ".join(company.get('role_qualifications', [])[:6])}
    [핵심가치] {", ".join(company.get('values', [])[:6])}
    [최근 이슈/뉴스] {news}
    """).strip()
    rag_note=""
    if supports:
        joined="\n".join([f"- ({s:.3f}) {txt[:500]}" for (_,s,txt) in supports])
        rag_note=f"\n[회사 근거 문서 발췌]\n{joined}\n"
    sys = f"""너는 톱티어 면접 코치다. 한국어로 아래 형식에 맞춰 답하라:
1) 총점: 0~10 정수 1개
2) 강점: 2~3개 불릿
3) 리스크: 2~3개 불릿
4) 개선 포인트: 3개 불릿 (행동·지표·임팩트 중심)
5) 수정본 답변: STAR(상황-과제-행동-성과) 구조로 간결하고 자연스럽게
6) 역량 점수: [{", ".join(comp)}] 각각 0~5 정수 (한 줄에 쉼표로 구분)
채점 기준은 회사/직무 맥락, 채용공고(주요업무/자격요건), 질문 내 포커스/키워드 부합 여부를 포함한다.
추가 설명 금지. 형식 유지."""
    user = f"""[회사/직무 컨텍스트]\n{ctx}\n{rag_note}[면접 질문]\n{question}\n\n[후보자 답변]\n{answer}"""
    resp = client.chat.completions.create(model=MODEL, temperature=0.35,
                                          messages=[{"role":"system","content":sys},{"role":"user","content":user}])
    content = resp.choices[0].message.content.strip()
    m = re.search(r'([0-9]{1,2})\s*(?:/10|점|$)', content)
    score=None
    if m:
        try: score=max(0,min(10,int(m.group(1))))
        except: pass
    nums = re.findall(r'\b([0-5])\b', content.splitlines()[-1])
    comp_scores=[int(x) for x in nums[:5]] if len(nums)>=5 else None
    return {"raw": content, "score": score, "competencies": comp_scores}

if "history" not in st.session_state:
    st.session_state.history = []

ans = st.text_area("여기에 답변을 작성하세요 (STAR 권장: 상황-과제-행동-성과)", height=180)
if st.button("채점 & 코칭", type="primary", use_container_width=True):
    if not st.session_state.get("current_question"):
        st.warning("먼저 '새 질문 받기'로 질문을 생성하세요.")
    elif not ans.strip():
        st.warning("답변을 작성해 주세요.")
    else:
        with st.spinner("코칭 중..."):
            sups=[]
            if st.session_state.get("rag_on"):
                q_for_rag = st.session_state["current_question"] + "\n" + ans[:800]
                sups = retrieve_supports(q_for_rag, st.session_state.get("topk",4))
            res = coach_answer(company, st.session_state["current_question"], ans, sups)
            st.session_state.history.append({
                "ts": pd.Timestamp.now(),
                "question": st.session_state["current_question"],
                "user_answer": ans,
                "score": res.get("score"),
                "feedback": res.get("raw"),
                "supports": sups,
                "competencies": res.get("competencies")
            })

# 결과/레이더/CSV --------------------------------------------
st.divider()
st.subheader("피드백 결과")
if st.session_state.history:
    last = st.session_state.history[-1]
    c1,c2 = st.columns([1,3])
    with c1: st.metric("총점(/10)", last.get("score","—"))
    with c2: st.markdown(last.get("feedback",""))

    if st.session_state.get("rag_on") and last.get("supports"):
        with st.expander("코칭에 사용된 근거 보기"):
            for i,(_,sc,txt) in enumerate(last["supports"],1):
                st.markdown(f"**[{i}] sim={sc:.3f}**\n\n{txt[:800]}{'...' if len(txt)>800 else ''}")
                st.markdown("---")
else:
    st.info("아직 결과가 없습니다.")

st.divider()
st.subheader("역량 레이더 (세션 누적)")
competencies = ["문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치"]
def comp_df(hist):
    rows=[h["competencies"] for h in hist if h.get("competencies") and len(h["competencies"])==5]
    return pd.DataFrame(rows, columns=competencies) if rows else None
cdf = comp_df(st.session_state.history)
if cdf is not None:
    avg = cdf.mean().values.tolist()
    if PLOTLY_OK:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=avg+[avg[0]], theta=competencies+[competencies[0]], fill='toself'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,5])), showlegend=False, height=420)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(pd.DataFrame({"score": avg}, index=competencies))
    st.dataframe(cdf, use_container_width=True)
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
        if comps and len(comps)==5:
            for k,v in zip(competencies, comps): row[f"comp_{k}"]=v
        sups=h.get("supports") or []
        row["supports_preview"]=" || ".join([s[2][:120].replace("\n"," ") for s in sups])
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["timestamp","question","user_answer","score","feedback_raw","supports_preview"])
rep = build_report(st.session_state.history)
st.download_button("CSV 다운로드", data=rep.to_csv(index=False).encode("utf-8-sig"),
                   file_name="interview_session_report.csv", mime="text/csv")

st.caption("Tip) 네이버 키(NAVER_CLIENT_ID/SECRET)를 넣으면 뉴스·웹 탐색 정확도가 크게 향상됩니다. 위키 후보에서 정확한 회사를 선택하고, 가능하면 공고 URL을 입력하세요.")
