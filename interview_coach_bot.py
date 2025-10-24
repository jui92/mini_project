# -*- coding: utf-8 -*-
import os, io, re, json, textwrap, urllib.parse, difflib, random, time, hashlib
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ---------------- Optional deps ----------------
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

import requests
from bs4 import BeautifulSoup

# ---------------- Page config ----------------
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

def _name_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

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

def filehash(bytes_or_str: bytes|str) -> str:
    b = bytes_or_str if isinstance(bytes_or_str, bytes) else bytes(bytes_or_str, "utf-8", errors="ignore")
    return hashlib.md5(b).hexdigest()

# =========================================================
# NAVER Open API
# =========================================================
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

# =========================================================
# Site / Careers crawling (light)
# =========================================================
VAL_KEYWORDS = ["value","values","mission","vision","culture","고객","가치","문화","원칙","철학","혁신"]

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_site_intro(base_url: str|None) -> dict:
    """
    회사 소개만 요약하기 위한 가벼운 문장 후보 수집 (about/intro).
    """
    if not base_url:
        return {"site_name": None, "about_candidates": []}
    url0 = base_url.strip()
    if not url0.startswith("http"): url0 = "https://" + url0

    cand_paths = ["", "/", "/about", "/company", "/about-us"]
    site_name = None
    about_candidates = []

    for path in cand_paths:
        url = url0.rstrip("/") + path
        try:
            r = requests.get(url, timeout=6, headers={"User-Agent":"Mozilla/5.0"})
            if r.status_code != 200 or "text/html" not in r.headers.get("content-type",""):
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            if site_name is None and soup.title and soup.title.string:
                site_name = _clean_text(soup.title.string.split("|")[0])

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
    return {"site_name":site_name, "about_candidates":outs[:8]}

CAREER_HINTS = ["careers","recruit","jobs","career","채용","인재영입","recruitment","join"]
@st.cache_data(show_spinner=False, ttl=3600)
def discover_job_from_homepage(homepage: str, limit: int = 5) -> list[str]:
    if not homepage: return []
    try:
        if not homepage.startswith("http"): homepage = "https://" + homepage
        r = requests.get(homepage, timeout=8, headers={"User-Agent":"Mozilla/5.0"})
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

# =========================================================
# Job posting parsing + LLM 분류 요약 폴백
# =========================================================
JOB_SITES = ["wanted.co.kr","saramin.co.kr","jobkorea.co.kr","rocketpunch.com","linkedin.com","indeed.com"]
@st.cache_data(show_spinner=False, ttl=3600)
def discover_job_posting_urls(company: str, role: str, homepage: str|None, limit: int=5) -> list[str]:
    urls=[]
    if homepage:
        urls += discover_job_from_homepage(homepage, limit=limit)
    if urls: return urls[:limit]

    # NAVER site: 검색
    if NAVER_ID and NAVER_SECRET:
        for dom in JOB_SITES:
            if len(urls)>=limit: break
            q = f"{company} {role} site:{dom}" if role else f"{company} 채용 site:{dom}"
            links = naver_search_web(q, display=5, sort="date")
            for lk in links:
                if _domain(lk) and dom in _domain(lk) and lk not in urls:
                    urls.append(lk)
                if len(urls)>=limit: break
    return urls[:limit]

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

@st.cache_data(show_spinner=False, ttl=900)
def fetch_page_text(url: str) -> str:
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code != 200 or "text/html" not in r.headers.get("content-type",""):
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        # 가시 텍스트만
        for s in soup(["script","style","noscript"]): s.decompose()
        txt = _clean_text(soup.get_text(separator="\n"))
        return txt
    except Exception:
        return ""

def parse_job_posting_structured(url: str) -> dict:
    """
    구조화(원문) 우선 파싱. 실패 시 빈 목록.
    """
    out = {"responsibilities":[], "qualifications":[], "preferences":[]}
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code != 200 or "text/html" not in r.headers.get("content-type",""): return out
        soup = BeautifulSoup(r.text, "html.parser")

        # JSON-LD
        jp = _extract_json_ld_job(soup)
        if jp:
            desc = _clean_text(jp.get("description",""))
            if desc:
                bullets = [x.strip(" -•·▪️▶︎") for x in re.split(r"[•\-\n•·▪️▶︎]+", desc) if len(x.strip())>3]
                # 라벨을 알 수 없으므로 responsibilities에 우선 담고, 아래 섹션 파싱으로 보완
                out["responsibilities"] += bullets[:12]

        # 섹션 추출
        sections={}
        for h in soup.find_all(re.compile("^h[1-4]$")):
            head=_clean_text(h.get_text())
            if not head: continue
            nxt=[]; sib=h.find_next_sibling(); stop={"h1","h2","h3","h4"}
            while sib and sib.name not in stop:
                if sib.name in {"p","li","ul","ol","div"}:
                    t=_clean_text(sib.get_text(" "))
                    if len(t)>5: nxt.append(t)
                sib=sib.find_next_sibling()
            if nxt: sections[head]="\n".join(nxt)

        def explode(txt):
            return [x.strip() for x in re.split(r"[•\-\n•·▪️▶︎]+", txt) if len(x.strip())>2][:12]

        # 한국어/영어 키
        for k,v in sections.items():
            lk=k.lower()
            if any(s in lk for s in ["주요 업무","담당 업무","업무","responsibilities","what you will do","role"]):
                out["responsibilities"] += explode(v)
            if any(s in lk for s in ["자격 요건","지원 자격","requirements","qualifications","must have"]):
                out["qualifications"] += explode(v)
            if any(s in lk for s in ["우대","prefer","nice to have","preferred"]):
                out["preferences"] += explode(v)

        # dedup
        for key in out:
            seen=set(); arr=[]
            for t in out[key]:
                if t not in seen:
                    seen.add(t); arr.append(_snippetize(t, 150))
            out[key]=arr[:12]
    except Exception:
        pass
    return out

# ---------------- LLM helpers ----------------
EVAL_FACTORS = [
    "문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치"
]
EVAL_SCHEMA_HINT = {
  "type":"object",
  "properties":{
    "overall": {"type":"integer", "minimum":0, "maximum":100},
    "factors": {
      "type":"object",
      "properties": {k: {
        "type":"object",
        "properties":{
          "score":{"type":"integer","minimum":0,"maximum":20},
          "comment":{"type":"string"},
          "deduct":{"type":"string"},
          "improve":{"type":"string"}
        },
        "required":["score"]
      } for k in EVAL_FACTORS},
      "additionalProperties": False
    },
    "strengths":{"type":"array","items":{"type":"string"}},
    "risks":{"type":"array","items":{"type":"string"}},
    "improvements":{"type":"array","items":{"type":"string"}},
    "revised":{"type":"string"}
  },
  "required":["overall","factors","revised"]
}

def get_openai_client() -> OpenAI:
    api_key = st.session_state.get("API_KEY")
    return OpenAI(api_key=api_key, timeout=30.0)

def embed_texts(client: OpenAI, embed_model: str, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    resp = client.embeddings.create(model=embed_model, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

def cosine_topk(matrix: np.ndarray, query: np.ndarray, k: int = 4):
    if matrix.size == 0:
        return np.array([]), np.array([], dtype=int)
    qn = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
    mn = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    sims = mn @ qn.T
    sims = sims.reshape(-1)
    idx = np.argsort(-sims)[:k]
    return sims[idx], idx

# =========================================================
# Sidebar
# =========================================================
with st.sidebar:
    st.title("⚙️ 설정")
    API_KEY = load_api_key_from_env_or_secrets()
    if not API_KEY:
        st.info("환경변수/Secrets에서 키를 못 찾았습니다. 아래에 입력 후 엔터.")
        API_KEY = st.text_input("OPENAI_API_KEY", type="password")
    st.session_state.API_KEY = API_KEY

    MODEL = st.selectbox("챗 모델", ["gpt-4o-mini","gpt-4o","gpt-4.1-mini"], index=0)
    EMBED_MODEL = st.selectbox("임베딩 모델", ["text-embedding-3-small","text-embedding-3-large"], index=0)

    with st.expander("디버그: 시크릿/버전 상태"):
        _openai_ver = None; _httpx_ver = None
        try:
            import openai as _openai_pkg; _openai_ver = getattr(_openai_pkg, "__version__", None)
        except Exception: pass
        try:
            import httpx as _httpx_pkg; _httpx_ver = getattr(_httpx_pkg, "__version__", None)
        except Exception: pass
        st.write({
            "api_key_provided": bool(API_KEY),
            "naver_keys": bool(NAVER_ID and NAVER_SECRET),
            "openai_version": _openai_ver,
            "httpx_version": _httpx_ver,
        })

if not OpenAI or not API_KEY:
    st.error("OpenAI API Key가 필요합니다. (Cloud: Settings → Secrets)")
    st.stop()

# =========================================================
# ① 회사/직무 입력
# =========================================================
st.subheader("① 회사/직무 입력")
colA, colB = st.columns(2)
with colA:
    company_name_input = st.text_input("회사 이름", placeholder="예: 네이버 / Kakao / 삼성SDS")
with colB:
    role_title = st.text_input("지원 직무명", placeholder="데이터 엔지니어 / ML 엔지니어 ...")

job_url_input  = st.text_input("채용 공고 URL(선택) — 없으면 자동 탐색/요약 폴백")
homepage_input = st.text_input("공식 홈페이지 URL(선택)", placeholder="https://...")

# 세션 state 준비
for key, default in [
    ("company", None),
    ("company_summary", None),
    ("rag_store", {"chunks":[], "embeds": None}),
    ("history", []),
    ("answer_text", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# =========================================================
# 회사 컨텍스트 구성 (소개는 요약 / 채용 3요소는 원문 우선, 실패시 LLM 요약 분류)
# =========================================================
def llm_summarize_intro(candidates: list[str], company: str) -> str:
    """
    후보 문장들을 2~3문장으로 요약.
    """
    if not candidates:
        return ""
    client = get_openai_client()
    sys = "너는 채용 담당자다. 회사 소개 문장 후보를 2~3문장으로 간결하게 요약하라. 과장/광고 문구는 제거."
    user = "회사명: {}\n\n후보 문장:\n- {}".format(company, "\n- ".join([_snippetize(t, 400) for t in candidates[:8]]))
    resp = client.chat.completions.create(
        model=MODEL, temperature=0.2,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}]
    )
    return resp.choices[0].message.content.strip()

def llm_split_job_text_to_sections(raw_text: str) -> dict:
    """
    원문에서 '주요업무/자격요건/우대사항'을 불릿으로 분류하는 JSON 출력 강제.
    """
    client = get_openai_client()
    schema = {
        "type":"object",
        "properties":{
            "responsibilities":{"type":"array","items":{"type":"string"}},
            "qualifications":{"type":"array","items":{"type":"string"}},
            "preferences":{"type":"array","items":{"type":"string"}}
        },
        "required":["responsibilities","qualifications","preferences"]
    }
    sys = (
        "입력 원문에서 '주요업무','자격요건','우대사항'을 각각 3~8개의 한국어 불릿으로 분류해 JSON만 출력."
        "모호하면 빈 배열([]) 유지, 추측 금지. 불릿은 간결한 명사형/문장형으로."
        f"스키마: {json.dumps(schema, ensure_ascii=False)}"
    )
    resp = client.chat.completions.create(
        model=MODEL, temperature=0.2,
        messages=[{"role":"system","content":sys},{"role":"user","content":raw_text[:6000]}],
        response_format={"type":"json_object"}
    )
    data = json.loads(resp.choices[0].message.content)
    # 길이 제한
    for k in data:
        data[k] = [ _snippetize(x, 140) for x in data[k] ][:12]
    return data

def build_company_context(name: str, homepage: str|None, role: str|None, job_url: str|None) -> dict:
    """
    1) 회사 소개(요약) : 홈페이지 intro 후보 → 요약
    2) 채용 3요소(원문 우선, 실패 시 요약 분류)
    3) 최신 뉴스(네이버)
    """
    # 소개 요약
    site_info = fetch_site_intro(homepage or "")
    intro = llm_summarize_intro(site_info.get("about_candidates", []), name) if site_info.get("about_candidates") else ""

    # 채용 3요소
    responsibilities, qualifications, preferences = [], [], []

    discovered = [job_url] if job_url else discover_job_posting_urls(name, role or "", homepage, limit=3)
    raw_text = ""
    if discovered:
        # 원문 구조 파싱
        parsed = parse_job_posting_structured(discovered[0])
        responsibilities = parsed["responsibilities"]
        qualifications = parsed["qualifications"]
        preferences = parsed["preferences"]
        # 부족하면 요약 분류 폴백
        if (not qualifications) or (not preferences):
            raw_text = fetch_page_text(discovered[0])
    else:
        # 공고 URL 없으면 홈페이지/포털 텍스트로 폴백 요약
        # (홈페이지 텍스트 + 간단 검색)
        texts = []
        if homepage:
            texts.append(fetch_page_text(homepage))
        links = []
        if NAVER_ID and NAVER_SECRET:
            for dom in JOB_SITES:
                links += naver_search_web(f"{name} 채용 site:{dom}", display=3, sort="date")
        for lk in links[:2]:
            texts.append(fetch_page_text(lk))
        raw_text = "\n\n".join([t for t in texts if t])

    if raw_text and (not qualifications or not preferences):
        try:
            sections = llm_split_job_text_to_sections(raw_text)
            # 이미 원문으로 가져온 항목이 있으면 합치되 중복 제거
            def merge(a,b):
                seen=set(a); out=a[:]
                for x in b:
                    if x not in seen: seen.add(x); out.append(x)
                return out[:12]
            responsibilities = merge(responsibilities, sections.get("responsibilities",[]))
            qualifications = merge(qualifications, sections.get("qualifications",[]))
            preferences = merge(preferences, sections.get("preferences",[]))
        except Exception:
            pass

    news_items = naver_search_news(name, display=6, sort="date")

    return {
        "company_name": name.strip() or "(회사명 미설정)",
        "homepage": homepage or None,
        "role": role or "",
        "company_intro": intro,
        "job_url": discovered[0] if discovered else (job_url or None),
        "responsibilities": responsibilities,
        "qualifications": qualifications,
        "preferences": preferences,
        "news": news_items
    }

# =========================================================
# 버튼: 회사/직무 정보 불러오기 (Primary)
# =========================================================
if st.button("회사/직무 정보 불러오기", type="primary"):
    if not company_name_input.strip():
        st.warning("회사 이름을 입력해 주세요.")
    else:
        with st.spinner("회사/채용/뉴스 수집 및 요약 중..."):
            st.session_state.company = build_company_context(
                company_name_input, homepage_input or None, role_title or None, job_url_input or None
            )
        # 회사 변경 시 아래 실행결과 초기화
        st.session_state.history = []
        st.session_state.answer_text = ""
        st.success("회사 정보 갱신 및 실행결과 초기화 완료!")

company = st.session_state.get("company")

# =========================================================
# ② 회사 요약 / 채용 요건 (세로형)
# =========================================================
st.subheader("② 회사 요약 / 채용 요건")

if company:
    st.markdown(f"**회사명**  \n{company['company_name']}")
    intro = company.get("company_intro") or "회사 소개를 요약할 수 있는 정보가 충분하지 않습니다."
    st.markdown(f"**간단한 회사 소개(요약)**  \n{intro}")

    link_cols = st.columns(2)
    with link_cols[0]:
        if company.get("job_url"): st.link_button("채용 공고 열기", company["job_url"])
    with link_cols[1]:
        if company.get("homepage"): st.link_button("홈페이지 열기", company["homepage"])

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    def vlist(col, title, items):
        with col:
            st.markdown(f"### {title}(요약)")
            if items:
                st.markdown("\n".join([f"- {x}" for x in items]))
            else:
                st.caption("요약 가능한 항목이 없습니다.")

    vlist(col1, "주요업무", company.get("responsibilities", []))
    vlist(col2, "자격요건", company.get("qualifications", []))
    vlist(col3, "우대사항", company.get("preferences", []))

    with st.expander("디버그: 공고 요약 상태"):
        st.json({
            "job_url": company.get("job_url"),
            "resp_cnt": len(company.get("responsibilities",[])),
            "qual_cnt": len(company.get("qualifications",[])),
            "pref_cnt": len(company.get("preferences",[]))
        })
else:
    st.info("위의 입력을 완료하고 ‘회사/직무 정보 불러오기’를 눌러 요약을 생성하세요.")

# =========================================================
# ③ 질문 생성
# =========================================================
st.subheader("③ 질문 생성")

TYPE_INSTRUCTIONS = {
    "행동(STAR)": "S(상황)-T(과제)-A(행동)-R(성과)를 유도하는 실무 사례 질문",
    "기술 심층": "핵심 기술적 의사결정·트레이드오프·성능/비용/품질 지표를 파고드는 심층 질문",
    "핵심가치 적합성": "핵심가치와 태도를 검증하는 상황형 질문",
    "역질문": "지원자가 회사를 평가할 수 있도록 통찰력 있는 역질문"
}
q_type = st.selectbox("질문 유형", list(TYPE_INSTRUCTIONS.keys()))
level  = st.selectbox("난이도/연차", ["주니어","미들","시니어"])
hint   = st.text_input("질문 생성 힌트(선택)", placeholder="예: 전환 퍼널 / 모델 성능-비용 / 데이터 품질")

def build_ctx_for_q(c: dict) -> str:
    if not c: return ""
    news = ", ".join([_snippetize(n["title"], 70) for n in c.get("news", [])[:3]])
    return textwrap.dedent(f"""
    [회사명] {c.get('company_name','')}
    [모집 분야] {c.get('role','')}
    [주요 업무] {", ".join(c.get('responsibilities', [])[:6])}
    [자격 요건] {", ".join(c.get('qualifications', [])[:6])}
    [우대 사항] {", ".join(c.get('preferences', [])[:4])}
    [최근 이슈/뉴스] {news}
    """).strip()

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

if "current_question" not in st.session_state:
    st.session_state.current_question = ""

# Primary 버튼 + 클릭 시 답변칸 초기화
if st.button("새 질문 받기", use_container_width=True, type="primary"):
    st.session_state.answer_text = ""
    try:
        client = get_openai_client()
        ctx = build_ctx_for_q(company or {})
        seed = int(time.time()*1000) % 2_147_483_647
        sys = f"""너는 '{company.get('company_name','') if company else '해당 회사'}'의 면접관이다.
컨텍스트/채용 3요소/최근 이슈를 반영하여 **{q_type}** 유형({TYPE_INSTRUCTIONS[q_type]})의 질문 **6개**를 한국어로 생성하라.
서로 형태·관점·키워드가 달라야 하며 난이도는 {level}. 
포맷: 1) ... 2) ... 3) ... (한 줄씩)"""
        user = f"""[컨텍스트]\n{ctx}\n\n[힌트]\n{hint}\n[랜덤시드] {seed}"""
        resp = client.chat.completions.create(
            model=MODEL, temperature=0.9,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}]
        )
        raw = resp.choices[0].message.content.strip()
        cands = [re.sub(r'^\s*\d+\)\s*','',line).strip() for line in raw.splitlines() if re.match(r'^\s*\d+\)', line)]
        if not cands:
            cands = [l.strip("- ").strip() for l in raw.splitlines() if len(l.strip())>0][:6]
        hist_qs = [h["question"] for h in st.session_state.history][-12:]
        st.session_state.current_question = pick_diverse(cands, hist_qs) or (cands[0] if cands else "질문 생성 실패")
    except Exception as e:
        st.error(f"질문 생성 오류: {e}")

st.text_area("질문", height=110, value=st.session_state.get("current_question",""))

# =========================================================
# ④ 나의 답변 / 채점 & 코칭 (JSON 고정, 총점=합산)
# =========================================================
st.subheader("④ 나의 답변 / 코칭")
ans = st.text_area("여기에 답변을 작성하세요 (STAR 권장: 상황-과제-행동-성과)", height=200, key="answer_text")

def evaluate_answer(company: dict, question: str, answer: str) -> dict:
    client = get_openai_client()
    news = ", ".join([_snippetize(n["title"], 70) for n in (company or {}).get("news", [])[:3]])
    ctx = textwrap.dedent(f"""
    [회사명] {(company or {}).get('company_name','')}
    [모집 분야] {(company or {}).get('role','')}
    [주요 업무] {", ".join((company or {}).get('responsibilities', [])[:6])}
    [자격 요건] {", ".join((company or {}).get('qualifications', [])[:6])}
    [우대 사항] {", ".join((company or {}).get('preferences', [])[:4])}
    [최근 이슈/뉴스] {news}
    """).strip()

    sys = (
        "너는 톱티어 면접 코치다. 아래 스키마에 맞춘 **한국어 JSON만** 출력하라.\n"
        f"스키마: {json.dumps(EVAL_SCHEMA_HINT, ensure_ascii=False)}\n"
        "설명/추가 텍스트 금지. 각 기준(score 0~20)은 질문과 답변/회사 맥락/채용 3요소 부합 여부로 채점하라."
        "각 기준에 대해 comment(짧은 칭찬/핵심요지), deduct(감점요인), improve(개선 포인트)를 간단히 채워라."
    )
    user = f"""[회사/직무 컨텍스트]\n{ctx}\n\n[면접 질문]\n{question}\n\n[후보자 답변]\n{answer}"""
    resp = client.chat.completions.create(
        model=MODEL, temperature=0.3,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}],
        response_format={"type":"json_object"}
    )
    data = json.loads(resp.choices[0].message.content)

    # 총점 일원화: 합산
    factors = data.get("factors", {})
    sum_score = sum(int(factors[k]["score"]) for k in EVAL_FACTORS if k in factors and isinstance(factors[k].get("score"), int))
    data["sum_score"] = max(0, min(100, sum_score))  # 5*20 = 100
    return data

if st.button("채점 & 코칭", type="primary", use_container_width=True):
    if not st.session_state.get("current_question"):
        st.warning("먼저 '새 질문 받기'로 질문을 생성하세요.")
    elif not st.session_state.answer_text.strip():
        st.warning("답변을 작성해 주세요.")
    else:
        with st.spinner("채점/코칭 중..."):
            data = evaluate_answer(company or {}, st.session_state["current_question"], st.session_state.answer_text)

            # 히스토리 저장 (누적에 사용)
            row = {k: (data["factors"].get(k,{}).get("score") if data.get("factors") else None) for k in EVAL_FACTORS}
            st.session_state.history.append({
                "ts": pd.Timestamp.now(),
                "question": st.session_state["current_question"],
                "answer": st.session_state.answer_text,
                "sum_score": data.get("sum_score"),
                "factors": row,
                "comments": {k: data["factors"].get(k,{}).get("comment") for k in EVAL_FACTORS} if data.get("factors") else {},
                "deducts": {k: data["factors"].get(k,{}).get("deduct") for k in EVAL_FACTORS} if data.get("factors") else {},
                "improves": {k: data["factors"].get(k,{}).get("improve") for k in EVAL_FACTORS} if data.get("factors") else {},
                "strengths": data.get("strengths",[]),
                "risks": data.get("risks",[]),
                "improvements": data.get("improvements",[]),
                "revised": data.get("revised",""),
                "raw": data
            })

# =========================================================
# 결과 렌더링 (총점=합산으로 일원화 / 표+수정본)
# =========================================================
st.divider()
st.subheader("피드백 결과")

if st.session_state.history:
    last = st.session_state.history[-1]
    # 좌측/우측 총점 동일 (sum_score)
    c1,c2 = st.columns([1,3])
    with c1:
        st.metric("총점(/100)", last.get("sum_score","—"))
    with c2:
        st.markdown(f"**총점: {last.get('sum_score','—')}/100**")
        # 기준별 근거(점수/감점/개선)
        st.markdown("**2. 기준별 근거(점수/감점/개선):**")
        table_rows=[]
        for k in EVAL_FACTORS:
            sc = last["factors"].get(k)
            comment = (last["comments"] or {}).get(k,"") if isinstance(last.get("comments"), dict) else ""
            deduct  = (last["deducts"] or {}).get(k,"") if isinstance(last.get("deducts"), dict) else ""
            improve = (last["improves"] or {}).get(k,"") if isinstance(last.get("improves"), dict) else ""
            table_rows.append((f"{k}({sc if sc is not None else '-'}/20)", f"강점: {comment or '-'} / 감점: {deduct or '-'} / 개선: {improve or '-'}"))
        df = pd.DataFrame(table_rows, columns=["기준(점수)","코멘트"])
        st.dataframe(df, use_container_width=True, hide_index=True)

        # 강점/리스크/개선 포인트(모델 제공)
        if last.get("strengths"):
            st.markdown("**3. 강점:**\n" + "\n".join([f"- {x}" for x in last["strengths"]]))
        if last.get("risks"):
            st.markdown("**4. 리스크:**\n" + "\n".join([f"- {x}" for x in last["risks"]]))
        if last.get("improvements"):
            st.markdown("**5. 개선 포인트:**\n" + "\n".join([f"- {x}" for x in last["improvements"]]))

        # 수정본 답변
        if last.get("revised"):
            st.markdown("**6. 수정본 답변:**")
            st.markdown(last["revised"])

else:
    st.caption("아직 채점 결과가 없습니다.")

# =========================================================
# ⑥ 역량 레이더 (최근 vs 세션 평균) + 누적 테이블(합계 포함)
# =========================================================
st.divider()
st.subheader("역량 레이더 (세션 누적, NA는 0으로 표시)")

def history_df(hist):
    """
    최근 점수 / 세션 평균 DataFrame
    """
    if not hist: return None, None
    rows=[]
    for h in hist:
        rows.append([h["factors"].get(k) for k in EVAL_FACTORS])
    df = pd.DataFrame(rows, columns=EVAL_FACTORS)

    # 최근
    latest = df.iloc[-1].copy()
    # 평균 (NaN 제외)
    avg = df.astype("float").mean(skipna=True)
    return latest, avg

latest, avg = history_df(st.session_state.history)

if latest is not None:
    # 레이더 데이터(NA는 0으로 대체해 시각화만)
    r_latest = [float(x) if pd.notna(x) else 0.0 for x in latest.tolist()]
    r_avg = [float(x) if pd.notna(x) else 0.0 for x in avg.tolist()]

    if PLOTLY_OK:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=r_latest+[r_latest[0]], theta=EVAL_FACTORS+[EVAL_FACTORS[0]],
            fill='toself', name="최근"
        ))
        fig.add_trace(go.Scatterpolar(
            r=r_avg+[r_avg[0]], theta=EVAL_FACTORS+[EVAL_FACTORS[0]],
            fill='toself', name="세션 평균", opacity=0.35
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,20])), showlegend=True, height=450)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(pd.DataFrame({"최근": r_latest, "세션평균": r_avg}, index=EVAL_FACTORS))

    # 누적 테이블 (최근 행 + 합계 열)
    table = pd.DataFrame([latest.tolist()], columns=EVAL_FACTORS)
    table["합계(0~100)"] = table[EVAL_FACTORS].sum(axis=1, numeric_only=True)
    st.dataframe(table, use_container_width=True)
    st.caption("표의 각 축은 최신 결과의 점수(NA는 '-')입니다. 위 레이더/아래 표에는 합계(0~100)와 세션 누적 평균을 보여줍니다.")
else:
    st.caption("아직 역량 점수가 파싱된 코칭 결과가 없습니다.")

# =========================================================
# ⑦ 세션 리포트 (CSV)
# =========================================================
st.divider()
st.subheader("세션 리포트 (CSV)")

def build_report(hist):
    rows=[]
    for h in hist:
        row={"timestamp":h.get("ts"),"question":h.get("question"),"answer":h.get("answer"),
             "sum_score":h.get("sum_score")}
        # factors
        for k in EVAL_FACTORS:
            row[f"score_{k}"] = (h.get("factors") or {}).get(k)
            row[f"comment_{k}"] = (h.get("comments") or {}).get(k)
            row[f"deduct_{k}"] = (h.get("deducts") or {}).get(k)
            row[f"improve_{k}"] = (h.get("improves") or {}).get(k)
        row["revised"] = h.get("revised","")
        rows.append(row)
    cols = ["timestamp","question","answer","sum_score"] + \
           [f"score_{k}" for k in EVAL_FACTORS] + \
           [f"comment_{k}" for k in EVAL_FACTORS] + \
           [f"deduct_{k}" for k in EVAL_FACTORS] + \
           [f"improve_{k}" for k in EVAL_FACTORS] + ["revised"]
    return pd.DataFrame(rows)[cols] if rows else pd.DataFrame(columns=cols)

rep = build_report(st.session_state.history)
st.download_button("CSV 다운로드", data=rep.to_csv(index=False).encode("utf-8-sig"),
                   file_name="interview_session_report.csv", mime="text/csv")

st.caption("Tip) 공고 URL이 없더라도 홈페이지/포털 텍스트로 요약 분류 폴백을 사용합니다. 회사 변경 시 결과는 초기화됩니다.")
