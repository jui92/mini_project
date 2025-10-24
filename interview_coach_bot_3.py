# -*- coding: utf-8 -*-
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

# ---------- Page config ----------
st.set_page_config(page_title="회사 특화 가상 면접 코치", page_icon="🤖", layout="wide")

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

# ---------- Domain / helpers ----------
VAL_KEYWORDS = ["핵심가치","가치","미션","비전","문화","원칙","철학","고객","데이터","혁신",
                "values","mission","vision","culture","principles","philosophy","customer","data","innovation"]

def _name_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

def _domain(u: str|None) -> str|None:
    if not u: return None
    try:
        if not u.startswith("http"): u = "https://" + u
        return urllib.parse.urlparse(u).netloc.lower().replace("www.","")
    except Exception:
        return None

# ---------------- NAVER Open API (선택) ----------------
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

# --- NEW: WebBaseLoader 전체 본문 덤프 ---------------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def webbase_dump_all_text(url: str) -> str:
    """
    해당 URL을 WebBaseLoader로 로드해서 페이지의 텍스트를 '전부' 합쳐 반환.
    - 여러 문서로 쪼개져 오면 page_content를 \n\n로 이어 붙임
    - HTML 태그 제거는 WebBaseLoader가 처리
    """
    if not WEBBASE_OK or not url:
        return ""
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        full = "\n\n".join([getattr(d, "page_content", "") for d in docs if getattr(d, "page_content", "")])
        # UI 렌더 보호를 위해 너무 길면 잘라서 표시하고, 다운로드는 전체 제공해도 됨
        return full
    except Exception as e:
        return f"[WebBaseLoader 에러] {e}"

# ---------- 사이트 크롤링 (About/Values 추정) ----------
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

# ---------- 홈페이지에서 커리어/채용 링크 자동 탐색 ----------
CAREER_HINTS = ["careers", "career", "jobs", "job", "recruit", "recruiting", "join", "hire", "hiring",
                "채용", "인재", "입사지원", "채용공고", "인재영입", "사람", "커리어"]

def discover_job_from_homepage(homepage: str, limit: int = 5) -> list[str]:
    if not homepage: return []
    try:
        if not homepage.startswith("http"): homepage = "https://" + homepage
        r = requests.get(homepage, timeout=8, headers={"User-Agent":"Mozilla/5.0"})
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

# ---------- 뉴스: 네이버 우선, 폴백 구글RSS ----------
def fetch_news(company_name: str, max_items: int = 6) -> list[dict]:
    news = naver_search_news(company_name, display=max_items, sort="date")
    if news:
        return news
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

# ---------- 채용 공고 후보 검색 ----------
SEARCH_ENGINES = ["https://duckduckgo.com/html/?q={query}"]
JOB_SITES = ["wanted.co.kr","saramin.co.kr","jobkorea.co.kr","rocketpunch.com",
             "indeed.com","linkedin.com","recruit.navercorp.com","kakao.recruit","naver"]

def discover_job_posting_urls(company_name: str, role: str, homepage: str|None, limit: int = 5) -> list[str]:
    urls = []
    urls += discover_job_from_homepage(homepage, limit=limit) if homepage else []
    if urls:
        return urls[:limit]

    if NAVER_ID and NAVER_SECRET:
        for dom in JOB_SITES:
            if len(urls) >= limit: break
            q = f"{company_name} {role} site:{dom}" if role else f"{company_name} 채용 site:{dom}"
            links = naver_search_web(q, display=5, sort="date")
            for lk in links:
                if _domain(lk) and dom in _domain(lk) and lk not in urls:
                    urls.append(lk)
                if len(urls) >= limit: break
        if urls:
            return urls[:limit]

    site_part = " OR ".join([f'site:{d}' for d in JOB_SITES])
    q = f'{company_name} {role} ({site_part})' if role else f'{company_name} 채용 ({site_part})'
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

# ---------- 공고 원문 → 요약 섹션(주요업무/자격/우대) ----------
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}

def fetch_page_text(url: str, max_chars: int = 16000) -> str:
    try:
        r = requests.get(url, timeout=12, headers=UA)
        if r.status_code != 200 or "text/html" not in r.headers.get("content-type",""): return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for s in soup(["script","style","noscript"]): s.decompose()
        text = _clean_text(soup.get_text(" "))
        return text[:max_chars]
    except Exception:
        return ""

def _extract_json_block(text: str) -> Optional[str]:
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.S)
    if m: return m.group(1)
    m = re.search(r"\{.*\}", text, re.S)
    if m: return m.group(0)
    return None

def llm_summarize_job_sections(client: OpenAI, model: str, raw_text: str) -> dict:
    """공고 원문에서 responsibilities/qualifications/preferences 요약을 JSON으로 받는다."""
    if not raw_text.strip():
        return {"responsibilities":[], "qualifications":[], "preferences":[]}
    sys = (
        "너는 채용 공고 전문을 분석해 아래 세 섹션을 한국어 불릿으로 요약한다. "
        "각 항목은 최대 12개, 문장은 간결하게. "
        "반드시 JSON만 출력: "
        '{"responsibilities":[], "qualifications":[], "preferences":[]}'
    )
    user = f"다음은 채용공고 원문이다. 요약 결과만 JSON으로 반환하라.\n\n<<<원문>>>\n{raw_text}\n<<<끝>>>"
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.2, max_tokens=800,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}]
        )
        content = resp.choices[0].message.content.strip()
        jb = _extract_json_block(content) or ""
        if jb:
            data = json.loads(jb)
            def _norm(arr): return [_clean_text(x)[:300] for x in (arr or []) if _clean_text(x)]
            return {
                "responsibilities": _norm(data.get("responsibilities", []))[:12],
                "qualifications":   _norm(data.get("qualifications", []))[:12],
                "preferences":      _norm(data.get("preferences", []))[:12],
            }
    except Exception:
        pass
    return {"responsibilities":[], "qualifications":[], "preferences":[]}

def _first_detail_from_list(url: str, role_hint: str = "") -> Optional[str]:
    try:
        r = requests.get(url, timeout=10, headers=UA)
        if r.status_code != 200 or "text/html" not in r.headers.get("content-type",""): return None
        soup = BeautifulSoup(r.text, "html.parser")
        dom = _domain(url) or ""
        if "wanted.co.kr" in dom:
            for a in soup.select("a[href*='/wd/']"):
                href = urllib.parse.urljoin(url, a.get("href"))
                title = (a.get_text() or "").strip()
                if (not role_hint) or (role_hint in title):
                    return href
        if "saramin.co.kr" in dom:
            for a in soup.select("a[href*='view?idx=']"):
                return urllib.parse.urljoin(url, a.get("href"))
        if "jobkorea.co.kr" in dom:
            for a in soup.select("a[href*='/Recruit/GI_Read/']"):
                return urllib.parse.urljoin(url, a.get("href"))
        for a in soup.find_all("a", href=True):
            href = urllib.parse.urljoin(url, a["href"])
            if re.search(r"/(wd|view|read|detail|posting|jobs?)/", href, re.I):
                return href
    except Exception:
        return None
    return None

def ensure_detail_url(u: str, role_hint: str) -> str:
    if re.search(r"/(wd|view|read|detail|posting|jobs?)/", u, re.I): return u
    deep = _first_detail_from_list(u, role_hint)
    return deep or u

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
# ① 회사/직무 입력 (통합)
# ==========================================================
st.subheader("① 회사/직무 입력")
company_name_input = st.text_input("회사 이름", placeholder="예: 네이버 / Kakao / 삼성SDS")
role_title         = st.text_input("지원 직무명", placeholder="데이터 애널리스트 / ML 엔지니어 ...")
job_url_input      = st.text_input("채용 공고 URL(선택) — 없다면 자동 탐색")
homepage_input     = st.text_input("공식 홈페이지 URL(선택)", placeholder="https://...")

if "company_state" not in st.session_state:
    st.session_state.company_state = {}
if "answer_text" not in st.session_state:
    st.session_state.answer_text = ""   # 사용자가 쓰는 답변 상태

# ---------------- 회사 요약/공고 요약 핵심 함수 ----------------
def build_company_obj(name: str, homepage: str|None, role: str|None, job_url: str|None) -> dict:
    site = fetch_site_snippets(homepage or None, name)

    # 후보 URL 구성
    candidates = []
    if job_url:
        candidates.append(job_url)
    else:
        try:
            candidates += discover_job_posting_urls(name, role or "", homepage, limit=5)
        except Exception:
            pass

    parsed = {"responsibilities":[], "qualifications":[], "preferences":[]}
    used_url = None
    for u in candidates:
        if not u: continue
        detail = ensure_detail_url(u, role or "")
        raw = fetch_page_text(detail, max_chars=16000)
        if not raw: continue
        p = llm_summarize_job_sections(client, MODEL, raw)
        if any([p["responsibilities"], p["qualifications"], p["preferences"]]):
            parsed = p
            used_url = detail
            break

    news_items = fetch_news(name, max_items=6)

    return {
        "company_name": name.strip() or "(회사명 미설정)",
        "homepage": homepage or None,
        "company_intro_site": site.get("about"),
        "values": site.get("values", []),
        "role": role or "",
        "role_requirements": parsed["responsibilities"],
        "role_qualifications": parsed["qualifications"],
        "preferences": parsed["preferences"],
        "job_url": used_url or (candidates[0] if candidates else (job_url or None)),
        "news": news_items,
    }

def generate_company_summary(c: dict) -> str:
    """회사 소개만 요약, 채용 섹션은 요약 불릿(없으면 최소 보강) 세로형 렌더"""
    ctx_src = textwrap.dedent(f"""
    [원자료(요약 힌트)]
    - 회사명: {c.get('company_name','')}
    - 홈페이지 소개문(있으면): {c.get('company_intro_site') or ''}
    - 핵심가치/키워드: {', '.join(c.get('values', [])[:6])}
    - 최근 이슈/뉴스: {', '.join([_snippetize(n['title'],70) for n in c.get('news', [])[:4]])}
    - 모집 분야(직무명): {c.get('role','')}
    - 공고 요약(이미 계산됨): 
      * 주요업무: {', '.join(c.get('role_requirements', [])[:8])}
      * 자격요건: {', '.join(c.get('role_qualifications', [])[:8])}
      * 우대사항: {', '.join(c.get('preferences', [])[:8])}
    """).strip()

    # 회사 소개만 간결 요약(JSON 강제)
    sys = (
        "너는 채용담당자다. 입력 힌트에서 '회사 소개'만 2~3문장으로 간결히 요약해라. "
        "JSON으로만 반환: {\"company_name\":\"\", \"about\":\"\"}"
    )
    user = ctx_src
    about = _clean_text(c.get("company_intro_site") or "")
    cname = c.get("company_name","")
    try:
        resp = client.chat.completions.create(
            model=MODEL, temperature=0.2,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}]
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r'\{.*\}', raw, re.S)
        if m:
            d = json.loads(m.group(0))
            cname = d.get("company_name") or cname
            about = _clean_text(d.get("about") or about)
    except Exception:
        pass

    # 최소 보강(채용 섹션이 전부 비면 더미 1~2개)
    resp_items = c.get("role_requirements", [])[:]
    qual_items = c.get("role_qualifications", [])[:]
    pref_items = c.get("preferences", [])[:]
    if not resp_items:
        resp_items = [f"{c.get('role') or '해당 직무'}의 핵심 과업 수행", "협업 부서와 목표/지표 정렬 및 실행"]
    if not qual_items:
        qual_items = ["관련 분야 실무 경험 또는 동급 역량", "문제정의/지표 설계 역량"]
    # pref는 없어도 OK

    # 세로형 렌더
    md = []
    md.append("**회사명**  \n" + (cname or c.get("company_name","")))
    md.append("\n**간단한 회사 소개(요약)**  \n" + (about or "회사 소개 요약 정보가 제한적입니다."))
    if c.get("job_url"):
        md.append(f"\n[채용 공고 열기]({c['job_url']})")

    def bullets(title, items):
        md.append(f"\n### {title}")
        if items:
            md.extend([f"- {it}" for it in items])
        else:
            md.append(f"요약 가능한 {title.replace('(요약)','')}이 없습니다.")

    bullets("주요업무", resp_items)
    bullets("자격요건", qual_items)
    bullets("우대사항", pref_items)

    return "\n".join(md)

# 빨간색(Primary) 버튼 — 회사/직무 정보 불러오기
if st.button("회사/직무 정보 불러오기", type="primary"):
    if not company_name_input.strip():
        st.warning("회사 이름을 입력해 주세요.")
    else:
        with st.spinner("회사/직무/공고/뉴스를 수집 중..."):
            cobj = build_company_obj(company_name_input, homepage_input or None, role_title or None, job_url_input or None)
            st.session_state.company_state["company"] = cobj
            # 요약을 즉시 저장 (기존 UI 렌더와 호환)
            st.session_state.company_state["summary_md"] = generate_company_summary(cobj)
        st.success("회사 정보 갱신 완료")

company = st.session_state.get("company_state",{}).get("company", {
    "company_name": "(회사명 미설정)", "homepage": None, "values": [], "recent_projects": [],
    "company_intro_site": None, "role": "", "role_requirements": [], "role_qualifications": [],
    "preferences": [], "job_url": None, "news": []
})
summary_md = st.session_state.get("company_state",{}).get("summary_md", None)

# ==========================================================
# ② 회사 요약 (LLM 생성)
# ==========================================================
st.subheader("② 회사 요약 / 채용 요건")
if summary_md:
    st.markdown(summary_md)
else:
    st.info("위의 입력을 완료하고 ‘회사/직무 정보 불러오기’를 눌러 요약을 생성하세요.")
# --- NEW: 원문 보기(WebBaseLoader) -------------------------------------
with st.expander("원문 보기 (WebBaseLoader로 페이지 텍스트 그대로 보기)"):
    if not WEBBASE_OK:
        st.info("langchain-community 패키지가 필요합니다. requirements.txt에 `langchain-community>=0.2.0` 추가 후 배포하세요.")
    else:
        # 기본은 현재 감지된 채용 공고 URL 사용 (없으면 빈칸)
        default_url = company.get("job_url") or ""
        raw_url = st.text_input("대상 URL", value=default_url, placeholder="https://... (공고 상세 URL 권장)")
        fetch_btn = st.button("원문 불러오기", use_container_width=True, key="btn_dump_webbase")
        if fetch_btn:
            if not raw_url:
                st.warning("URL을 입력하세요.")
            else:
                with st.spinner("WebBaseLoader로 원문 수집 중..."):
                    fulltext = webbase_dump_all_text(raw_url)
                if not fulltext:
                    st.error("텍스트를 가져오지 못했습니다. URL을 확인하거나 로그인/차단이 없는지 확인하세요.")
                else:
                    st.success("원문을 불러왔습니다.")
                    # 화면 지연을 줄이고 싶으면 아래 value를 fulltext[:50000]처럼 자르세요.
                    st.text_area("페이지 텍스트(전부)", value=fulltext, height=420)
                    st.download_button(
                        "원문 텍스트 다운로드",
                        data=fulltext.encode("utf-8"),
                        file_name="job_posting_raw.txt",
                        mime="text/plain"
                    )


# ==========================================================
# ③ 질문 생성 (RAG 옵션 포함)
# ==========================================================
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
    [회사 소개] {c.get('company_intro_site') or ''}
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

st.subheader("③ 질문 생성")
q_type = st.selectbox("질문 유형", list(TYPE_INSTRUCTIONS.keys()))
level  = st.selectbox("난이도/연차", ["주니어","미들","시니어"])
hint   = st.text_input("질문 생성 힌트(선택)", placeholder="예: 전환 퍼널 / 모델 성능-비용 / 데이터 품질")

if "history" not in st.session_state:
    st.session_state.history = []
if "current_question" not in st.session_state:
    st.session_state.current_question = ""

# 새 질문 받기 (Primary) — 클릭 시 답변 입력 초기화
if st.button("새 질문 받기", use_container_width=True, type="primary"):
    st.session_state.answer_text = ""   # 입력칸 초기화
    try:
        supports=[]
        if st.session_state.get("rag_on"):
            base_q = hint.strip() or f"{company.get('role','')} {' '.join(company.get('role_requirements', [])[:3])}"
            supports = retrieve_supports(base_q, st.session_state.get("topk",4))

        ctx = build_ctx(company)
        focuses = build_focuses(company, supports, k=4)
        rag_note = ""
        if supports:
            joined="\n".join([f"- ({s:.2f}) {txt[:200]}" for _,s,txt in supports[:3]] )
            rag_note=f"\n[근거 발췌]\n{joined}"

        seed = int(time.time()*1000) % 2_147_483_647
        sys = f"""너는 '{company.get('company_name','')}'의 '{company.get('role','')}' 면접관이다.
회사/직무 컨텍스트와 채용공고(주요업무/자격요건), 최근 이슈/뉴스, (있다면) 근거 문서를 반영하여 **{q_type}** 유형({TYPE_INSTRUCTIONS[q_type]})의 질문 **6개 후보**를 한국어로 생성하라.
서로 **형태·관점·키워드**가 달라야 하며 난이도는 {level}.
아래 '포커스' 중 최소 1개 키워드를 문장에 **명시적으로 포함**하고, 지표/수치/기간/규모/리스크 요소를 적절히 섞어라.
포맷: 1) ... 2) ... 3) ... ... (한 줄씩)"""
        user = f"""[회사/직무 컨텍스트]\n{ctx}\n[포커스]\n- {chr(10).join(focuses)}{rag_note}\n[랜덤시드] {seed}"""

        resp = client.chat.completions.create(
            model=MODEL,
            temperature=0.95,
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
# ④ 나의 답변 / 코칭 (100점제)
# ==========================================================
st.subheader("④ 나의 답변 / 코칭")

def coach_answer(company: dict, question: str, answer: str, supports: list[Tuple[str,float,str]]) -> dict:
    comp = ["문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치"]
    news = ", ".join([_snippetize(n["title"], 70) for n in company.get("news", [])[:3]])
    ctx = textwrap.dedent(f"""
    [회사명] {company.get('company_name','')}
    [회사 소개] {company.get('company_intro_site') or ''}
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
1) 총점: 0~100 정수 1개
2) 강점: 2~3개 불릿
3) 리스크: 2~3개 불릿
4) 개선 포인트: 3개 불릿 (행동·지표·임팩트 중심)
5) 수정본 답변: STAR(상황-과제-행동-성과) 구조로 간결하고 자연스럽게
6) 역량 점수(각 0~20 정수): [문제정의, 데이터/지표, 실행력/주도성, 협업/커뮤니케이션, 고객가치] — 한 줄에 숫자 5개만 쉼표로 구분해 출력
추가로, 각 항목별 감점/아쉬움/개선 힌트를 한 줄 주석처럼 적어라. (예: 문제정의: 원인-가설 연결 부족)
형식/숫자 범위 엄수. 다른 설명 금지."""
    user = f"""[회사/직무 컨텍스트]\n{ctx}\n{rag_note}[면접 질문]\n{question}\n\n[후보자 답변]\n{answer}"""
    resp = client.chat.completions.create(model=MODEL, temperature=0.35,
                                          messages=[{"role":"system","content":sys},{"role":"user","content":user}])
    content = resp.choices[0].message.content.strip()

    # ----- 총점 0~100 -----
    score = None
    m = re.search(r'(\d{1,3})\s*(?:/100|점|$)', content)
    if m: score = int(m.group(1))
    if score is None:
        m10 = re.search(r'(\d{1,2})\s*/\s*10', content)
        if m10: score = max(0, min(100, int(m10.group(1)) * 10))
    if score is None:
        m_any = re.search(r'\b(\d{1,3})\b', content)
        if m_any: score = max(0, min(100, int(m_any.group(1))))
    if score is not None:
        score = max(0, min(100, score))

    # ----- 역량 5개(0~20) -----
    # 마지막 줄 우선 파싱
    lastline = content.splitlines()[-1]
    nums = re.findall(r'\b(\d{1,2})\b', lastline)
    if len(nums) < 5:
        nums = re.findall(r'\b(\d{1,2})\b', content)
    comp_scores = None
    if len(nums) >= 5:
        cand = [int(x) for x in nums[:5]]
        if all(0 <= x <= 5 for x in cand):
            cand = [x * 4 for x in cand]
        if all(0 <= x <= 10 for x in cand) and any(x > 5 for x in cand):
            cand = [x * 2 for x in cand]
        comp_scores = [max(0, min(20, x)) for x in cand]

    return {"raw": content, "score": score, "competencies": comp_scores}

# 답변 입력칸
ans = st.text_area("여기에 답변을 작성하세요 (STAR 권장: 상황-과제-행동-성과)", height=180, key="answer_text")

# 채점 버튼
if st.button("채점 & 코칭", type="primary", use_container_width=True):
    if not st.session_state.get("current_question"):
        st.warning("먼저 '새 질문 받기'로 질문을 생성하세요.")
    elif not st.session_state.answer_text.strip():
        st.warning("답변을 작성해 주세요.")
    else:
        with st.spinner("코칭 중..."):
            sups=[]
            if st.session_state.get("rag_on"):
                q_for_rag = st.session_state["current_question"] + "\n" + st.session_state.answer_text[:800]
                sups = retrieve_supports(q_for_rag, st.session_state.get("topk",4))
            res = coach_answer(company, st.session_state["current_question"], st.session_state.answer_text, sups)
            st.session_state.history.append({
                "ts": pd.Timestamp.now(),
                "question": st.session_state["current_question"],
                "user_answer": st.session_state.answer_text,
                "score": res.get("score"),
                "feedback": res.get("raw"),
                "supports": sups,
                "competencies": res.get("competencies")
            })

# ---------------- 결과/레이더/CSV ----------------
st.divider()
st.subheader("피드백 결과")
if st.session_state.history:
    last = st.session_state.history[-1]
    left, right = st.columns([1,3])
    with left:  # 좌측 총점
        st.metric("총점(/100)", last.get("score","—"))
    with right: # 우측 원문 피드백 (총점 문구 포함)
        st.markdown(last.get("feedback",""))

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
    # 각 레코드 합계 열
    cdf["합계"] = cdf.sum(axis=1)
    # 세션 평균(레이더용)
    avg = cdf[competencies].mean().values.tolist()

    if PLOTLY_OK:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=avg+[avg[0]], theta=competencies+[competencies[0]], fill='toself', name="세션 평균(0~20)"))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,20])), showlegend=False, height=420)
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
            row["comp_sum"] = sum(comps)
        sups=h.get("supports") or []
        row["supports_preview"]=" || ".join([s[2][:120].replace("\n"," ") for s in sups])
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["timestamp","question","user_answer","score","feedback_raw","supports_preview"])
rep = build_report(st.session_state.history)
st.download_button("CSV 다운로드", data=rep.to_csv(index=False).encode("utf-8-sig"),
                   file_name="interview_session_report.csv", mime="text/csv")

st.caption("Tip) 홈페이지/공고 URL이 있으면 공고 원문을 요약해 주요업무/자격/우대가 더 정확하게 채워집니다. 없으면 커리어 링크→국내 포털 순으로 자동 탐색합니다.")
