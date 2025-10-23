# -*- coding: utf-8 -*-
# =====================================================================
# 회사 특화 가상 면접 코치 (KR 최적화)
# - 회사 소개만 LLM 요약 / 업무·자격·우대는 원문 그대로 노출
# - 회사 변경시 하단 결과 초기화
# - 100점제 채점(질문 유형별 루브릭 적용, 비적용 항목은 '-')
# - 총점 표시 일관화(좌·우 동일)
# - 레이더 표에 '합계' 컬럼 추가, 계산 오류 수정
# =====================================================================

import os, io, re, json, textwrap, urllib.parse, difflib, random, time, functools
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
except ImportError:
    st.error("`openai` 패키지가 필요합니다. requirements.txt에 openai를 추가했는지 확인하세요.")
    st.stop()

import requests
from bs4 import BeautifulSoup

# ---------------- Page config ----------------
st.set_page_config(page_title="회사 특화 가상 면접 코치", page_icon="🎯", layout="wide")

# ---------------- Secrets/Keys ----------------
def _secrets_file_exists() -> bool:
    paths = [
        os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml"),
        os.path.join(os.getcwd(), ".streamlit", "secrets.toml"),
    ]
    return any(os.path.exists(p) for p in paths)

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

# ---------------- Utils ----------------
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

# ---------------- Simple cache wrapper ----------------
@functools.lru_cache(maxsize=256)
def _cached_get(url: str, timeout: int = 8) -> Optional[str]:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code == 200 and "text/html" in r.headers.get("content-type",""):
            return r.text
    except Exception:
        pass
    return None

# ---------------- NAVER OPEN API ----------------
def _naver_api_get(api: str, params: dict, cid: str, csec: str):
    url = f"https://openapi.naver.com/v1/search/{api}.json"
    headers = {
        "X-Naver-Client-Id": cid,
        "X-Naver-Client-Secret": csec,
        "User-Agent": "Mozilla/5.0",
    }
    try:
        r = requests.get(url, headers=headers, params=params, timeout=8)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def naver_search_news(query: str, display: int = 6, sort: str = "date") -> list[dict]:
    cid, csec = load_naver_keys()
    if not (cid and csec): return []
    js = _naver_api_get("news", {"query": query, "display": display, "sort": sort}, cid, csec)
    if not js: return []
    out=[]
    for it in js.get("items", []):
        title = _clean_text(re.sub(r"</?b>|&quot;|&apos;|&amp;|&lt;|&gt;", "", it.get("title","")))
        out.append({"title": title, "link": it.get("link"), "pubDate": it.get("pubDate")})
    return out

def naver_search_web(query: str, display: int = 5, sort: str = "date") -> list[str]:
    cid, csec = load_naver_keys()
    if not (cid and csec): return []
    js = _naver_api_get("webkr", {"query": query, "display": display, "sort": sort}, cid, csec)
    if not js: return []
    links=[]
    for it in js.get("items", []):
        link = it.get("link")
        if link and link not in links: links.append(link)
    return links

# ---------------- Crawl: site snippets ----------------
def fetch_site_snippets(base_url: str | None, company_name_hint: str | None = None) -> dict:
    if not base_url:
        return {"values": [], "recent": [], "site_name": None, "about": None}
    url0 = base_url.strip()
    if not url0.startswith("http"): url0 = "https://" + url0
    cand_paths = ["", "/", "/about", "/company", "/about-us", "/mission", "/values", "/culture"]
    values_found, recent_found = [], []
    site_name, about_para = None, None
    for path in cand_paths[:4]:  # 속도: 경로 수 제한
        url = url0.rstrip("/") + path
        html = _cached_get(url, timeout=6)
        if not html: continue
        soup = BeautifulSoup(html, "html.parser")
        if site_name is None:
            og = soup.find("meta", {"property":"og:site_name"}) or soup.find("meta", {"name":"application-name"})
            if og and og.get("content"): site_name = _clean_text(og["content"])
            elif soup.title and soup.title.string: site_name = _clean_text(soup.title.string.split("|")[0])
        if about_para is None:
            hero = soup.find(["p","div"], class_=re.compile(r"(lead|hero|intro)", re.I))
            if hero: about_para = _snippetize(hero.get_text(" "))
        for tag in soup.find_all(["h1","h2","h3","p","li"]):
            txt = _clean_text(tag.get_text(separator=" "))
            if 10 <= len(txt) <= 220:
                if any(k.lower() in txt.lower() for k in VAL_KEYWORDS):
                    values_found.append(txt)
                if any(k in txt for k in ["프로젝트","개발","출시","성과","project","launched","release","delivered","improved"]):
                    recent_found.append(txt)
    if company_name_hint and site_name and _name_similarity(company_name_hint, site_name) < 0.35:
        values_found, recent_found = [], []
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

# ---------------- Discover job/career links ----------------
CAREER_HINTS = ["careers","career","jobs","job","recruit","recruiting","join","hire","hiring","채용","인재","입사지원","채용공고","인재영입","커리어"]

def discover_job_from_homepage(homepage: str, limit: int = 4) -> list[str]:
    if not homepage: return []
    try:
        if not homepage.startswith("http"): homepage = "https://" + homepage
        html = _cached_get(homepage, timeout=8)
        if not html: return []
        soup = BeautifulSoup(html, "html.parser")
        links=[]
        for path in ["careers","recruit","jobs","career","채용","인재영입","recruitment","join"]:
            links.append(urllib.parse.urljoin(homepage.rstrip("/") + "/", path))
        for a in soup.find_all("a", href=True):
            href = a["href"]; text = (a.get_text() or "").lower()
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

# ---------------- News fetch ----------------
def fetch_news(company_name: str, max_items: int = 6) -> list[dict]:
    news = naver_search_news(company_name, display=max_items, sort="date")
    if news: return news
    # fallback: Google News RSS
    q = urllib.parse.quote(company_name)
    url = f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"
    items=[]
    try:
        r = requests.get(url, timeout=8); 
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

# ---------------- Job posting parser ----------------
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
    """
    반환: title, responsibilities(원문 리스트), qualifications(원문 리스트), preferred(원문 리스트), company_intro
    - '원문 그대로'를 최대한 보존
    """
    out = {"title": None, "responsibilities": [], "qualifications": [], "preferred": [], "company_intro": None}
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code != 200 or "text/html" not in r.headers.get("content-type",""): return out
        soup = BeautifulSoup(r.text, "html.parser")

        # JSON-LD 우선
        jp = _extract_json_ld_job(soup)
        if jp:
            out["title"] = jp.get("title")
            desc = _clean_text(jp.get("description", ""))
            if desc:
                bullets = re.split(r"[\n\r]+|[•·▪️▶︎\-]\s+", desc)
                bullets = [b.strip(" -•·▪️▶︎") for b in bullets if len(b.strip()) > 3]
                for b in bullets:
                    low = b.lower()
                    if any(k in low for k in ["preferred","우대","nice to have"]):
                        out["preferred"].append(b)
                    elif any(k in low for k in ["requirement","자격","요건","qualification","필수"]):
                        out["qualifications"].append(b)
                    else:
                        out["responsibilities"].append(b)

        # 헤더/섹션 휴리스틱
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
            if nxt: sections[head]="\n".join(nxt)

        def pick(keys):
            for k in sections:
                if any(kk.lower() in k.lower() for kk in keys): return sections[k]
            return None

        resp = pick(["주요 업무","담당 업무","업무","Responsibilities","What you will do","Role"])
        qual = pick(["자격 요건","지원 자격","Requirements","Qualifications","Must have"])
        pref = pick(["우대","우대사항","Preferred","Nice to have"])

        def bullets_from(txt):
            if not txt: return []
            arr = re.split(r"[\n\r]+|[•·▪️▶︎\-]\s+", txt)
            return [a.strip(" -•·▪️▶︎") for a in arr if len(a.strip())>3][:20]

        if resp and not out["responsibilities"]: out["responsibilities"]=bullets_from(resp)
        if qual and not out["qualifications"]:   out["qualifications"]=bullets_from(qual)
        if pref and not out["preferred"]:        out["preferred"]=bullets_from(pref)

        meta_desc = soup.find("meta", {"name":"description"}) or soup.find("meta", {"property":"og:description"})
        if meta_desc and meta_desc.get("content"): out["company_intro"]=_snippetize(meta_desc["content"], 220)

        # 원문 보존(요약 금지) → 길이만 클램프
        out["responsibilities"]=[_snippetize(x,200) for x in out["responsibilities"]][:15]
        out["qualifications"]  =[_snippetize(x,200) for x in out["qualifications"]][:15]
        out["preferred"]       =[_snippetize(x,200) for x in out["preferred"]][:15]

    except Exception:
        pass
    return out

# ---------------- OpenAI setup ----------------
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
    st.error("OpenAI API Key가 필요합니다."); st.stop()
try:
    client = OpenAI(api_key=API_KEY, timeout=30.0)
except Exception as e:
    st.error(f"OpenAI 초기화 오류: {e}"); st.stop()

# =====================================================================
# ① 회사/직무 입력
# =====================================================================
st.subheader("① 회사/직무 입력")
company_name_input = st.text_input("회사 이름 (그대로 사용)", placeholder="예: 네이버 / Kakao / 삼성SDS")
homepage_input     = st.text_input("공식 홈페이지 URL(선택)", placeholder="https://...")
role_title         = st.text_input("지원 직무명", placeholder="데이터 애널리스트 / ML 엔지니어 ...")
job_url_input      = st.text_input("채용 공고 URL(선택) — 없다면 자동 탐색")

# 세션 초기값
for key, val in [
    ("company_state", {}),
    ("history", []),
    ("current_question", ""),
    ("answer_text", ""),
    ("rag_store", {"chunks": [], "embeds": None}),
]:
    if key not in st.session_state: st.session_state[key] = val

def discover_job_posting_urls(company_name: str, role: str, homepage: str|None, limit: int = 4) -> list[str]:
    urls=[]
    urls += discover_job_from_homepage(homepage, limit=limit) if homepage else []
    if urls: return urls[:limit]
    if NAVER_ID and NAVER_SECRET:
        JOB_SITES = ["wanted.co.kr","saramin.co.kr","jobkorea.co.kr","rocketpunch.com",
                     "indeed.com","linkedin.com","recruit.navercorp.com","kakao.recruit","naver"]
        for dom in JOB_SITES:
            if len(urls) >= limit: break
            q = f"{company_name} {role} site:{dom}" if role else f"{company_name} 채용 site:{dom}"
            links = naver_search_web(q, display=5, sort="date")
            for lk in links:
                if lk not in urls: urls.append(lk)
            if len(urls) >= limit: break
    if urls: return urls[:limit]
    # fallback duckduckgo
    engine = "https://duckduckgo.com/html/?q={query}"
    site_part = " OR ".join([f'site:{d}' for d in ["wanted.co.kr","saramin.co.kr","jobkorea.co.kr","rocketpunch.com"]])
    q = f'{company_name} {role} ({site_part})' if role else f'{company_name} 채용 ({site_part})'
    url = engine.format(query=urllib.parse.quote(q))
    html = _cached_get(url, timeout=8)
    if html:
        soup = BeautifulSoup(html, "html")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/l/?kh=-1&uddg="):
                href = urllib.parse.unquote(href.split("/l/?kh=-1&uddg=")[-1])
            if href not in urls: urls.append(href)
            if len(urls) >= limit: break
    return urls[:limit]

def build_company_obj(name: str, homepage: str|None, role: str|None, job_url: str|None) -> dict:
    site = fetch_site_snippets(homepage or None, name)
    discovered = [job_url] if job_url else discover_job_posting_urls(name, role or "", homepage, limit=3)
    jp = parse_job_posting(discovered[0]) if discovered else {"title":None,"responsibilities":[],"qualifications":[],"preferred":[],"company_intro":None}
    news_items = fetch_news(name, max_items=6)
    return {
        "company_name": name.strip() or "(회사명 미설정)",
        "homepage": homepage or None,
        "values": site.get("values", []),
        "recent_projects": site.get("recent", []),
        "company_intro_site": site.get("about"),
        "role": role or "",
        "job_url": discovered[0] if discovered else (job_url or None),
        # ---- 원문 그대로(요약 금지) ----
        "role_responsibilities": jp.get("responsibilities", []),
        "role_qualifications":   jp.get("qualifications", []),
        "role_preferred":        jp.get("preferred", []),
        "news": news_items
    }

def summarize_company_intro_only(c: dict) -> str:
    """회사 소개만 LLM 요약. 업무·자격·우대는 아래 섹션에서 원문 그대로 노출."""
    ctx = textwrap.dedent(f"""
    [홈페이지 소개(발췌)] {c.get('company_intro_site') or ''}
    [최근 이슈/뉴스 타이틀] {', '.join([_snippetize(n['title'],70) for n in c.get('news', [])[:3]])}
    """).strip()
    sys = ("너는 채용담당자다. 아래 정보를 바탕으로 '회사 소개'만 2~3문장 한국어 요약으로 작성하라. "
           "광고성 문구/형용사는 최소화하고, 사실 위주로 간결하게.")
    user = f"{ctx}\n\n[회사명] {c.get('company_name','')}"
    try:
        resp = client.chat.completions.create(
            model=MODEL, temperature=0.2,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}]
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return c.get("company_intro_site") or "회사 소개 정보가 충분하지 않습니다."

# ---- 회사/직무 정보 불러오기 ----
if st.button("회사/직무 정보 불러오기", type="primary"):
    if not company_name_input.strip():
        st.warning("회사 이름을 입력해 주세요.")
    else:
        with st.spinner("회사/공고/뉴스 수집 중..."):
            cobj = build_company_obj(company_name_input, homepage_input or None, role_title or None, job_url_input or None)
            intro_md = summarize_company_intro_only(cobj)
            # 상태 저장
            st.session_state.company_state = {"company": cobj, "intro_md": intro_md}
            # ---- 회사 변경 시 하단 초기화 ----
            st.session_state.current_question = ""
            st.session_state.answer_text = ""
            st.session_state.history = []
            st.session_state.rag_store = {"chunks": [], "embeds": None}
        st.success("회사 정보 갱신 및 결과 초기화 완료")

company = st.session_state.get("company_state",{}).get("company", None)
intro_md = st.session_state.get("company_state",{}).get("intro_md", None)

# =====================================================================
# ② 회사 요약 (소개만 요약) + 업무/자격/우대 원문
# =====================================================================
st.subheader("② 회사 요약 / 채용 요건")
if company and intro_md:
    st.markdown(f"**회사명**: {company.get('company_name')}")
    st.markdown("**회사 소개(요약)**")
    st.markdown(intro_md)
    if company.get("homepage"): st.link_button("홈페이지 열기", company["homepage"])
    if company.get("job_url"):  st.link_button("채용 공고 열기", company["job_url"])
    st.markdown("---")
    st.markdown(f"**모집 분야**: {company.get('role') or '—'}")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("**주요 업무(원문)**")
        if company["role_responsibilities"]:
            st.markdown("- " + "\n- ".join(company["role_responsibilities"]))
        else:
            st.caption("공고에서 추출된 주요 업무가 없습니다.")
    with c2:
        st.markdown("**자격 요건(원문)**")
        if company["role_qualifications"]:
            st.markdown("- " + "\n- ".join(company["role_qualifications"]))
        else:
            st.caption("공고에서 추출된 자격 요건이 없습니다.")
    with c3:
        st.markdown("**우대 사항(원문)**")
        if company["role_preferred"]:
            st.markdown("- " + "\n- ".join(company["role_preferred"]))
        else:
            st.caption("공고에서 추출된 우대 사항이 없습니다.")
else:
    st.info("위 입력 후 ‘회사/직무 정보 불러오기’를 눌러 요약을 생성하세요.")

# =====================================================================
# ③ 질문 생성
# =====================================================================
st.subheader("③ 질문 생성")

def embed_texts(client: OpenAI, embed_model: str, texts: list[str]) -> np.ndarray:
    if not texts: return np.zeros((0, 1536), dtype=np.float32)
    resp = client.embeddings.create(model=embed_model, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

with st.expander("RAG 옵션 (선택)"):
    rag_enabled = st.toggle("회사 문서 기반 질문/코칭 사용", value=True, key="rag_on")
    top_k = st.slider("검색 상위 K", 1, 8, 4, 1, key="topk")
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
                st.session_state.rag_store["chunks"] = chunks
                st.session_state.rag_store["embeds"] = embs
                st.success(f"청크 {len(chunks)}개 인덱싱 완료")

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
    if not st.session_state.get("rag_on") or embs is None or not chs: return []
    qv = embed_texts(client, "text-embedding-3-small", [qtext])
    scores, idxs = cosine_topk(embs, qv, k=k)
    return [("회사자료", float(s), chs[int(i)]) for s,i in zip(scores, idxs)]

TYPE_INSTRUCTIONS = {
    "행동(STAR)": "과거 실무 사례를 끌어내도록 S(상황)-T(과제)-A(행동)-R(성과)를 유도",
    "기술 심층": "핵심 기술적 의사결정·트레이드오프·성능/비용/품질 지표를 파고드는 심층 질문",
    "핵심가치 적합성": "핵심가치/태도 적합성을 상황기반으로 검증",
    "역질문": "지원자가 회사를 평가할 수 있도록 통찰력 있는 역질문",
}

def build_ctx(c: dict) -> str:
    news = ", ".join([_snippetize(n["title"], 70) for n in c.get("news", [])[:3]]) if c else ""
    return textwrap.dedent(f"""
    [회사명] {c.get('company_name','') if c else ''}
    [모집 분야] {c.get('role','') if c else ''}
    [주요 업무] {", ".join(c.get('role_responsibilities', [])[:6]) if c else ''}
    [자격 요건] {", ".join(c.get('role_qualifications', [])[:6]) if c else ''}
    [핵심가치] {", ".join(c.get('values', [])[:6]) if c else ''}
    [최근 뉴스] {news}
    """).strip()

def build_focuses(c: dict, supports: list[Tuple[str,float,str]], k: int = 4) -> list[str]:
    pool=[]
    if c:
        if c.get("role"): pool.append(c["role"])
        pool += c.get("role_responsibilities", [])[:6]
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

if st.button("새 질문 받기", use_container_width=True, type="primary"):
    st.session_state.answer_text = ""  # 이전 답변 초기화
    try:
        supports=[]
        if st.session_state.get("rag_on"):
            base_q = hint.strip() or (company.get('role','') if company else '')
            supports = retrieve_supports(base_q, st.session_state.get("topk",4))
        ctx = build_ctx(company) if company else ""
        focuses = build_focuses(company, supports, k=4)
        rag_note = ""
        if supports:
            joined="\n".join([f"- ({s:.2f}) {txt[:200]}" for _,s,txt in supports[:3]])
            rag_note=f"\n[근거 발췌]\n{joined}"
        seed = int(time.time()*1000) % 2_147_483_647
        sys = f"""너는 면접관이다. **{q_type}** 유형({TYPE_INSTRUCTIONS[q_type]})의 질문 **6개 후보**를 한국어로 생성하라.
서로 형태·관점·키워드가 달라야 하며 난이도는 {level}. '포커스' 키워드 중 최소 1개를 문장에 명시적으로 포함.
포맷: 1) ... 2) ... 3) ... (한 줄씩)"""
        user = f"""[회사/직무 컨텍스트]\n{ctx}\n[포커스]\n- {chr(10).join(focuses)}{rag_note}\n[랜덤시드] {seed}"""
        resp = client.chat.completions.create(
            model=MODEL, temperature=0.95,
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

# =====================================================================
# ④ 나의 답변 / 코칭 (질문 유형별 루브릭 적용)
# =====================================================================
st.subheader("④ 나의 답변 / 코칭")

# 루브릭 적용: 어떤 축을 점수화할지 결정
AXES = ["문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치"]
def applicable_axes(question_type: str) -> list[bool]:
    if question_type == "행동(STAR)":
        return [True, True, True, True, True]
    if question_type == "기술 심층":
        return [True, True, True, False, False]
    if question_type == "핵심가치 적합성":
        return [True, False, False, True, True]
    if question_type == "역질문":
        return [True, False, False, True, True]
    return [True, True, True, True, True]

def coach_answer(company: dict, question: str, answer: str, supports, qtype: str) -> dict:
    news = ", ".join([_snippetize(n["title"], 70) for n in (company.get("news", []) if company else [])[:3]])
    ctx = build_ctx(company) if company else ""
    # 적용 축 안내(비적용은 '-')
    applies = applicable_axes(qtype)
    apply_text = ", ".join([f"{AXES[i]}({'O' if applies[i] else '-'})" for i in range(5)])
    rag_note=""
    if supports:
        joined="\n".join([f"- ({s:.3f}) {txt[:300]}" for (_,s,txt) in supports])
        rag_note=f"\n[회사 근거 문서 발췌]\n{joined}\n"
    sys = f"""너는 톱티어 면접 코치다. 한국어로 아래 형식으로만 답하라.
1) 총점: 0~100 정수 1개  # 반드시 '총점: NN/100' 형식의 한 줄로 출력
2) 강점: 2~3개 불릿
3) 리스크: 2~3개 불릿
4) 개선 포인트: 3개 불릿 (행동·지표·임팩트 중심)
5) 수정본 답변: STAR(상황-과제-행동-성과) 구조로 간결하게
6) 역량 점수(각 0~20, 비적용은 '-'로만 출력): [{', '.join(AXES)}] — 이 순서로 숫자 5개 혹은 '-'를 쉼표로 구분해 한 줄에 출력
채점은 '{qtype}' 유형에 맞춰 비적용 항목은 반드시 '-'로 표기하라.
총점은 적용된 항목들의 평균(0~20)을 5배 하여 0~100으로 환산하라."""
    user = f"""[회사/직무 컨텍스트]\n{ctx}\n{rag_note}[질문 유형 적용 축]\n{apply_text}\n\n[면접 질문]\n{question}\n\n[후보자 답변]\n{answer}"""
    resp = client.chat.completions.create(model=MODEL, temperature=0.35,
                                          messages=[{"role":"system","content":sys},{"role":"user","content":user}])
    content = resp.choices[0].message.content.strip()

    # --- 총점 파싱(엄격): '총점:' 라인에서만 추출 ---
    score=None
    for line in content.splitlines():
        if "총점" in line:
            m = re.search(r'총점\s*:\s*(\d{1,3})\s*/\s*100', line)
            if m:
                score = max(0, min(100, int(m.group(1))))
            break
    # 보정: 없으면 역량 점수에서 계산
    comps_raw_line = content.splitlines()[-1]
    tokens = [t.strip() for t in re.split(r"[,\s]+", comps_raw_line) if t.strip()!=""]
    comps: list[Optional[int]] = []
    for i, tok in enumerate(tokens[:5]):
        if tok == "-" or tok == "–":
            comps.append(None)
        elif tok.isdigit():
            comps.append(max(0, min(20, int(tok))))
        else:
            comps.append(None)
    # 길이가 5 미만이면 전체 텍스트에서 보조 추출(숫자만)
    while len(comps) < 5: comps.append(None)

    # 총점이 없으면 적용 항목 평균으로 산출
    applies = applicable_axes(qtype)
    used = [c for c, a in zip(comps, applies) if a and isinstance(c, int)]
    if score is None and used:
        score = round(sum(used)/len(used)*5)
    score = score if score is not None else 0

    # 비적용 항목('-')은 None으로 보관
    comp_scores = [c if isinstance(c, int) else None for c in comps[:5]]

    return {"raw": content, "score": score, "competencies": comp_scores}

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
                q_for_rag = st.session_state["current_question"] + "\n" + st.session_state.answer_text[:800]
                sups = retrieve_supports(q_for_rag, st.session_state.get("topk",4))
            res = coach_answer(company, st.session_state["current_question"], st.session_state.answer_text, sups, q_type)
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
    # 좌/우 총점 일관화: 좌측 메트릭은 저장된 score만 사용
    total_score = last.get("score","—")
    c1,c2 = st.columns([1,3])
    with c1: st.metric("총점(/100)", total_score)
    with c2:
        # 우측도 '총점: NN/100'을 한 번 더 명시하여 시각적 일치 강화
        if isinstance(total_score, int):
            st.markdown(f"**총점(시스템 산출)**: {total_score}/100")
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

def comp_df(hist):
    rows=[]
    for h in hist:
        comps=h.get("competencies")
        if not comps: continue
        # None은 비적용('-') → 0으로 채우지 않고 별도 처리 위해 NaN 사용
        vals=[(float(v) if isinstance(v,int) else np.nan) for v in comps]
        rows.append(vals)
    if not rows: return None
    df = pd.DataFrame(rows, columns=AXES)
    # 각 행의 합계(비적용은 제외하고 합산)
    df["합계"] = df[AXES].sum(axis=1, skipna=True)
    return df

cdf = comp_df(st.session_state.history)
if cdf is not None:
    # 평균(비적용 제외)
    means = cdf[AXES].mean(axis=0, skipna=True).tolist()
    # 레이더용: NaN→0으로 표시(시각화용)
    radar_vals = [0 if (np.isnan(v)) else float(v) for v in means]
    if PLOTLY_OK:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=radar_vals+[radar_vals[0]], theta=AXES+[AXES[0]], fill='toself'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,20])), showlegend=False, height=420)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(pd.DataFrame({"score": radar_vals}, index=AXES))
    st.dataframe(cdf.fillna("-"), use_container_width=True)
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
            for k,v in zip(AXES, comps): row[f"comp_{k}"]=("-" if v is None else v)
            # 합계(비적용 제외)
            cvals=[v for v in comps if isinstance(v,int)]
            row["comp_sum"]=sum(cvals) if cvals else 0
        sups=h.get("supports") or []
        row["supports_preview"]=" || ".join([s[2][:120].replace("\n"," ") for s in sups])
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["timestamp","question","user_answer","score","feedback_raw","supports_preview"])
rep = build_report(st.session_state.history)
st.download_button("CSV 다운로드", data=rep.to_csv(index=False).encode("utf-8-sig"),
                   file_name="interview_session_report.csv", mime="text/csv")

st.caption("Tip) 홈페이지/공고 URL을 넣으면 정확도가 크게 올라갑니다. 캐시/요청수 제한으로 속도를 최적화했습니다.")
