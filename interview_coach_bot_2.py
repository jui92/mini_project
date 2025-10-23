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
from datetime import datetime, timezone

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

# ---------- Text utils ----------
def _clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()

def _snippetize(text: str, maxlen: int = 240) -> str:
    t = _clean_text(text)
    return t if len(t) <= maxlen else t[: maxlen - 1] + "…"

def chunk_text(text: str, size: int = 900, overlap: int = 150) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text: return []
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        if end == len(text): break
        start = max(0, end - overlap)
    return chunks

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
VAL_KEYWORDS = [
    "핵심가치","가치","미션","비전","문화","원칙","철학","고객","데이터","혁신",
    "values","mission","vision","culture","principles","philosophy","customer","data","innovation"
]

def _domain(u: str|None) -> str|None:
    if not u: return None
    try:
        if not u.startswith("http"): u = "https://" + u
        return urllib.parse.urlparse(u).netloc.lower().replace("www.","")
    except Exception:
        return None

def _name_similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()

# ---------- Wikipedia summary ----------
def fetch_wikipedia_summary(company_name: str, homepage: str|None=None) -> dict|None:
    if wikipedia is None:
        return None
    try:
        candidates = wikipedia.search(company_name, results=10)
        if not candidates:
            return None
        target_dom = _domain(homepage)
        best = None; best_score = -1.0
        for title in candidates:
            try:
                page = wikipedia.page(title, auto_suggest=False, redirect=True)
            except Exception:
                continue
            first = _clean_text((page.summary or "").split("\n")[0])
            score = _name_similarity(company_name, page.title)
            if any(k in first for k in ["회사","기업","Company","Corporation","Inc","Co., Ltd"]):
                score += 0.15
            page_dom = None
            try:
                page_dom = _domain(page.url)
            except Exception:
                pass
            if target_dom and page_dom and target_dom in page_dom:
                score += 0.25
            if score > best_score:
                best_score = score; best = (page, first)
        if not best: return None
        page, first = best
        return {
            "company_name": page.title,
            "wiki_summary": first
        }
    except Exception:
        return None

# ---------- Simple site scrape for values/recent ----------
def fetch_site_snippets(base_url: str | None, company_name: str | None = None) -> dict:
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
            r = requests.get(url, timeout=6)
            if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
                continue
            soup = BeautifulSoup(r.text, "html.parser")

            # site name / about
            if site_name is None:
                og = soup.find("meta", {"property":"og:site_name"}) or soup.find("meta", {"name":"application-name"})
                if og and og.get("content"): site_name = _clean_text(og["content"])
                elif soup.title and soup.title.string: site_name = _clean_text(soup.title.string.split("|")[0])
            if about_para is None:
                # hero/lead 단락 추정
                hero = soup.find(["p","div"], class_=re.compile(r"(lead|hero|intro)", re.I)) if soup else None
                if hero:
                    about_para = _snippetize(hero.get_text(" "))

            for tag in soup.find_all(["h1","h2","h3","p","li"]):
                txt = _clean_text(tag.get_text(separator=" "))
                if 10 <= len(txt) <= 240:
                    if any(k.lower() in txt.lower() for k in VAL_KEYWORDS):
                        values_found.append(txt)
                    if any(k in txt for k in ["프로젝트","개발","출시","성과","project","launched","release","delivered","improved"]):
                        recent_found.append(txt)
        except Exception:
            continue

    # name check
    if company_name and site_name and _name_similarity(company_name, site_name) < 0.35:
        values_found, recent_found = [], []

    # dedup & trim
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

# ---------- Google News RSS (최근 뉴스) ----------
def fetch_news(company_name: str, max_items: int = 6, lang: str = "ko") -> List[dict]:
    # Google News RSS (공식 API 아님) — 단순 파싱
    q = urllib.parse.quote(company_name)
    url = f"https://news.google.com/rss/search?q={q}&hl=ko&gl=KR&ceid=KR:ko"
    items = []
    try:
        r = requests.get(url, timeout=8)
        if r.status_code != 200:
            return []
        soup = BeautifulSoup(r.text, "xml")
        for it in soup.find_all("item")[:max_items]:
            title = _clean_text(it.title.get_text()) if it.title else ""
            link = it.link.get_text() if it.link else ""
            pub = it.pubDate.get_text() if it.pubDate else ""
            items.append({"title": title, "link": link, "pubDate": pub})
    except Exception:
        return []
    return items

# ---------- Job posting discovery & parsing ----------
SEARCH_ENGINES = [
    # DuckDuckGo HTML endpoint (심플, 차단 가능성 낮음)
    "https://duckduckgo.com/html/?q={query}"
]
JOB_SITES = [
    "wanted.co.kr", "saramin.co.kr", "jobkorea.co.kr", "rocketpunch.com",
    "indeed.com", "linkedin.com", "recruit.navercorp.com", "kakao.recruit", "naver"
]

def discover_job_posting_urls(company_name: str, role: str, limit: int = 5) -> List[str]:
    # 간단 검색: "company role site:wanted.co.kr OR site:saramin.co.kr ..."
    site_part = " OR ".join([f'site:{d}' for d in JOB_SITES])
    q = f'{company_name} {role} ({site_part})'
    urls = []
    for engine in SEARCH_ENGINES:
        url = engine.format(query=urllib.parse.quote(q))
        try:
            r = requests.get(url, timeout=8, headers={"User-Agent":"Mozilla/5.0"})
            if r.status_code != 200: 
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for a in soup.find_all("a", href=True):
                href = a["href"]
                # DDG는 리다이렉트 링크일 수 있음
                if href.startswith("/l/?kh=-1&uddg="):
                    href = urllib.parse.unquote(href.split("/l/?kh=-1&uddg=")[-1])
                dom = _domain(href)
                if not dom: 
                    continue
                if any(d in dom for d in JOB_SITES):
                    if href not in urls:
                        urls.append(href)
                if len(urls) >= limit:
                    break
        except Exception:
            continue
    return urls[:limit]

def _extract_json_ld_job(soup: BeautifulSoup) -> Optional[dict]:
    # schema.org JobPosting JSON-LD 추출
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(s.string or "")
            if isinstance(data, list):
                for obj in data:
                    typ = obj.get("@type") if isinstance(obj, dict) else None
                    if (isinstance(typ, list) and "JobPosting" in typ) or typ == "JobPosting":
                        return obj
            elif isinstance(data, dict):
                typ = data.get("@type")
                if (isinstance(typ, list) and "JobPosting" in typ) or typ == "JobPosting":
                    return data
        except Exception:
            continue
    return None

def parse_job_posting(url: str) -> dict:
    # 공고 페이지에서 모집분야/주요업무/자격요건 추출 (1) JSON-LD 우선 (2) 헤딩/키워드 휴리스틱
    out = {"title": None, "responsibilities": [], "qualifications": [], "company_intro": None}
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code != 200 or "text/html" not in r.headers.get("content-type",""):
            return out
        soup = BeautifulSoup(r.text, "html.parser")

        # (1) JSON-LD JobPosting
        jp = _extract_json_ld_job(soup)
        if jp:
            out["title"] = jp.get("title")
            # desc → 문장/불릿 분해
            desc = _clean_text(jp.get("description", ""))
            if desc:
                bullets = re.split(r"[•\-\n•·▪️▶︎]+", desc)
                bullets = [b.strip(" -•·▪️▶︎") for b in bullets if len(b.strip()) > 3]
                # 간단 규칙으로 responsibilities/qualifications 분할
                for b in bullets:
                    if any(k in b for k in ["자격", "요건", "requirements", "qualification", "필수", "우대"]):
                        out["qualifications"].append(b)
                    else:
                        out["responsibilities"].append(b)

        # (2) 휴리스틱: 헤더에 기반한 섹션 추출
        sections = {}
        for h in soup.find_all(re.compile("^h[1-4]$")):
            head = _clean_text(h.get_text())
            if not head: 
                continue
            nxt = []
            sib = h.find_next_sibling()
            stop_at = {"h1","h2","h3","h4"}
            while sib and sib.name not in stop_at:
                if sib.name in {"p","li","ul","ol","div"}:
                    txt = _clean_text(sib.get_text(" "))
                    if len(txt) > 5: nxt.append(txt)
                sib = sib.find_next_sibling()
            if nxt:
                sections[head] = " ".join(nxt)

        # 키워드 매칭
        resp_keys = ["주요 업무","담당 업무","업무","Responsibilities","What you will do","Role"]
        qual_keys = ["자격 요건","지원 자격","우대","Requirements","Qualifications","Must have","Preferred"]
        def pick(keys):
            for k in sections:
                if any(kk.lower() in k.lower() for kk in keys):
                    return sections[k]
            return None

        if not out["responsibilities"]:
            resp = pick(resp_keys)
            if resp:
                out["responsibilities"] = [x for x in re.split(r"[•\-\n•·▪️▶︎]+", resp) if len(x.strip())>3][:12]

        if not out["qualifications"]:
            qual = pick(qual_keys)
            if qual:
                out["qualifications"] = [x for x in re.split(r"[•\-\n•·▪️▶︎]+", qual) if len(x.strip())>3][:12]

        # 회사 소개 추정
        meta_desc = soup.find("meta", {"name":"description"}) or soup.find("meta", {"property":"og:description"})
        if meta_desc and meta_desc.get("content"):
            out["company_intro"] = _snippetize(meta_desc["content"], 220)
    except Exception:
        pass
    # 최종 정리
    out["responsibilities"] = [ _snippetize(x, 140) for x in out["responsibilities"] ][:12]
    out["qualifications"]  = [ _snippetize(x, 140) for x in out["qualifications"]   [:12]]
    return out

# ---------- OpenAI client ----------
with st.sidebar:
    st.title("🎯 가상 면접 코치")

    API_KEY = load_api_key_from_env_or_secrets()
    if not API_KEY:
        st.info("환경변수/Secrets에서 키를 못 찾았습니다. 아래에 입력하면 즉시 사용 가능해요.")
        API_KEY = st.text_input("OPENAI_API_KEY", type="password")

    MODEL = st.selectbox("챗 모델", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
    EMBED_MODEL = st.selectbox("임베딩 모델", ["text-embedding-3-small", "text-embedding-3-large"], index=0)

    # 디버그
    _openai_ver = None; _httpx_ver = None
    try:
        import openai as _openai_pkg; _openai_ver = getattr(_openai_pkg, "__version__", None)
    except Exception: pass
    try:
        import httpx as _httpx_pkg; _httpx_ver = getattr(_httpx_pkg, "__version__", None)
    except Exception: pass
    with st.expander("디버그: 시크릿/버전 상태"):
        st.write({
            "env_has_key": bool(os.getenv("OPENAI_API_KEY")),
            "api_key_provided": bool(API_KEY),
            "openai_version": _openai_ver,
            "httpx_version": _httpx_ver,
        })

if not API_KEY:
    st.error("OpenAI API Key가 필요합니다. (Cloud: App → Settings → Secrets)")
    st.stop()

try:
    client = OpenAI(api_key=API_KEY, timeout=30.0)
except TypeError:
    st.error("OpenAI 초기화 TypeError. requirements.txt에서 openai==1.44.0, httpx==0.27.2로 고정 후 Clear cache → Reboot 해주세요.")
    st.stop()
except Exception as e:
    st.error(f"OpenAI 클라이언트 초기화 오류: {e}")
    st.stop()

# ---------- Sidebar: 회사/직무 + 자동 프로필 + 공고/뉴스 + RAG ----------
with st.sidebar:
    st.markdown("---")
    st.markdown("#### 회사/직무 설정")

    # 직무 직접 선택/입력 (문제 1 해결)
    role_title = st.text_input("지원 직무명 (예: 데이터 애널리스트, ML 엔지니어)")

    st.markdown("#### 🔎 자동 프로필 생성 (회사/홈페이지/채용공고)")
    auto_name = st.text_input("회사 이름 (예: 네이버, Kakao, Samsung SDS)")
    auto_home = st.text_input("홈페이지 URL (선택)")
    job_url = st.text_input("채용 공고 URL (선택) — 없으면 아래 버튼으로 검색 시도")

    col_a, col_b = st.columns(2)
    with col_a:
        auto_add_to_rag = st.checkbox("홈페이지/뉴스/공고를 RAG에 추가", value=True)
    with col_b:
        diversity_k = st.slider("질문 후보 개수", 3, 8, 6, 1)

    if st.button("회사/직무 자동 세팅"):
        if not auto_name.strip():
            st.warning("회사 이름을 입력해 주세요.")
        else:
            with st.spinner("회사·직무·공고·뉴스 수집 중..."):
                # 회사 기본(위키+홈페이지)
                wiki = fetch_wikipedia_summary(auto_name.strip(), auto_home.strip() or None) or {}
                site = fetch_site_snippets(auto_home.strip() or None, auto_name.strip()) if auto_home.strip() else {"values": [], "recent": [], "site_name": None, "about": None}

                # 공고 URL이 없으면 검색 시도 (문제 2 해결)
                jp_data = {"title": None, "responsibilities": [], "qualifications": [], "company_intro": None}
                discovered = []
                if job_url.strip():
                    discovered = [job_url.strip()]
                else:
                    if role_title.strip():
                        discovered = discover_job_posting_urls(auto_name.strip(), role_title.strip(), limit=4)

                if discovered:
                    jp_data = parse_job_posting(discovered[0])

                # 뉴스 (문제 3 해결)
                news_items = fetch_news(auto_name.strip(), max_items=6, lang="ko")

                # company 객체 구성
                company = {
                    "company_name": site.get("site_name") or wiki.get("company_name") or auto_name.strip(),
                    "homepage": auto_home.strip() or None,
                    "wiki_summary": wiki.get("wiki_summary"),
                    "values": site.get("values", []),
                    "recent_projects": site.get("recent", []),
                    "role": role_title.strip(),
                    "role_requirements": jp_data["responsibilities"] or [],
                    "role_qualifications": jp_data["qualifications"] or [],
                    "job_url": discovered[0] if discovered else (job_url.strip() or None),
                    "company_intro": jp_data["company_intro"] or site.get("about"),
                    "news": news_items
                }
                st.session_state["company_override"] = company

                # RAG 자동 투입 (문제 3, 5의 근거 강화)
                if auto_add_to_rag:
                    texts = []
                    # 홈페이지 텍스트 몇 경로
                    if auto_home.strip():
                        for p in ["", "/about", "/values", "/mission", "/company"]:
                            u = auto_home.strip().rstrip("/") + p
                            try:
                                r = requests.get(u, timeout=6)
                                if r.status_code == 200 and "text/html" in r.headers.get("content-type",""):
                                    s = BeautifulSoup(r.text, "html.parser")
                                    txts = [_clean_text(t.get_text(" ")) for t in s.find_all(["h1","h2","h3","p","li"])]
                                    page = "\n".join([t for t in txts if len(t) > 10])
                                    if page: texts.append(page)
                            except Exception:
                                pass
                    # 뉴스 본문은 도메인 차단이 있을 수 있으니 제목+링크+날짜로만
                    if news_items:
                        news_text = "\n".join([f"[NEWS] {n['title']} ({n.get('pubDate','')}) {n['link']}" for n in news_items])
                        texts.append(news_text)
                    # 채용 공고 텍스트
                    if jp_data["responsibilities"] or jp_data["qualifications"]:
                        job_text = "주요 업무:\n- " + "\n- ".join(jp_data["responsibilities"]) + "\n자격 요건:\n- " + "\n- ".join(jp_data["qualifications"])
                        texts.append(job_text)

                    if texts:
                        chs = []
                        for t in texts:
                            chs += chunk_text(t, 900, 150)
                        try:
                            embs = client.embeddings.create(model="text-embedding-3-small", input=chs)
                            embs = np.array([d.embedding for d in embs.data], dtype=np.float32)
                            if "rag_store" not in st.session_state:
                                st.session_state.rag_store = {"chunks": [], "embeds": None}
                            st.session_state.rag_store["chunks"] = (st.session_state.rag_store.get("chunks", []) or []) + chs
                            if st.session_state.rag_store.get("embeds") is None or st.session_state.rag_store["embeds"].size == 0:
                                st.session_state.rag_store["embeds"] = embs
                            else:
                                st.session_state.rag_store["embeds"] = np.vstack([st.session_state.rag_store["embeds"], embs])
                        except Exception:
                            st.info("일부 텍스트는 RAG 인덱싱에서 제외되었습니다(요청 제한/차단 가능).")
            st.success("자동 세팅 완료!")

    st.markdown("---")
    st.markdown("#### RAG (선택 업로드)")
    rag_enabled = st.toggle("회사 문서 기반 질문/코칭 사용", value=True)
    chunk_size = st.slider("청크 길이(문자)", 400, 2000, 900, 100)
    chunk_overlap = st.slider("오버랩(문자)", 0, 400, 150, 10)
    top_k = st.slider("검색 상위 K", 1, 8, 4, 1)
    st.caption("TXT/MD/PDF 업로드 가능 (세션 메모리 내 처리)")
    docs = st.file_uploader("회사 문서 업로드 (여러 파일 가능)", type=["txt", "md", "pdf"], accept_multiple_files=True)

# ---------- company 결정 ----------
if "company_override" in st.session_state:
    company = st.session_state["company_override"]
else:
    # 초기 더미 (수동 업로드/기본 파일 대신 자동 세팅 유도)
    company = {
        "company_name": "(회사명 미설정)",
        "homepage": None,
        "wiki_summary": None,
        "values": [],
        "recent_projects": [],
        "role": role_title,
        "role_requirements": [],
        "role_qualifications": [],
        "job_url": None,
        "company_intro": None,
        "news": []
    }

# ---------- session states ----------
if "rag_store" not in st.session_state:
    st.session_state.rag_store = {"chunks": [], "embeds": None}
if "history" not in st.session_state:
    st.session_state.history = []
if "current_question" not in st.session_state:
    st.session_state.current_question = ""

# ---------- Upload → RAG ----------
def embed_texts(client: OpenAI, embed_model: str, texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    resp = client.embeddings.create(model=embed_model, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

if rag_enabled and docs:
    with st.spinner("문서 처리 중..."):
        all_chunks = []
        for up in docs:
            text = read_file_to_text(up)
            if not text: continue
            all_chunks.extend(chunk_text(text, chunk_size, chunk_overlap))
        if all_chunks:
            embeds = embed_texts(client, EMBED_MODEL, all_chunks)
            st.session_state.rag_store["chunks"] = (st.session_state.rag_store.get("chunks", []) or []) + all_chunks
            if st.session_state.rag_store.get("embeds") is None or st.session_state.rag_store["embeds"].size == 0:
                st.session_state.rag_store["embeds"] = embeds
            else:
                st.session_state.rag_store["embeds"] = np.vstack([st.session_state.rag_store["embeds"], embeds])
            st.success(f"RAG 준비 완료: 청크 {len(all_chunks)}개 추가")
        else:
            st.info("업로드 문서에서 추출된 텍스트가 없습니다.")

# ---------- Retrieval helpers ----------
def cosine_topk(matrix: np.ndarray, query: np.ndarray, k: int = 4):
    if matrix.size == 0:
        return np.array([]), np.array([], dtype=int)
    qn = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
    mn = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    sims = mn @ qn.T
    sims = sims.reshape(-1)
    idx = np.argsort(-sims)[:k]
    scores = sims[idx]
    return scores, idx

def retrieve_supports(query_text: str, k: int) -> List[Tuple[str, float, str]]:
    store = st.session_state.rag_store
    chunks, embeds = store.get("chunks", []), store.get("embeds", None)
    if not rag_enabled or embeds is None or len(chunks) == 0:
        return []
    qv = embed_texts(client, EMBED_MODEL, [query_text])
    scores, idxs = cosine_topk(embeds, qv, k=k)
    return [("회사자료", float(s), chunks[int(i)]) for s, i in zip(scores, idxs)]

# ---------- Diversity helpers ----------
def _similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def pick_diverse(candidates: list[str], history: list[str], gamma: float = 0.25) -> str:
    # 후보별 점수 = 평균 유사도 + gamma*표준편차 → 최소 점수 선택(다양성 최대화)
    if not candidates: return ""
    if not history: return random.choice(candidates)
    best = None; best_score = 1e9
    for q in candidates:
        sims = [ _similarity(q,h) for h in history ]
        if not sims: sims=[0.0]
        score = (sum(sims)/len(sims)) + gamma*(np.std(sims))
        if score < best_score:
            best_score = score; best = q
    return best

TYPE_INSTRUCTIONS = {
    "행동(STAR)": "과거 실무 사례를 끌어내도록 S(상황)-T(과제)-A(행동)-R(성과)를 유도하는 질문",
    "기술 심층": "핵심 기술적 의사결정·트레이드오프·성능/비용/품질 지표를 파고드는 심층 질문",
    "핵심가치 적합성": "핵심가치와 태도를 검증하는, 상황기반 행동을 유도하는 질문",
    "역질문": "지원자가 회사를 평가할 수 있도록 통찰력 있는 역질문"
}

# ---------- Company context (채용공고 기준 요약: 문제 5 해결) ----------
def build_company_context_for_prompt(c: dict) -> str:
    # 프롬프트용 (간략)
    base = textwrap.dedent(f"""
    [회사명] {c.get('company_name','')}
    [회사 소개] {c.get('company_intro') or c.get('wiki_summary') or ''}
    [모집 분야] {c.get('role','')}
    [주요 업무] {", ".join(c.get('role_requirements', [])[:6])}
    [자격 요건] {", ".join(c.get('role_qualifications', [])[:6])}
    [핵심가치] {", ".join(c.get('values', [])[:6])}
    [최근 이슈/뉴스] {", ".join([_snippetize(n['title'], 70) for n in c.get('news', [])[:3]])}
    """).strip()
    return base

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

def build_focuses(company: dict, supports: List[Tuple[str,float,str]], k: int = 4) -> list[str]:
    # 포커스는 "직무 → 공고(업무/요건) → 가치 → 최근이슈 → RAG문장" 우선
    pool = []
    if company.get("role"): pool.append(company["role"])
    pool += company.get("role_requirements", [])[:6]
    pool += company.get("role_qualifications", [])[:6]
    pool += company.get("values", [])[:6]
    pool += [ _snippetize(n['title'], 60) for n in company.get("news", [])[:4] ]
    for _,_,txt in (supports or [])[:3]:
        pool += [t.strip() for t in re.split(r"[•\-\n\.]", txt) if 6 < len(t.strip()) < 100][:3]
    pool = [p for p in pool if p]
    random.shuffle(pool)
    return pool[:k]

# ---------- Question generation (문제 3,4,5 & 맞춤감 강화) ----------
def gen_question(company: dict, qtype: str, level: str, supports: List[Tuple[str, float, str]], num_candidates: int = 6) -> str:
    ctx = build_company_context_for_prompt(company)
    focuses = build_focuses(company, supports, k=min(4, num_candidates))
    style = TYPE_INSTRUCTIONS.get(qtype, "구체적이고 행동을 이끌어내는 질문")
    rag_note = ""
    if supports:
        joined = "\n".join([f"- ({s:.2f}) {txt[:200]}" for _, s, txt in supports[:3]])
        rag_note = f"\n[근거 발췌]\n{joined}"

    # 랜덤성 향상: seed를 시간/세션 기반으로 섞음 (문제 4)
    seed = int(time.time()*1000) % 2_147_483_647
    random_factor = random.random()

    sys = f"""너는 '{company.get('company_name','')}'의 '{company.get('role','')}' 면접관이다.
회사 맥락, 채용공고(주요업무/자격요건), 최근 뉴스/이슈, (있다면) 근거 문서를 반영하여 **{qtype}** 유형({style})의 질문 **{num_candidates}개 후보**를 한국어로 생성하라.
각 후보는 서로 **형태·관점·키워드**가 달라야 한다. 난이도는 {level}.
아래 '포커스' 중 최소 1개 키워드를 질의문에 **명시적으로 포함**하라.
사소한 재구성(지표/수치/기간/규모/리스크 요인 등)을 섞어 **서로 다른 질문**이 되도록 한다.
포맷: 1) ... 2) ... ... (한 줄씩)"""
    user = f"""[회사/직무 컨텍스트]
{ctx}
[포커스(무작위 일부)]
- {chr(10).join(focuses)}{rag_note}
[랜덤시드] {seed}; rf={random_factor:.4f}"""

    resp = client.chat.completions.create(
        model=MODEL, temperature=0.95,  # 다양성 ↑
        messages=[{"role":"system","content":sys},{"role":"user","content":user}]
    )
    raw = resp.choices[0].message.content.strip()
    cands = [re.sub(r'^\s*\d+\)\s*','',line).strip() for line in raw.splitlines() if re.match(r'^\s*\d+\)', line)]
    if not cands:
        cands = [l.strip("- ").strip() for l in raw.splitlines() if len(l.strip())>0][:num_candidates]

    # 최근 질문들과의 반중복 선택 (문제 4)
    hist_qs = [h["question"] for h in st.session_state.get("history", [])][-10:]
    selected = pick_diverse(cands, hist_qs, gamma=0.35)
    return selected or (cands[0] if cands else "질문 생성 실패")

# ---------- Coaching ----------
def coach_answer(company: dict, question: str, user_answer: str, supports: List[Tuple[str, float, str]]) -> Dict:
    ctx = build_company_context_for_prompt(company)
    rag_note = ""
    if supports:
        joined = "\n".join([f"- ({s:.3f}) {txt[:500]}" for (_, s, txt) in supports])
        rag_note = f"\n[회사 근거 문서 발췌]\n{joined}\n"
    competencies = ["문제정의", "데이터/지표", "실행력/주도성", "협업/커뮤니케이션", "고객가치"]
    comp_str = ", ".join(competencies)
    sys = f"""너는 톱티어 면접 코치다. 한국어로 아래 형식에 맞춰 답하라:
1) 총점: 0~10 정수 1개
2) 강점: 2~3개 불릿
3) 리스크: 2~3개 불릿
4) 개선 포인트: 3개 불릿 (행동·지표·임팩트 중심)
5) 수정본 답변: STAR(상황-과제-행동-성과) 구조로 자연스럽고 간결하게
6) 역량 점수: [{comp_str}] 각각 0~5 정수 (한 줄에 쉼표로 구분)
채점 기준은 회사/직무 맥락, 채용공고(주요업무/자격요건), 질문 내 **포커스/키워드** 부합 여부를 포함한다.
추가 설명 금지. 형식 유지."""
    user = f"""[회사/직무 컨텍스트]
{ctx}
{rag_note}
[면접 질문]
{question}

[후보자 답변]
{user_answer}
"""
    resp = client.chat.completions.create(
        model=MODEL, temperature=0.35,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}]
    )
    content = resp.choices[0].message.content.strip()
    m = re.search(r'([0-9]{1,2})\s*(?:/10|점|$)', content)
    score = None
    if m:
        try:
            score = int(m.group(1)); score = max(0, min(10, score))
        except: pass
    comp_scores = None
    nums = re.findall(r'\b([0-5])\b', content.splitlines()[-1])
    if len(nums) >= 5:
        comp_scores = [int(x) for x in nums[:5]]
    return {"raw": content, "score": score, "competencies": comp_scores}

# ---------- UI ----------
left, right = st.columns([1, 1])

with left:
    st.header("① 채용공고 기준 회사 요약")
    st.json(build_company_summary_for_ui(company), expanded=True)

    st.header("② 질문 생성")
    st.caption("‘질문 생성 힌트’에 키워드(예: 전환 퍼널, 성능-비용 트레이드오프) 1~2개 정도만 넣으면 더 맞춤화됩니다.")
    q_type = st.selectbox("질문 유형", ["행동(STAR)", "기술 심층", "핵심가치 적합성", "역질문"], index=0)
    level  = st.selectbox("난이도/연차", ["주니어", "미들", "시니어"], index=0)
    prompt_hint = st.text_input("질문 생성 힌트(선택)")

    if st.button("새 질문 받기", use_container_width=True):
        try:
            supports = []
            if rag_enabled and (docs or st.session_state.rag_store.get("chunks")):
                base_q = prompt_hint.strip() or f"{company.get('role','')} {' '.join(company.get('role_requirements', [])[:3])}"
                supports = retrieve_supports(base_q, top_k)
            q = gen_question(company, q_type, level, supports, num_candidates=diversity_k)
            st.session_state.current_question = q
            st.session_state.last_supports_q = supports
        except Exception as e:
            st.error(f"질문 생성 오류: {e}")

    st.text_area("질문", height=110, value=st.session_state.get("current_question",""))

    if rag_enabled and st.session_state.get("last_supports_q"):
        with st.expander("질문 생성에 사용된 근거 보기"):
            for i, (_, sc, txt) in enumerate(st.session_state.last_supports_q, 1):
                st.markdown(f"**[{i}] sim={sc:.3f}**\n\n{txt[:600]}{'...' if len(txt)>600 else ''}")
                st.markdown("---")

with right:
    st.header("③ 나의 답변")
    answer = st.text_area("여기에 답변을 작성하세요 (STAR 권장: 상황-과제-행동-성과)", height=180)
    if st.button("채점 & 코칭", type="primary", use_container_width=True):
        if not st.session_state.get("current_question"):
            st.warning("먼저 '새 질문 받기'로 질문을 생성하세요.")
        elif not answer.strip():
            st.warning("답변을 작성해 주세요.")
        else:
            with st.spinner("코칭 중..."):
                try:
                    supports = []
                    if rag_enabled and (docs or st.session_state.rag_store.get("chunks")):
                        q_for_rag = st.session_state["current_question"] + "\n" + answer[:800]
                        supports = retrieve_supports(q_for_rag, top_k)
                    res = coach_answer(company, st.session_state["current_question"], answer, supports)
                    st.session_state.history.append({
                        "ts": pd.Timestamp.now(),
                        "question": st.session_state["current_question"],
                        "user_answer": answer,
                        "score": res.get("score"),
                        "feedback": res.get("raw"),
                        "supports": supports,
                        "competencies": res.get("competencies")
                    })
                except Exception as e:
                    st.error(f"코칭 오류: {e}")

# ---------- Results / Radar / CSV ----------
st.divider()
st.subheader("④ 피드백 결과")
if st.session_state.history:
    last = st.session_state.history[-1]
    c1, c2 = st.columns([1, 3])
    with c1:
        st.metric("총점(/10)", last.get("score", "—"))
    with c2:
        st.markdown(last.get("feedback", ""))

    if rag_enabled and last.get("supports"):
        with st.expander("코칭에 사용된 근거 보기"):
            for i, (_, sc, txt) in enumerate(last["supports"], 1):
                st.markdown(f"**[{i}] sim={sc:.3f}**\n\n{txt[:800]}{'...' if len(txt)>800 else ''}")
                st.markdown("---")

st.divider()
st.subheader("⑤ 역량 레이더 (세션 누적)")
competencies = ["문제정의", "데이터/지표", "실행력/주도성", "협업/커뮤니케이션", "고객가치"]

def compute_comp_df(hist):
    rows = []
    for h in hist:
        if h.get("competencies") and len(h["competencies"]) == 5:
            rows.append(h["competencies"])
    if not rows:
        return None
    return pd.DataFrame(rows, columns=competencies)

comp_df = compute_comp_df(st.session_state.history)
if comp_df is not None:
    avg_scores = comp_df.mean().values.tolist()
    if PLOTLY_OK:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=avg_scores + [avg_scores[0]],
            theta=competencies + [competencies[0]],
            fill='toself',
            name='평균(0~5)'
        ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,5])), showlegend=False, height=420)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Plotly 미설치 상태 — 막대 그래프로 대체합니다.")
        st.bar_chart(pd.DataFrame({"score": avg_scores}, index=competencies))
    st.dataframe(comp_df, use_container_width=True)
else:
    st.info("아직 역량 점수가 파싱된 코칭 결과가 없습니다.")

st.divider()
st.subheader("⑥ 세션 리포트 (CSV)")
def build_report_df(hist):
    rows = []
    for h in hist:
        row = {
            "timestamp": h.get("ts"),
            "question": h.get("question"),
            "user_answer": h.get("user_answer"),
            "score": h.get("score"),
            "feedback_raw": h.get("feedback"),
        }
        comps = h.get("competencies")
        if comps and len(comps) == 5:
            for k, v in zip(competencies, comps):
                row[f"comp_{k}"] = v
        sups = h.get("supports") or []
        row["supports_preview"] = " || ".join([s[2][:120].replace("\n"," ") for s in sups])
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["timestamp","question","user_answer","score","feedback_raw","supports_preview"])
    return pd.DataFrame(rows)

report_df = build_report_df(st.session_state.history)
st.download_button("CSV 다운로드", data=report_df.to_csv(index=False).encode("utf-8-sig"),
                   file_name="interview_session_report.csv", mime="text/csv")

st.caption("Tip) 공고 URL을 직접 넣으면 정확도가 크게 올라갑니다. 뉴스/홈페이지/공고 텍스트는 RAG에 자동 투입 옵션으로 근거 기반 질문/코칭을 강화할 수 있어요.")
