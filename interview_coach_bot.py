# -*- coding: utf-8 -*-
# ==========================================================
# 회사 특화 가상 면접 코치 (KR)
# - 회사 소개만 LLM 요약 / 채용 요건(업무·자격·우대)은 '원문 그대로'
# - 동적 채용 공고 대응: r.jina.ai 스냅샷 + 사이트별 처리 + 섹션 헤더 우선 분리 + 불릿 분류기
# - 자동 상세 공고 URL 우선 선택(목록 페이지 제거)
# - 질문 다양화 / RAG(선택)
# - 채점: 적용 가능한 축에 한해 0~20 채점, 총점 = (적용축 평균 × 5) → 100점
# - 점수 일관화: 좌/우/CSV/레이더 모두 동일 점수 사용
# - 레이더 표에 '합계' 추가(비적용은 제외하고 합산)
# ==========================================================

import os, io, re, json, textwrap, urllib.parse, difflib, random, functools
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ---------- optional dependencies ----------
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
from urllib.parse import urlparse

# ---------- page ----------
st.set_page_config(page_title="회사 특화 가상 면접 코치", page_icon="🎯", layout="wide")

# ---------- keys ----------
def _secrets_file_exists() -> bool:
    return any(os.path.exists(p) for p in [
        os.path.join(os.path.expanduser("~"), ".streamlit", "secrets.toml"),
        os.path.join(os.getcwd(), ".streamlit", "secrets.toml"),
    ])

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

# ---------- utils ----------
def _clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()

def _snippetize(text: str, n: int = 220) -> str:
    t = _clean_text(text)
    return t if len(t) <= n else t[: n-1] + "…"

def chunk_text(text: str, size: int = 900, overlap: int = 150):
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text: return []
    out, i = [], 0
    while i < len(text):
        j = min(len(text), i + size)
        out.append(text[i:j])
        if j == len(text): break
        i = max(0, j - overlap)
    return out

def read_file_to_text(up) -> str:
    name = up.name.lower()
    data = up.read()
    if name.endswith((".txt",".md")):
        for enc in ("utf-8","cp949","euc-kr"):
            try: return data.decode(enc)
            except Exception: continue
        return data.decode("utf-8", errors="ignore")
    if name.endswith(".pdf"):
        if pypdf is None: return ""
        try:
            reader = pypdf.PdfReader(io.BytesIO(data))
            return "\n\n".join([(reader.pages[i].extract_text() or "") for i in range(len(reader.pages))])
        except Exception:
            return ""
    return ""

def _domain(u: str|None) -> str|None:
    if not u: return None
    try:
        if not u.startswith(("http://","https://")):
            u = "https://" + u
        return urllib.parse.urlparse(u).netloc.lower().replace("www.","")
    except Exception:
        return None

# ---------- http cache ----------
@functools.lru_cache(maxsize=256)
def _cached_get(url: str, timeout: int = 8) -> Optional[str]:
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code==200 and "text/html" in r.headers.get("content-type",""):
            return r.text
    except Exception:
        pass
    return None

# ---------- NAVER open API ----------
def _naver_api_get(api: str, params: dict, cid: str, csec: str):
    url = f"https://openapi.naver.com/v1/search/{api}.json"
    headers = {"X-Naver-Client-Id": cid, "X-Naver-Client-Secret": csec, "User-Agent": "Mozilla/5.0"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=8)
        if r.status_code != 200: return None
        return r.json()
    except Exception:
        return None

def naver_search_news(q: str, display: int = 6) -> list[dict]:
    cid, csec = load_naver_keys()
    if not (cid and csec): return []
    js = _naver_api_get("news", {"query": q, "display": display, "sort": "date"}, cid, csec)
    if not js: return []
    out=[]
    for it in js.get("items", []):
        title = _clean_text(re.sub(r"</?b>|&quot;|&apos;|&amp;|&lt;|&gt;", "", it.get("title","")))
        out.append({"title": title, "link": it.get("link"), "pubDate": it.get("pubDate")})
    return out

def naver_search_web(q: str, display: int = 5) -> list[str]:
    cid, csec = load_naver_keys()
    if not (cid and csec): return []
    js = _naver_api_get("webkr", {"query": q, "display": display, "sort": "date"}, cid, csec)
    if not js: return []
    links=[]
    for it in js.get("items", []):
        link = it.get("link")
        if link and link not in links: links.append(link)
    return links

# ---------- site snippets ----------
VAL_KEYWORDS = ["핵심가치","가치","미션","비전","문화","원칙","철학","고객","데이터","혁신",
                "values","mission","vision","culture","principles","philosophy","customer","data","innovation"]

def fetch_site_snippets(home: str|None) -> dict:
    if not home: return {"values":[], "recent":[], "about":None}
    if not home.startswith(("http://","https://")): home = "https://" + home
    values, recent, about = [], [], None
    for path in ["","/about","/company","/about-us"]:
        html = _cached_get(home.rstrip("/") + path, timeout=6)
        if not html: continue
        soup = BeautifulSoup(html, "html.parser")
        if about is None:
            hero = soup.find(["p","div"], class_=re.compile(r"(lead|hero|intro)", re.I))
            if hero: about = _snippetize(hero.get_text(" "))
        for tag in soup.find_all(["h1","h2","h3","p","li"]):
            txt = _clean_text(tag.get_text(" "))
            if 8 <= len(txt) <= 220:
                if any(k in txt.lower() for k in [k.lower() for k in VAL_KEYWORDS]): values.append(txt)
                if any(k in txt for k in ["프로젝트","개발","출시","성과","launched","release","delivered"]): recent.append(txt)
    def dedup(x): 
        s=set(); out=[]
        for t in x:
            if t not in s: s.add(t); out.append(t)
        return out
    values = dedup(values)[:6]; recent = dedup(recent)[:6]
    return {"values": values, "recent": recent, "about": about}

# ---------- job discover ----------
CAREER_HINTS = ["careers","career","jobs","job","recruit","recruiting","join","hire","hiring","채용","인재","입사지원","채용공고","인재영입","커리어"]
JOB_SITES   = ["wanted.co.kr","saramin.co.kr","jobkorea.co.kr","rocketpunch.com","indeed.com","linkedin.com"]

DETAIL_HINTS = re.compile(r"(job|position|posting|recruit|notice|opening).*(\d{3,}|detail)|/positions/|/jobs/|/recruit/|/careers/\w+", re.I)

def discover_job_from_homepage(home: str, limit: int = 3) -> list[str]:
    if not home: return []
    if not home.startswith(("http://","https://")): home = "https://" + home
    html = _cached_get(home, timeout=8)
    if not html: return []
    soup = BeautifulSoup(html, "html.parser")
    links=[]
    for path in ["careers","recruit","jobs","career","채용","인재영입","recruitment","join"]:
        links.append(urllib.parse.urljoin(home.rstrip("/") + "/", path))
    for a in soup.find_all("a", href=True):
        href=a["href"]; text=(a.get_text() or "").lower()
        if any(k in href.lower() or k in text for k in CAREER_HINTS):
            links.append(urllib.parse.urljoin(home, href))
    out=[]; seen=set()
    for lk in links:
        if lk not in seen: seen.add(lk); out.append(lk)
        if len(out)>=limit: break
    return out

def discover_job_urls(name: str, role: str, home: str|None, limit: int = 3) -> list[str]:
    urls=[]
    if home: urls += discover_job_from_homepage(home, limit=limit)
    if NAVER_ID and NAVER_SECRET:
        for dom in JOB_SITES:
            if len(urls)>=limit: break
            q = f"{name} {role} site:{dom}" if role else f"{name} 채용 site:{dom}"
            links = naver_search_web(q, display=7)
            for lk in links:
                urls.append(lk)
    # 상세 공고 URL 우선 정렬 + 중복 제거
    uniq = []
    for u in urls:
        if u not in uniq: uniq.append(u)
    uniq = sorted(uniq, key=lambda u: 0 if DETAIL_HINTS.search(u or "") else 1)
    return uniq[:limit]

# ---------- dynamic snapshot (Jina Reader) ----------
def fetch_text_snapshot(url: str, timeout: int = 12) -> str:
    """렌더된 페이지 텍스트 스냅샷(r.jina.ai). lstrip 버그 없이 안전하게."""
    try:
        if not url.startswith(("http://","https://")):
            url = "https://" + url
        parsed = urlparse(url)
        snap_url = f"https://r.jina.ai/http/{parsed.geturl()}"
        r = requests.get(snap_url, timeout=timeout, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code == 200 and r.text:
            return re.sub(r"\s+", " ", r.text).strip()
    except Exception:
        pass
    return ""

# ---------- job posting parser (강화) ----------
RESP_KEYS = ["주요 업무","담당 업무","업무","Responsibilities","What you will do","Role","Your role","What you'll do"]
QUAL_KEYS = ["자격 요건","지원 자격","Requirements","Qualifications","Must have","Required","Basic qualifications"]
PREF_KEYS = ["우대","우대사항","Preferred","Nice to have","Plus","Preferred qualifications"]

RESP_HINTS = [
    "업무","담당","책임","역할","Role","Responsibilities","Work you'll do","What you will do","What you'll do",
    "You will","Key responsibilities","미션","Mission"
]
QUAL_HINTS = [
    "자격","요건","필수","필수조건","필수역량","Requirements","Qualifications","Must have","Required",
    "Basic qualifications","조건","경력","학력","필요 스킬","필수 기술","필수 경험","우리가 찾는 인재"
]
PREF_HINTS = [
    "우대","가산점","Preferred","Nice to have","Plus","우대사항","있으면 좋은","우대 역량","가점","선호",
    "Preferred qualifications","Bonus points"
]

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

def _split_bullets(txt: str) -> list[str]:
    arr = re.split(r"[\n\r]+|[•·▪️▶︎\-]\s+|•\s*", txt or "")
    return [a.strip(" -•·▪️▶︎") for a in arr if len(a.strip())>2]

def classify_bullets(lines: list[str]) -> tuple[list[str], list[str], list[str]]:
    resp, qual, pref = [], [], []
    for s in lines:
        t = s.strip(" -•·▪️▶︎").strip()
        if len(t) < 3: 
            continue
        low = t.lower()
        # 우대 → 자격 → 업무 순으로 우선 분류
        if any(k.lower() in low for k in PREF_HINTS):
            pref.append(t);  continue
        if any(k.lower() in low for k in QUAL_HINTS):
            qual.append(t);  continue
        if any(k.lower() in low for k in RESP_HINTS):
            resp.append(t);  continue
        # 신호 기반 휴리스틱
        if re.search(r"(경력\s*\d+|~\s*\d+\s*년|신입|인턴|정규직|학력|전공|자격증|영어|토익|OPIc|JLPT|정보처리기사|Certification|Bachelor|Master|PhD)", t, re.I):
            qual.append(t); continue
        if re.search(r"^(설계|구현|개발|운영|분석|작성|개선|관리|Design|Implement|Build|Operate|Analyze|Lead)\b", t, re.I):
            resp.append(t); continue
        # 기본값: 업무
        resp.append(t)
    def clean(xs):
        out, seen = [], set()
        for x in xs:
            x = _snippetize(x, 200)
            if x and x not in seen:
                seen.add(x); out.append(x)
        return out[:25]
    return clean(resp), clean(qual), clean(pref)

def _find_section_bullets(soup: BeautifulSoup, keys: list[str]) -> list[str]:
    for h in soup.find_all(re.compile("^h[1-4]$")):
        head = _clean_text(h.get_text())
        if any(k.lower() in head.lower() for k in keys):
            bul=[]
            nxt=h.find_next_sibling()
            stop=set(["h1","h2","h3","h4"])
            while nxt and nxt.name not in stop:
                if nxt.name in {"ul","ol"}:
                    for li in nxt.find_all("li"):
                        t=_clean_text(li.get_text(" "))
                        if len(t)>2: bul.append(t)
                elif nxt.name in {"p","div"}:
                    bul += _split_bullets(nxt.get_text(" "))
                nxt=nxt.find_next_sibling()
            if bul: return bul
    body_text = soup.get_text("\n")
    if any(k.lower() in body_text.lower() for k in keys):
        return _split_bullets(body_text)
    return []

def split_by_sections(text: str) -> dict:
    """스냅샷 텍스트에서 섹션 헤더로 큰 덩어리를 먼저 분리"""
    pats = {
        "resp": r"(주요\s*업무|담당\s*업무|Responsibilities?|Role|What you will do|What you'll do)",
        "qual": r"(자격\s*요건|지원\s*자격|Requirements?|Qualifications?|Must\s+have|Required|Basic\s+qualifications?)",
        "pref": r"(우대\s*사항?|Preferred|Nice\s+to\s+have|Plus|Preferred\s+qualifications?)"
    }
    sec = {"resp":"", "qual":"", "pref":""}
    try:
        spans=[]
        for k,pat in pats.items():
            for m in re.finditer(pat, text, re.I):
                spans.append((m.start(), k))
        spans.sort()
        if not spans:
            return sec
        for i,(pos,k) in enumerate(spans):
            end = spans[i+1][0] if i+1<len(spans) else len(text)
            sec[k] += text[pos:end]
    except Exception:
        pass
    return sec

def parse_job_posting(url: str) -> dict:
    out = {"title": None, "responsibilities": [], "qualifications": [], "preferred": [], "company_intro": None}
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code == 200 and "text/html" in r.headers.get("content-type",""):
            soup = BeautifulSoup(r.text, "html.parser")
            if soup.title and soup.title.string:
                out["title"] = _clean_text(soup.title.string)
            meta_desc = soup.find("meta", {"name":"description"}) or soup.find("meta", {"property":"og:description"})
            if meta_desc and meta_desc.get("content"):
                out["company_intro"] = _snippetize(meta_desc["content"], 220)

            # 1) JSON-LD
            jp = _extract_json_ld_job(soup)
            if jp:
                out["title"] = jp.get("title") or out["title"]
                desc = _clean_text(jp.get("description",""))
                if desc:
                    lines = _split_bullets(desc)
                    r1,q1,p1 = classify_bullets(lines)
                    out["responsibilities"] += r1; out["qualifications"] += q1; out["preferred"] += p1

            host = (_domain(url) or "")
            got = lambda: (out["responsibilities"] or out["qualifications"] or out["preferred"])

            # 2) 사이트별 전용
            if "wanted.co.kr" in host and not got():
                nd = soup.find("script", id="__NEXT_DATA__")
                if nd and nd.string:
                    try:
                        data = json.loads(nd.string)
                        text_fields=[]
                        def walk(x):
                            if isinstance(x, dict):
                                for k,v in x.items():
                                    if isinstance(v, (dict, list)):
                                        walk(v)
                                    elif isinstance(v, str) and len(v) > 20:
                                        if any(tk in k.lower() for tk in ["description","requirement","qualification","preference","responsibil"]):
                                            text_fields.append(v)
                            elif isinstance(x, list):
                                for it in x: walk(it)
                        walk(data)
                        lines=[]
                        for t in text_fields: lines += _split_bullets(t)
                        r2,q2,p2 = classify_bullets(lines)
                        out["responsibilities"] += r2; out["qualifications"] += q2; out["preferred"] += p2
                    except Exception:
                        pass

            if "saramin.co.kr" in host and not got():
                body = soup.select_one("#job_summary, .user_content, .wrap_jview, .content")
                if body:
                    lines = _split_bullets(body.get_text("\n"))
                    r3,q3,p3 = classify_bullets(lines)
                    out["responsibilities"] += r3; out["qualifications"] += q3; out["preferred"] += p3

            if "jobkorea.co.kr" in host and not got():
                body = soup.select_one("#tab02, .detailArea, .recruitMent, .smartApply, .viewContents")
                if body:
                    lines = _split_bullets(body.get_text("\n"))
                    r4,q4,p4 = classify_bullets(lines)
                    out["responsibilities"] += r4; out["qualifications"] += q4; out["preferred"] += p4

            if "rocketpunch.com" in host and not got():
                body = soup.select_one(".job-detail, .description, .job-detail__full")
                if body:
                    lines = _split_bullets(body.get_text("\n"))
                    r5,q5,p5 = classify_bullets(lines)
                    out["responsibilities"] += r5; out["qualifications"] += q5; out["preferred"] += p5

            # 3) 일반 섹션 헤더
            if not got():
                body_text = soup.get_text("\n")
                # 헤더 분리 먼저
                sec = split_by_sections(body_text)
                if sec.get("resp"): out["responsibilities"] += _split_bullets(sec["resp"])
                if sec.get("qual"): out["qualifications"]   += _split_bullets(sec["qual"])
                if sec.get("pref"): out["preferred"]        += _split_bullets(sec["pref"])
                # 그래도 부족하면 전체를 분류기로
                if not got():
                    lines = _split_bullets(body_text)
                    r6,q6,p6 = classify_bullets(lines)
                    out["responsibilities"] += r6; out["qualifications"] += q6; out["preferred"] += p6

        # 4) 스냅샷 폴백(섹션 헤더 → 불릿 분류)
        if not (out["responsibilities"] or out["qualifications"] or out["preferred"]):
            snap = fetch_text_snapshot(url)
            if snap:
                sec = split_by_sections(snap)
                if sec.get("resp"): out["responsibilities"] += _split_bullets(sec["resp"])
                if sec.get("qual"): out["qualifications"]   += _split_bullets(sec["qual"])
                if sec.get("pref"): out["preferred"]        += _split_bullets(sec["pref"])
                if not (out["responsibilities"] or out["qualifications"] or out["preferred"]):
                    r7,q7,p7 = classify_bullets(_split_bullets(snap))
                    out["responsibilities"] += r7; out["qualifications"] += q7; out["preferred"] += p7

        # 5) <li> 폴백(최후)
        if not (out["responsibilities"] or out["qualifications"] or out["preferred"]):
            try:
                soup2 = BeautifulSoup(r.text, "html.parser")
                lis = [ _clean_text(li.get_text(" ")) for li in soup2.find_all("li") ]
                r8,q8,p8 = classify_bullets(lis)
                out["responsibilities"] += r8; out["qualifications"] += q8; out["preferred"] += p8
            except Exception:
                pass

        def dedup(xs):
            seen, out2 = set(), []
            for x in xs:
                x = _snippetize(x, 200)
                if x and x not in seen:
                    seen.add(x); out2.append(x)
            return out2[:25]
        out["responsibilities"] = dedup(out["responsibilities"])
        out["qualifications"]   = dedup(out["qualifications"])
        out["preferred"]        = dedup(out["preferred"])

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
    with st.expander("디버그: 키/버전"):
        try:
            import openai as _o; ov = getattr(_o, "__version__", "?")
        except Exception:
            ov = "?"
        st.write({"api_key": bool(API_KEY), "naver_keys": bool(NAVER_ID and NAVER_SECRET), "openai": ov})

if not API_KEY:
    st.error("OpenAI API Key가 필요합니다."); st.stop()
client = OpenAI(api_key=API_KEY, timeout=30.0)

# ==========================================================
# 입력
# ==========================================================
st.subheader("① 회사/직무 입력")
company_name = st.text_input("회사 이름", placeholder="예: 네이버 / Kakao / 삼성SDS")
homepage     = st.text_input("공식 홈페이지 URL(선택)", placeholder="https://...")
role_title   = st.text_input("지원 직무명", placeholder="데이터 애널리스트 / ML 엔지니어 ...")
job_url_in   = st.text_input("채용 공고 URL(선택) — 없다면 자동 탐색")

# 세션 상태
defaults = {
    "company_state": {},
    "history": [],
    "current_question": "",
    "answer_text": "",
    "rag_store": {"chunks": [], "embeds": None},
}
for k,v in defaults.items():
    if k not in st.session_state: st.session_state[k]=v

def build_company(name: str, home: str|None, role: str|None, job_url: str|None) -> dict:
    site = fetch_site_snippets(home) if home else {"values":[], "recent":[], "about":None}
    urls = [job_url] if job_url else discover_job_urls(name, role or "", home, limit=5)
    jp = parse_job_posting(urls[0]) if urls else {"title":None,"responsibilities":[],"qualifications":[],"preferred":[],"company_intro":None}
    news = naver_search_news(name, display=6) or []
    return {
        "company_name": name.strip() or "(회사명 미설정)",
        "homepage": home or None,
        "values": site.get("values", []),
        "recent": site.get("recent", []),
        "company_intro_site": site.get("about"),
        "role": role or "",
        "job_url": urls[0] if urls else (job_url or None),
        # 원문 그대로
        "role_responsibilities": jp.get("responsibilities", []),
        "role_qualifications":   jp.get("qualifications", []),
        "role_preferred":        jp.get("preferred", []),
        "news": news
    }

def summarize_intro_only(c: dict) -> str:
    ctx = textwrap.dedent(f"""
    [홈페이지 소개(발췌)] {c.get('company_intro_site') or ''}
    [최근 뉴스] {', '.join([_snippetize(n['title'],70) for n in c.get('news',[])[:3]])}
    """).strip()
    sys = "아래 정보를 바탕으로 '회사 소개'만 2~3문장 한국어 요약으로 작성하라. 광고성 문구는 배제하고 사실 위주로."
    user = f"{ctx}\n\n[회사명] {c.get('company_name','')}"
    try:
        r = client.chat.completions.create(model=MODEL, temperature=0.2,
                                           messages=[{"role":"system","content":sys},{"role":"user","content":user}])
        return r.choices[0].message.content.strip()
    except Exception:
        return c.get("company_intro_site") or "회사 소개 정보가 충분하지 않습니다."

if st.button("회사/직무 정보 불러오기", type="primary"):
    if not company_name.strip():
        st.warning("회사 이름을 입력해 주세요.")
    else:
        with st.spinner("회사/공고/뉴스 수집 중..."):
            cobj = build_company(company_name, homepage or None, role_title or None, job_url_in or None)
            intro = summarize_intro_only(cobj)
            st.session_state.company_state={"company":cobj, "intro":intro}
            # 하단 초기화
            st.session_state.current_question=""
            st.session_state.answer_text=""
            st.session_state.history=[]
            st.session_state.rag_store={"chunks":[],"embeds":None}
        st.success("회사 정보 갱신 및 결과 초기화 완료")

company = st.session_state.get("company_state",{}).get("company")
intro   = st.session_state.get("company_state",{}).get("intro")

# ==========================================================
# ② 회사 요약 / 채용 요건 (원문)
# ==========================================================
st.subheader("② 회사 요약 / 채용 요건")
if company and intro:
    st.markdown(f"**회사명**: {company['company_name']}")
    st.markdown("**회사 소개(요약)**")
    st.markdown(intro)
    cols = st.columns(2)
    with cols[0]:
        if company.get("homepage"): st.link_button("홈페이지 열기", company["homepage"])
    with cols[1]:
        if company.get("job_url"):  st.link_button("채용 공고 열기", company["job_url"])
    st.markdown("---")
    st.markdown(f"**모집 분야**: {company.get('role') or '—'}")
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown("**주요 업무(원문)**")
        arr = company.get("role_responsibilities") or []
        st.markdown("- " + "\n- ".join(arr) if arr else "_공고에서 추출된 주요 업무가 없습니다._")
    with c2:
        st.markdown("**자격 요건(원문)**")
        arr = company.get("role_qualifications") or []
        st.markdown("- " + "\n- ".join(arr) if arr else "_공고에서 추출된 자격 요건이 없습니다._")
    with c3:
        st.markdown("**우대 사항(원문)**")
        arr = company.get("role_preferred") or []
        st.markdown("- " + "\n- ".join(arr) if arr else "_공고에서 추출된 우대 사항이 없습니다._")
    with st.expander("디버그: 공고 추출 상태"):
        st.write({
            "job_url": company.get("job_url"),
            "resp_cnt": len(company.get("role_responsibilities") or []),
            "qual_cnt": len(company.get("role_qualifications") or []),
            "pref_cnt": len(company.get("role_preferred") or []),
        })
else:
    st.info("위 입력 후 ‘회사/직무 정보 불러오기’를 눌러 주세요.")

# ==========================================================
# ③ 질문 생성
# ==========================================================
st.subheader("③ 질문 생성")

def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts: return np.zeros((0,1536), dtype=np.float32)
    r = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([d.embedding for d in r.data], dtype=np.float32)

with st.expander("RAG 옵션(선택)"):
    rag_on = st.toggle("회사 문서 기반 질문/코칭 사용", value=True, key="rag_on")
    topk = st.slider("검색 상위 K", 1, 8, 4, 1, key="topk")
    ups = st.file_uploader("회사 문서 업로드 (TXT/MD/PDF, 여러 파일)", type=["txt","md","pdf"], accept_multiple_files=True)
    size = st.slider("청크 길이", 400, 2000, 900, 100)
    ovlp = st.slider("오버랩", 0, 400, 150, 10)
    if ups:
        with st.spinner("문서 인덱싱 중..."):
            chunks=[]
            for u in ups:
                t = read_file_to_text(u)
                if t: chunks += chunk_text(t, size, ovlp)
            if chunks:
                embs = embed_texts(chunks)
                st.session_state.rag_store={"chunks":chunks, "embeds":embs}
                st.success(f"청크 {len(chunks)}개 인덱싱 완료")

def cosine_topk(mat: np.ndarray, q: np.ndarray, k: int=4):
    if mat.size==0: return np.array([]), np.array([],dtype=int)
    mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True)+1e-12)
    qn = q / (np.linalg.norm(q, axis=1, keepdims=True)+1e-12)
    sims = mn @ qn.T
    sims = sims.reshape(-1)
    idx = np.argsort(-sims)[:k]
    return sims[idx], idx

def retrieve_supports(qtext: str, k: int):
    store = st.session_state.rag_store
    chs, embs = store.get("chunks", []), store.get("embeds")
    if not st.session_state.get("rag_on") or embs is None or not chs: return []
    qv = embed_texts([qtext])
    s, idx = cosine_topk(embs, qv, k=k)
    return [("회사자료", float(sc), chs[int(i)]) for sc,i in zip(s, idx)]

TYPE_INSTRUCTIONS = {
    "행동(STAR)": "과거 실무 사례를 이끌어내는 STAR",
    "기술 심층": "성능/비용/지연/정확도/운영을 포함한 기술 심층",
    "핵심가치 적합성": "태도/가치/협업 스타일 검증",
    "역질문": "지원자가 회사를 평가하는 역질문",
}

def build_ctx(c: dict|None) -> str:
    if not c: return ""
    news = ", ".join([_snippetize(n["title"],70) for n in c.get("news",[])[:3]])
    return textwrap.dedent(f"""
    [회사명] {c.get('company_name','')}
    [모집 분야] {c.get('role','')}
    [주요 업무] {", ".join(c.get('role_responsibilities', [])[:6])}
    [자격 요건] {", ".join(c.get('role_qualifications', [])[:6])}
    [핵심가치] {", ".join(c.get('values', [])[:6])}
    [최근 뉴스] {news}
    """).strip()

def _sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def pick_diverse(cands: list[str], hist: list[str]) -> str:
    if not cands: return ""
    if not hist: return random.choice(cands)
    best=None; best_s=1e9
    for q in cands:
        sims=[_sim(q,h) for h in hist] or [0.0]
        s=(sum(sims)/len(sims)) + 0.35*np.std(sims)
        if s<best_s: best_s=s; best=q
    return best

q_type = st.selectbox("질문 유형", list(TYPE_INSTRUCTIONS.keys()))
level  = st.selectbox("난이도/연차", ["주니어","미들","시니어"])
hint   = st.text_input("질문 생성 힌트(선택)", placeholder="예: 전환 퍼널 / 모델 성능-비용 / 데이터 품질")

if "history" not in st.session_state: st.session_state.history=[]
if "current_question" not in st.session_state: st.session_state.current_question=""

if st.button("새 질문 받기", type="primary", use_container_width=True):
    st.session_state.answer_text=""
    try:
        sups=[]
        if st.session_state.get("rag_on"):
            base = hint.strip() or (company.get("role","") if company else "")
            sups = retrieve_supports(base, st.session_state.get("topk",4))
        ctx = build_ctx(company)
        focuses=[]
        if company:
            focuses += company.get("role_responsibilities", [])[:6] + company.get("role_qualifications", [])[:6]
        for _,_,txt in (sups or [])[:3]:
            focuses += [t.strip() for t in re.split(r"[•\-\n\.]", txt) if 6<len(t.strip())<100][:3]
        focuses = [f for f in focuses if f]
        random.shuffle(focuses)
        focuses = focuses[:4]
        sys = f"""너는 '{q_type}' 유형({TYPE_INSTRUCTIONS[q_type]})의 질문 6개를 한국어로 생성하라.
각 질문은 포커스 키워드를 1개 이상 포함하고 형태/관점/키워드가 서로 다르게 하라. 난이도 {level}.
포맷: 1) ... 2) ... 3) ... (한 줄씩)"""
        user = f"[컨텍스트]\n{ctx}\n[포커스]\n- " + "\n- ".join(focuses)
        r = client.chat.completions.create(model=MODEL, temperature=0.95,
                                           messages=[{"role":"system","content":sys},{"role":"user","content":user}])
        raw=r.choices[0].message.content.strip()
        cands=[re.sub(r'^\s*\d+\)\s*','',ln).strip() for ln in raw.splitlines() if re.match(r'^\s*\d+\)', ln)]
        if not cands: cands=[ln.strip("- ").strip() for ln in raw.splitlines() if ln.strip()][:6]
        hist=[h["question"] for h in st.session_state.history][-10:]
        st.session_state.current_question = pick_diverse(cands, hist) or cands[0]
        st.session_state.last_supports_q = sups
    except Exception as e:
        st.error(f"질문 생성 오류: {e}")

st.text_area("질문", height=110, value=st.session_state.get("current_question",""))

# ==========================================================
# ④ 코칭/채점 — 자동 루브릭 적용 & 점수 일원화
# ==========================================================
st.subheader("④ 나의 답변 / 코칭")
AXES = ["문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치"]

KEYMAP = {
    "문제정의":  [r"문제|가설|목표|제약|SLA|KPI|요구사항|트레이드오프|선택"],
    "데이터/지표":[r"지표|데이터|통계|A/B|실험|정확|재현|검정|샘플|가설|피처|모델|모니터링|로그|메트릭|ROC|리콜|정밀|F1|신뢰|품질|데이터 품질|CI|표본"],
    "실행력/주도성":[r"설계|아키텍처|구현|배포|파이프라인|운영|장애|복구|스케일|성능|튜닝|리드|주도|Flink|Spark|Kafka|Airflow|dbt|ETL"],
    "협업/커뮤니케이션":[r"협업|커뮤니케이션|조율|합의|이해관계자|문서|런북|데이터 계약|RFC|PR|리뷰|조정"],
    "고객가치":  [r"비용|ROI|수익|전환|리텐션|NPS|만족|규정|보안|개인정보|리스크|비즈니스|가치|임팩트|효과"],
}

RUBRIC_DESC = {
    "문제정의": "목표·제약(SLA/비용/지연/정확도) 명확화, 트레이드오프 정의, 선택 기준",
    "데이터/지표":"정량 지표/실험설계/품질 검증, 재현성, 모니터링",
    "실행력/주도성":"아키텍처/구현/운영/장애대응/스케일링·비용 최적화",
    "협업/커뮤니케이션":"요구사항 수집, 이해관계자 조율, 문서화/런북, 데이터 계약",
    "고객가치":"비즈니스 임팩트·비용/리스크/규정/보안·개인정보 고려",
}

def detect_axes_from_question(q: str) -> list[bool]:
    ql = q.lower()
    applies=[False]*5
    for i,axis in enumerate(AXES):
        pats=KEYMAP[axis]
        if any(re.search(p, ql, re.I) for p in pats):
            applies[i]=True
    if sum(applies)==0: applies=[True,False,False,False,True]  # 최소 보장
    return applies

def coach(company: dict|None, question: str, answer: str, supports, qtype: str) -> dict:
    ctx = build_ctx(company)
    applies = detect_axes_from_question(question)
    apply_text = ", ".join([f"{AXES[i]}({'O' if applies[i] else '-'})" for i in range(5)])
    details = "\n".join([f"- {k}: {RUBRIC_DESC[k]}" for k,a in zip(AXES, applies) if a])
    rag=""
    if supports:
        rag = "\n[회사 근거 문서 발췌]\n" + "\n".join([f"- ({s:.3f}) {txt[:300]}" for _,s,txt in supports]) + "\n"
    sys = f"""너는 한국어 면접 코치다. 아래 형식만 출력하라.
1) 총점: NN/100  # 이 줄은 반드시 첫 3줄 안에 위치
2) 강점: 2~3개 불릿
3) 리스크: 2~3개 불릿
4) 개선 포인트: 3개 불릿(행동·지표·임팩트 중심)
5) 수정본 답변: STAR(상황-과제-행동-성과) 구조로 간결하게
6) 역량 점수(각 0~20, 비적용은 '-' 그대로): [{', '.join(AXES)}] — 이 순서로 5개 값을 쉼표로 출력
채점은 '적용 축'에 한해 위 루브릭을 적용하고, 비적용 축은 '-'로 둔다."""
    user = f"""[컨텍스트]\n{ctx}\n{rag}[질문 유형] {qtype}\n[적용 축]\n{apply_text}\n[루브릭]\n{details}\n\n[면접 질문]\n{question}\n\n[후보자 답변]\n{answer}"""
    r = client.chat.completions.create(model=MODEL, temperature=0.35,
                                       messages=[{"role":"system","content":sys},{"role":"user","content":user}])
    content = r.choices[0].message.content.strip()

    # 역량 파싱
    last = content.splitlines()[-1]
    toks = [t.strip() for t in re.split(r"[,\s]+", last) if t.strip()!=""]
    comps: list[Optional[int]]=[]
    for t in toks[:5]:
        if t in ["-","–","—"]: comps.append(None)
        elif re.fullmatch(r"\d{1,2}", t): comps.append(max(0,min(20,int(t))))
        else: comps.append(None)
    while len(comps)<5: comps.append(None)

    # 총점(우리 계산): 적용축 평균 ×5
    used=[v for v,a in zip(comps, applies) if a and isinstance(v,int)]
    score = round(sum(used)/len(used)*5) if used else 0

    # 본문 내 총점 라인 교체
    lines=content.splitlines(); repl=False
    for i,L in enumerate(lines[:3]):
        if "총점" in L:
            lines[i]=re.sub(r"총점\s*:\s*\d{1,3}\s*/\s*100", f"총점: {score}/100", L) if re.search(r"총점\s*:", L) else f"총점: {score}/100"
            repl=True; break
    if not repl:
        lines.insert(0, f"총점: {score}/100")
    content_fixed="\n".join(lines)

    return {"raw": content_fixed, "score": score, "competencies": comps, "applies": applies}

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
            res = coach(company, st.session_state["current_question"], st.session_state.answer_text, sups, q_type)
            st.session_state.history.append({
                "ts": pd.Timestamp.now(),
                "question": st.session_state["current_question"],
                "user_answer": st.session_state.answer_text,
                "score": res["score"],
                "feedback": res["raw"],
                "supports": sups,
                "competencies": res["competencies"],
                "applies": res["applies"],
            })

# ---------- 결과 ----------
st.divider()
st.subheader("피드백 결과")
if st.session_state.history:
    last = st.session_state.history[-1]
    total = last["score"]
    c1,c2 = st.columns([1,3])
    with c1:
        st.metric("총점(/100)", total)
    with c2:
        st.markdown(f"**총점(시스템 산출)**: {total}/100")
        st.markdown(last["feedback"])
else:
    st.info("아직 결과가 없습니다.")

# ---------- 레이더 ----------
st.divider()
st.subheader("역량 레이더 (세션 누적)")
AX = ["문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치"]

def comp_df(hist):
    rows=[]
    for h in hist:
        vals = h.get("competencies")
        if not vals: continue
        rows.append([np.nan if v is None else float(v) for v in vals])
    if not rows: return None
    df = pd.DataFrame(rows, columns=AX)
    df["합계"] = df[AX].sum(axis=1, skipna=True)
    return df

cdf = comp_df(st.session_state.history)
if cdf is not None:
    means = cdf[AX].mean(axis=0, skipna=True).tolist()
    radar = [0 if np.isnan(x) else float(x) for x in means]
    if PLOTLY_OK:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=radar+[radar[0]], theta=AX+[AX[0]], fill='toself'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,20])), showlegend=False, height=420)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(pd.DataFrame({"score": radar}, index=AX))
    st.dataframe(cdf.fillna("-"), use_container_width=True)
else:
    st.caption("아직 역량 점수가 없습니다.")

# ---------- CSV ----------
st.divider()
st.subheader("세션 리포트 (CSV)")
def build_report(hist):
    rows=[]
    for h in hist:
        row={"timestamp":h["ts"],"question":h["question"],"user_answer":h["user_answer"],
             "score":h["score"],"feedback_raw":h["feedback"]}
        comps=h.get("competencies") or []
        for k,v in zip(AX, comps): row[f"comp_{k}"]=("-" if v is None else v)
        row["comp_sum"]=sum([v for v in comps if isinstance(v,int)]) if comps else 0
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["timestamp","question","user_answer","score","feedback_raw"])
rep = build_report(st.session_state.history)
st.download_button("CSV 다운로드", data=rep.to_csv(index=False).encode("utf-8-sig"),
                   file_name="interview_session_report.csv", mime="text/csv")

st.caption("속도 최적화: 상세 URL 우선 선택, HTTP 캐시, '회사 소개만' 요약(토큰 절감), RAG 조건부 실행")
