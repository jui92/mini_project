###############################################################################################
# Job_Helper_Bot (정밀 크롤러 확장판)
#  - 채용 공고 URL → 회사 요약(정규화) → 이력서 업로드/인덱싱 → 자소서 생성
#  - Wanted / Saramin / JobKorea 사이트별 맞춤 파서(정밀 크롤러) 추가
#
# 변경 요약
#   1) parse_portal_specific(url, soup, raw_text) 디스패처
#   2) parse_wanted / parse_saramin / parse_jobkorea 규칙 추출
#   3) 규칙 결과 우선 사용, 부족 시 LLM 정제로 보완
###############################################################################################

# -*- coding: utf-8 -*-
import os, re, json, urllib.parse, random, time, io
from typing import Optional, Tuple, Dict, List

import requests
from bs4 import BeautifulSoup
import html2text
import streamlit as st
import pandas as pd
import numpy as np

# ================== 기본 설정 ==================
st.set_page_config(page_title="Job_Helper_Bot (자소서 생성)", page_icon="📑", layout="wide")
st.title("Job_Helper_Bot : 채용 공고 URL → 회사 요약 → 이력서 등록 → 자소서 생성")

# ================== OpenAI ==================
try:
    from openai import OpenAI
except ImportError:
    st.error("`openai` 패키지가 필요합니다. requirements.txt에 openai를 추가하세요.")
    st.stop()

API_KEY = os.getenv("OPENAI_API_KEY") or (st.secrets.get("OPENAI_API_KEY", None) if hasattr(st,"secrets") else None)
if not API_KEY:
    API_KEY = st.text_input("OPENAI_API_KEY 입력", type="password")
if not API_KEY:
    st.stop()
client = OpenAI(api_key=API_KEY)

with st.sidebar:
    st.subheader("모델 설정")
    CHAT_MODEL = st.selectbox("대화/생성 모델", ["gpt-4o-mini","gpt-4o"], index=0)
    EMBED_MODEL = st.selectbox("임베딩 모델(내부용)", ["text-embedding-3-small","text-embedding-3-large"], index=0)

# ================== HTTP 유틸 ==================
def normalize_url(u: str) -> Optional[str]:
    if not u: return None
    u = u.strip()
    if not re.match(r"^https?://", u): u = "https://" + u
    parts = urllib.parse.urlsplit(u)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))

def http_get(url: str, timeout: int = 12) -> Optional[requests.Response]:
    try:
        r = requests.get(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
                "Accept-Language": "ko, en;q=0.9",
            },
            timeout=timeout,
        )
        if r.status_code == 200 and "text/html" in r.headers.get("content-type",""):
            return r
    except Exception:
        pass
    return None

# ================== 원문 수집 (Jina → Web → BS4) ==================
def fetch_jina_text(url: str, timeout: int = 15) -> str:
    """Jina reader 프록시를 통해 정적 텍스트를 우선 확보 (동적 로딩 회피용)"""
    try:
        parts = urllib.parse.urlsplit(url)
        prox = f"https://r.jina.ai/http://{parts.netloc}{parts.path}"
        if parts.query: prox += f"?{parts.query}"
        r = http_get(prox, timeout=timeout)
        return r.text.strip() if r else ""
    except Exception:
        return ""

def html_to_text(html_str: str) -> str:
    conv = html2text.HTML2Text()
    conv.ignore_links = True
    conv.ignore_images = True
    conv.body_width = 0
    txt = conv.handle(html_str)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt.strip()

def fetch_webbase_text(url: str) -> str:
    r = http_get(url, timeout=12)
    if not r: return ""
    return html_to_text(r.text)

def fetch_bs4_text(url: str) -> Tuple[str, Optional[BeautifulSoup]]:
    r = http_get(url, timeout=12)
    if not r: return "", None
    soup = BeautifulSoup(r.text, "lxml")
    # 큰 블록을 우선 합치는 전략
    blocks = []
    for sel in ["article","section","main","div","ul","ol"]:
        for el in soup.select(sel):
            txt = el.get_text(" ", strip=True)
            if txt and len(txt) > 300:
                txt = re.sub(r"\s+"," ", txt)
                blocks.append(txt)
    if not blocks:
        return soup.get_text(" ", strip=True)[:120000], soup
    seen, out = set(), []
    for b in blocks:
        if b not in seen:
            seen.add(b); out.append(b)
    return ("\n\n".join(out)[:120000], soup)

def fetch_all_text(url: str):
    url = normalize_url(url)
    if not url: return "", {"error":"invalid_url"}, None
    jina = fetch_jina_text(url)
    if jina:
        _, soup = fetch_bs4_text(url)
        return jina, {"source":"jina","len":len(jina),"url_final":url}, soup
    web = fetch_webbase_text(url)
    if web:
        _, soup = fetch_bs4_text(url)
        return web, {"source":"webbase","len":len(web),"url_final":url}, soup
    bs, soup = fetch_bs4_text(url)
    return bs, {"source":"bs4","len":len(bs),"url_final":url}, soup

# ================== 공통 유틸 (클리닝/섹션 수집) ==================
PREF_KW = re.compile(r"(우대|선호|preferred|plus|가산점|있으면\s*좋음)", re.I)
RESP_HDR = re.compile(r"(주요\s*업무|담당\s*업무|Role|Responsibilities?)", re.I)
QUAL_HDR = re.compile(r"(자격\s*요건|지원\s*자격|Requirements?|Qualifications?)", re.I)
PREF_HDR = re.compile(r"(우대\s*사항|우대|Preferred|Nice\s*to\s*have|Plus)", re.I)

def _clean_line(s: str) -> str:
    s = re.sub(r"\s+"," ", s or "").strip(" -•·▶▪️").strip()
    return s[:180]

def _push_unique(bucket: List[str], line: str, seen: set):
    line = _clean_line(line)
    if line and line not in seen:
        seen.add(line); bucket.append(line)

def extract_company_meta(soup: Optional[BeautifulSoup]) -> Dict[str,str]:
    """페이지 메타에서 회사명/소개/직무명 힌트 추출"""
    meta = {"company_name":"","company_intro":"","job_title":""}
    if not soup: return meta
    cand = []
    og = soup.find("meta", {"property":"og:site_name"})
    if og and og.get("content"): cand.append(og["content"])
    app = soup.find("meta", {"name":"application-name"})
    if app and app.get("content"): cand.append(app["content"])
    if soup.title and soup.title.string: cand.append(soup.title.string)
    cand = [re.split(r"[\-\|\·\—]", c)[0].strip() for c in cand if c]
    cand = [c for c in cand if 2 <= len(c) <= 40]
    meta["company_name"] = cand[0] if cand else ""

    md = soup.find("meta", {"name":"description"}) or soup.find("meta", {"property":"og:description"})
    if md and md.get("content"):
        meta["company_intro"] = re.sub(r"\s+"," ", md["content"]).strip()[:500]

    jt = ""
    ogt = soup.find("meta", {"property":"og:title"})
    if ogt and ogt.get("content"): jt = ogt["content"]
    if not jt:
        h1 = soup.find("h1")
        if h1 and h1.get_text(): jt = h1.get_text(strip=True)
    if not jt:
        h2 = soup.find("h2")
        if h2 and h2.get_text(): jt = h2.get_text(strip=True)
    meta["job_title"] = re.sub(r"\s+"," ", jt).strip()[:120]
    return meta

def collect_after_heading(soup: BeautifulSoup, head_regex: re.Pattern, limit: int = 12) -> List[str]:
    """
    h1~h4 중 제목에 정규식이 매칭되는 요소를 찾고,
    그 '다음 형제들'과 '바로 하위 ul/ol/li/p'에서 문장을 모아 리스트로 반환.
    """
    out, seen = [], set()
    heads = []
    for tag in soup.find_all(re.compile("^h[1-4]$")):
        if head_regex.search(tag.get_text(" ", strip=True) or ""):
            heads.append(tag)

    for h in heads:
        # 형제 순회
        sib = h.find_next_sibling()
        while sib and sib.name not in {"h1","h2","h3","h4"} and len(out) < limit:
            # 목록/문단 우선
            if sib.name in {"ul","ol"}:
                for li in sib.find_all("li", recursive=True):
                    _push_unique(out, li.get_text(" ", strip=True), seen)
                    if len(out) >= limit: break
            elif sib.name in {"p","div","section"}:
                txt = sib.get_text(" ", strip=True)
                if len(txt) > 4:
                    # 큰 덩어리면 문장 분리
                    lines = re.split(r"[•\-\n·▪️▶]+|\s{2,}", txt)
                    for l in lines:
                        _push_unique(out, l, seen)
                        if len(out) >= limit: break
            sib = sib.find_next_sibling()
        if len(out) >= limit: break

        # 바로 하위 목록/문단
        for sel in ["ul","ol","p","div","section"]:
            for el in h.find_all(sel, recursive=False):
                text = el.get_text(" ", strip=True)
                if sel in {"ul","ol"}:
                    for li in el.find_all("li", recursive=True):
                        _push_unique(out, li.get_text(" ", strip=True), seen)
                        if len(out) >= limit: break
                else:
                    lines = re.split(r"[•\-\n·▪️▶]+|\s{2,}", text)
                    for l in lines:
                        _push_unique(out, l, seen)
                        if len(out) >= limit: break
            if len(out) >= limit: break

    return out[:limit]

# ================== 사이트별 정밀 파서 ==================
def parse_wanted(soup: BeautifulSoup) -> Dict[str, List[str]]:
    """
    Wanted (wanted.co.kr) 추출 규칙:
      - h2/h3 제목에 '주요업무/자격요건/우대사항'이 자주 노출
      - data-cy 속성이나 동적 클래스는 변동 가능 → 제목기반 수집
    """
    res  = collect_after_heading(soup, RESP_HDR, limit=16)
    qual = collect_after_heading(soup, QUAL_HDR, limit=16)
    pref = collect_after_heading(soup, PREF_HDR, limit=16)

    # 보정: 자격요건 내 우대 키워드가 섞였으면 분리
    remain = []
    for q in qual:
        if PREF_KW.search(q): pref.append(q)
        else: remain.append(q)
    qual = remain

    return {
        "responsibilities": res[:12],
        "qualifications":   qual[:12],
        "preferences":      pref[:12],
    }

def parse_saramin(soup: BeautifulSoup) -> Dict[str, List[str]]:
    """
    Saramin (saramin.co.kr) 추출 규칙:
      - 상세본문에 표/정의목록(dl/dt/dd) 또는 h2/h3 섹션+ul/li가 혼용
      - 헤딩 텍스트 기반 + 정의목록의 제목 매칭 시 dd 텍스트 수집
    """
    out = {"responsibilities": [], "qualifications": [], "preferences": []}
    # 1) 표준 헤더 기반
    out["responsibilities"] += collect_after_heading(soup, RESP_HDR, 16)
    out["qualifications"]   += collect_after_heading(soup, QUAL_HDR, 16)
    out["preferences"]      += collect_after_heading(soup, PREF_HDR, 16)

    # 2) dl/dt/dd 구조 보조
    for dl in soup.find_all("dl"):
        for dt in dl.find_all("dt", recursive=False):
            title = (dt.get_text(" ", strip=True) or "")
            dd = dt.find_next_sibling("dd")
            if not dd: continue
            text = dd.get_text(" ", strip=True)
            if not text: continue
            lines = re.split(r"[•\-\n·▪️▶]+|\s{2,}", text)
            if RESP_HDR.search(title):
                for l in lines: _push_unique(out["responsibilities"], l, set(out["responsibilities"]))
            elif QUAL_HDR.search(title):
                for l in lines: _push_unique(out["qualifications"], l, set(out["qualifications"]))
            elif PREF_HDR.search(title) or PREF_KW.search(title):
                for l in lines: _push_unique(out["preferences"], l, set(out["preferences"]))

    # 자격요건 내 우대 키워드 보정
    remain = []
    for q in out["qualifications"]:
        if PREF_KW.search(q): out["preferences"].append(q)
        else: remain.append(q)
    out["qualifications"] = remain

    # 상한 제한/중복 제거
    for k in out:
        seen=set(); clean=[]
        for s in out[k]:
            s=_clean_line(s)
            if s and s not in seen:
                seen.add(s); clean.append(s)
        out[k] = clean[:12]
    return out

def parse_jobkorea(soup: BeautifulSoup) -> Dict[str, List[str]]:
    """
    JobKorea (jobkorea.co.kr) 추출 규칙:
      - 상세페이지에 '상세요강/지원자격/우대사항' 섹션이 많음
      - h2/h3 헤더 + 인접 ul/li 또는 div/p 수집
    """
    res  = collect_after_heading(soup, re.compile(r"(상세\s*요강|주요\s*업무|담당\s*업무|Responsibilities?)", re.I), 16)
    qual = collect_after_heading(soup, re.compile(r"(지원\s*자격|자격\s*요건|Requirements?|Qualifications?)", re.I), 16)
    pref = collect_after_heading(soup, re.compile(r"(우대\s*사항|우대|Preferred|Plus)", re.I), 16)

    remain=[]
    for q in qual:
        if PREF_KW.search(q): pref.append(q)
        else: remain.append(q)
    qual=remain

    return {
        "responsibilities": res[:12],
        "qualifications":   qual[:12],
        "preferences":      pref[:12],
    }

def parse_portal_specific(url: str, soup: Optional[BeautifulSoup], raw_text: str) -> Dict[str, List[str]]:
    """
    URL 도메인으로 파서 분기. soup이 없을 땐 raw_text로 최소 대응(헤더 키워드 기반).
    반환: {"responsibilities":[...], "qualifications":[...], "preferences":[...]}
    """
    out = {"responsibilities":[], "qualifications":[], "preferences":[]}
    if not soup:
        # soup 없으면 raw_text를 라인 스캔 (최소 보장)
        lines = [ _clean_line(x) for x in (raw_text or "").split("\n") if x.strip() ]
        bucket = None
        for l in lines:
            if RESP_HDR.search(l): bucket="responsibilities"; continue
            if QUAL_HDR.search(l): bucket="qualifications"; continue
            if PREF_HDR.search(l) or PREF_KW.search(l): bucket="preferences"; continue
            if bucket:
                _push_unique(out[bucket], l, set(out[bucket]))
        return out

    host = urllib.parse.urlsplit(normalize_url(url) or "").netloc.lower()
    if "wanted.co.kr" in host:
        out = parse_wanted(soup)
    elif "saramin.co.kr" in host:
        out = parse_saramin(soup)
    elif "jobkorea.co.kr" in host:
        out = parse_jobkorea(soup)
    else:
        # 기타 포털/자사 채용: 기본 헤더 기반
        out["responsibilities"] = collect_after_heading(soup, RESP_HDR, 16)
        out["qualifications"]   = collect_after_heading(soup, QUAL_HDR, 16)
        out["preferences"]      = collect_after_heading(soup, PREF_HDR, 16)

    # 안전 보정: 자격요건에 우대 포함된 경우
    remain=[]
    for q in out.get("qualifications", []):
        if PREF_KW.search(q): out["preferences"].append(q)
        else: remain.append(q)
    out["qualifications"] = remain

    # 중복/길이 제한
    for k in out:
        seen=set(); clean=[]
        for s in out[k]:
            s=_clean_line(s)
            if s and s not in seen:
                seen.add(s); clean.append(s)
        out[k]=clean[:12]
    return out

# ================== LLM 정제 (채용 공고 → 구조 JSON) ==================
PROMPT_SYSTEM_STRUCT = (
    "너는 채용 공고를 깔끔하게 구조화하는 보조원이다. "
    "입력 텍스트는 포털 광고 문구, UI잔재, 복수 직무가 섞여 있을 수 있다. "
    "한국어로 간결하고 중복없이 정제하라."
)

def llm_structurize(raw_text: str, meta_hint: Dict[str,str], model: str) -> Dict:
    ctx = (raw_text or "").strip()
    if len(ctx) > 9000:
        ctx = ctx[:9000]

    user_msg = {
        "role": "user",
        "content": (
            "다음 채용 공고 원문을 구조화해줘.\n\n"
            f"[힌트] 회사명 후보: {meta_hint.get('company_name','')}\n"
            f"[힌트] 직무명 후보: {meta_hint.get('job_title','')}\n"
            "--- 원문 시작 ---\n"
            f"{ctx}\n"
            "--- 원문 끝 ---\n\n"
            "JSON으로만 답하고, 키는 반드시 아래만 포함:\n"
            "{"
            "\"company_name\": str, "
            "\"company_intro\": str, "
            "\"job_title\": str, "
            "\"responsibilities\": [str], "
            "\"qualifications\": [str], "
            "\"preferences\": [str]"
            "}\n"
            "- '우대 사항(preferences)'은 비워두지 말고, 원문에서 '우대/선호/preferred/plus/가산점' 등 표시가 있는 항목을 그대로 담아라.\n"
            "- 불릿/마커/이모지 제거, 문장 간결화, 중복 제거."
        ),
    }

    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.2,
            response_format={"type": "json_object"},
            messages=[{"role":"system","content":PROMPT_SYSTEM_STRUCT}, user_msg],
        )
        data = json.loads(resp.choices[0].message.content)
    except Exception as e:
        data = {
            "company_name": meta_hint.get("company_name",""),
            "company_intro": meta_hint.get("company_intro","원문이 정제되지 않았습니다."),
            "job_title": meta_hint.get("job_title",""),
            "responsibilities": [],
            "qualifications": [],
            "preferences": [],
            "error": str(e),
        }

    # 클린업/보정
    for k in ["responsibilities","qualifications","preferences"]:
        arr = data.get(k, [])
        if not isinstance(arr, list): arr = []
        clean_list=[]; seen=set()
        for it in arr:
            t = _clean_line(str(it))
            if t and t not in seen:
                seen.add(t); clean_list.append(t)
        data[k] = clean_list[:12]

    for k in ["company_name","company_intro","job_title"]:
        if k in data and isinstance(data[k], str):
            data[k] = re.sub(r"\s+"," ", data[k]).strip()

    # LLM이 우대사항을 놓친 경우 추가 보정
    if len(data.get("preferences", [])) < 1:
        # 자격 요건 내 '우대' 키워드 이동
        kw_pref = PREF_KW
        remain=[]; moved=[]
        for q in data.get("qualifications", []):
            if kw_pref.search(q):
                moved.append(q)
            else:
                remain.append(q)
        if moved:
            data["preferences"] = moved[:12]
            data["qualifications"] = remain[:12]
    return data

# ================== 파일 리더 (PDF/TXT/MD/DOCX) ==================
try:
    import pypdf
except Exception:
    pypdf = None

def read_pdf(data: bytes) -> str:
    if pypdf is None:
        return ""
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        return "\n\n".join([(reader.pages[i].extract_text() or "") for i in range(len(reader.pages))])
    except Exception:
        return ""

def read_docx(data: bytes) -> str:
    try:
        import docx2txt, tempfile
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=True) as tmp:
            tmp.write(data); tmp.flush()
            text = docx2txt.process(tmp.name) or ""
            return text
    except Exception:
        return ""

def read_file_text(uploaded) -> str:
    name = uploaded.name.lower()
    data = uploaded.read()
    if name.endswith((".txt",".md")):
        for enc in ("utf-8","cp949","euc-kr"):
            try: return data.decode(enc)
            except Exception: continue
        return data.decode("utf-8", errors="ignore")
    elif name.endswith(".pdf"):
        return read_pdf(data)
    elif name.endswith(".docx"):
        return read_docx(data)
    return ""

# ================== 간단 청크/임베딩 ==================
def chunk(text: str, size: int = 600, overlap: int = 120) -> List[str]:
    t = re.sub(r"\s+"," ", text).strip()
    if not t: return []
    out, start = [], 0
    while start < len(t):
        end = min(len(t), start+size)
        out.append(t[start:end])
        if end == len(t): break
        start = max(0, end-overlap)
    return out

def embed_texts(texts: List[str], model_name: str) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    resp = client.embeddings.create(model=model_name, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return vecs

# ================== 세션 상태 ==================
if "clean_struct" not in st.session_state:
    st.session_state.clean_struct = None
if "resume_raw" not in st.session_state:
    st.session_state.resume_raw = ""
if "resume_chunks" not in st.session_state:
    st.session_state.resume_chunks = []
if "resume_embeds" not in st.session_state:
    st.session_state.resume_embeds = None

# ================== 1) 채용 공고 URL → 정제 ==================
st.header("1) 채용 공고 URL → 정제")
url = st.text_input("채용 공고 상세 URL", placeholder="채용 공고 사이트의 URL을 입력하세요")
if st.button("원문 수집 → 정제", type="primary"):
    if not url.strip():
        st.warning("URL을 입력하세요.")
    else:
        with st.spinner("원문 수집 중..."):
            raw, meta, soup = fetch_all_text(url.strip())
            hint = extract_company_meta(soup)

        if not raw:
            st.error("원문을 가져오지 못했습니다. (로그인/동적 렌더링/봇 차단 가능)")
        else:
            # 1) 사이트별 정밀 크롤러 우선 시도
            site_struct = parse_portal_specific(url.strip(), soup, raw)
            ok_cnt = sum(len(site_struct.get(k, [])) for k in ["responsibilities","qualifications","preferences"])
            # 2) 결과가 충분하지 않으면 LLM 정제 보완
            if ok_cnt < 3:  # 항목이 너무 적거나 거의 못 찾았으면
                with st.spinner("LLM으로 정제 중..."):
                    clean = llm_structurize(raw, hint, CHAT_MODEL)
                # 정밀+LLM 병합(정밀이 있는 항목은 우선 유지)
                for k in ["responsibilities","qualifications","preferences"]:
                    if site_struct.get(k):
                        clean[k] = site_struct[k]
                # 메타 보강
                if site_struct and not clean.get("company_name"): clean["company_name"] = hint.get("company_name","")
                if site_struct and not clean.get("job_title"):    clean["job_title"]    = hint.get("job_title","")
            else:
                # 정밀 결과로 clean 구성
                clean = {
                    "company_name": hint.get("company_name",""),
                    "company_intro": hint.get("company_intro",""),
                    "job_title": hint.get("job_title",""),
                    "responsibilities": site_struct.get("responsibilities",[]),
                    "qualifications":   site_struct.get("qualifications",[]),
                    "preferences":      site_struct.get("preferences",[]),
                }

            st.session_state.clean_struct = clean
            st.success("정제 완료!")

# ================== 2) 회사 요약 (정제 결과) ==================
st.header("2) 회사 요약")
clean = st.session_state.clean_struct
if clean:
    st.markdown(f"**회사명:** {clean.get('company_name','-')}")
    st.markdown(f"**간단한 회사 소개(요약):** {clean.get('company_intro','-')}")
    st.markdown(f"**모집 분야(직무명):** {clean.get('job_title','-')}")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**주요 업무**")
        for b in clean.get("responsibilities", []): st.markdown(f"- {b}")
    with c2:
        st.markdown("**자격 요건**")
        for b in clean.get("qualifications", []): st.markdown(f"- {b}")
    with c3:
        st.markdown("**우대 사항**")
        prefs = clean.get("preferences", [])
        if prefs:
            for b in prefs: st.markdown(f"- {b}")
        else:
            st.caption("우대 사항이 명시되지 않았습니다.")
else:
    st.info("먼저 URL을 정제해 주세요.")

st.divider()

# ================== 3) 내 이력서/프로젝트 업로드 (DOCX/TXT/MD/PDF) ==================
st.header("3) 내 이력서/프로젝트 업로드")
uploads = st.file_uploader(
    "이력서/프로젝트 파일 업로드 (PDF/TXT/MD/DOCX, 여러 개 가능)",
    type=["pdf","txt","md","docx"], accept_multiple_files=True
)

# 내부용 기본 파라미터 (UI 비노출)
_RESUME_CHUNK = 600
_RESUME_OVLP  = 120

if st.button("이력서 인덱싱(자동)", type="secondary"):
    if not uploads:
        st.warning("파일을 업로드하세요.")
    else:
        all_text=[]
        for up in uploads:
            t = read_file_text(up)
            if t: all_text.append(t)
        resume_text = "\n\n".join(all_text)
        if not resume_text.strip():
            st.error("텍스트를 추출하지 못했습니다.")
        else:
            chunks = chunk(resume_text, size=_RESUME_CHUNK, overlap=_RESUME_OVLP)
            with st.spinner("이력서 벡터화 중..."):
                embeds = embed_texts(chunks, EMBED_MODEL)
            st.session_state.resume_raw = resume_text
            st.session_state.resume_chunks = chunks
            st.session_state.resume_embeds = embeds
            st.success(f"인덱싱 완료 (청크 {len(chunks)}개)")

# ================== (Step4) 이력서 기반 자소서 생성 ==================
st.header("4) 이력서 기반 자소서 생성")
topic = st.text_input("회사 요청 주제(선택)", placeholder="예: 성장 과정 / 직무 지원동기 / 협업 경험 / 문제해결 사례 등")

def build_cover_letter(clean_struct: Dict, resume_text: str, topic_hint: str, model: str) -> str:
    company = json.dumps(clean_struct or {}, ensure_ascii=False)
    resume_snippet = resume_text.strip()
    if len(resume_snippet) > 9000:
        resume_snippet = resume_snippet[:9000]

    system = (
        "너는 한국어 자기소개서 전문가다. 채용 공고의 회사/직무 요건과 후보자의 이력서를 참고해 "
        "회사 특화 자소서를 작성한다. 과장/허위는 금지하고, 수치/지표/기간/임팩트 중심으로 구체화한다."
    )
    if topic_hint and topic_hint.strip():
        req = f"회사 측 요청 주제는 '{topic_hint.strip()}' 이다. 이 주제를 중심으로 서술하라."
    else:
        req = "특정 주제 요청이 없으므로, 채용 공고의 요건을 중심으로 지원동기와 직무적합성을 강조하라."

    user = (
        f"[회사/직무 요약(JSON)]\n{company}\n\n"
        f"[후보자 이력서(요약 가능)]\n{resume_snippet}\n\n"
        f"[작성 지시]\n- {req}\n"
        "- 분량: 600~1000자\n"
        "- 구성: 1) 지원 동기 2) 직무 관련 핵심 역량·경험 3) 성과/지표 4) 입사 후 기여 방안 5) 마무리\n"
        "- 자연스럽고 진정성 있는 1인칭 서술. 문장과 문단 가독성을 유지.\n"
        "- 불필요한 미사여구/중복/광고 문구 삭제."
    )
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.4,
            messages=[{"role":"system","content":system},{"role":"user","content":user}]
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"(자소서 생성 실패: {e})"

if st.button("자소서 생성", type="primary"):
    if not st.session_state.clean_struct:
        st.warning("먼저 회사 URL을 정제하세요.")
    elif not st.session_state.resume_raw.strip():
        st.warning("먼저 이력서를 업로드하고 '이력서 인덱싱(자동)'을 눌러주세요.")
    else:
        with st.spinner("자소서 생성 중..."):
            cover = build_cover_letter(st.session_state.clean_struct, st.session_state.resume_raw, topic, CHAT_MODEL)
        st.subheader("자소서 (생성 결과)")
        st.write(cover)
        st.download_button(
            "자소서 TXT 다운로드",
            data=cover.encode("utf-8"),
            file_name="cover_letter.txt",
            mime="text/plain"
        )
