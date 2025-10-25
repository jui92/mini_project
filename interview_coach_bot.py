# -*- coding: utf-8 -*-
import os, re, json, urllib.parse
from typing import Optional, Tuple, Dict, List

import requests
from bs4 import BeautifulSoup
import html2text
import streamlit as st

# ============== 기본 설정 ==============
st.set_page_config(page_title="채용 공고 파서 + LLM 정제", page_icon="🧾", layout="wide")
st.title("채용 공고 파서 · URL → 원문 수집 → LLM 정제 출력")

# ============== OpenAI 준비 ==============
try:
    from openai import OpenAI
except ImportError:
    st.error("`openai` 패키지가 필요합니다. requirements.txt에 openai를 추가하세요.")
    st.stop()

API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)
if not API_KEY:
    API_KEY = st.text_input("OPENAI_API_KEY 입력", type="password")
if not API_KEY:
    st.stop()
client = OpenAI(api_key=API_KEY)

CHAT_MODEL = st.sidebar.selectbox("LLM 모델", ["gpt-4o-mini","gpt-4o"], index=0)

# ============== HTTP 유틸 ==============
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

# ============== 원문 수집 (Jina → Web → BS4) ==============
def fetch_jina_text(url: str, timeout: int = 15) -> str:
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

# ============== 메타/섹션 보조 추출(LLM 힌트용) ==============
def extract_company_meta(soup: Optional[BeautifulSoup]) -> Dict[str,str]:
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

# ============== LLM 정제 (핵심) ==============
PROMPT_SYSTEM = (
    "너는 채용 공고를 깔끔하게 구조화하는 보조원이다. "
    "입력 텍스트는 포털 광고 문구, UI잔재, 복수 직무가 섞여 있을 수 있다. "
    "한국어로 간결하고 중복없이 정제하라."
)

def llm_structurize(raw_text: str, meta_hint: Dict[str,str], model: str) -> Dict:
    # 컨텍스트 과다 방지
    ctx = raw_text.strip()
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
            "}"
        ),
    }

    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[{"role":"system","content":PROMPT_SYSTEM}, user_msg],
        )
        data = json.loads(resp.choices[0].message.content)
        # 방어적 후처리
        for k in ["responsibilities","qualifications","preferences"]:
            if not isinstance(data.get(k, []), list):
                data[k] = []
            clean = []
            seen = set()
            for it in data[k]:
                t = re.sub(r"\s+"," ", str(it)).strip(" -•·").strip()
                if t and t not in seen:
                    seen.add(t); clean.append(t)
            data[k] = clean[:12]
        for k in ["company_name","company_intro","job_title"]:
            if k in data and isinstance(data[k], str):
                data[k] = re.sub(r"\s+"," ", data[k]).strip()
        return data
    except Exception as e:
        return {
            "company_name": meta_hint.get("company_name",""),
            "company_intro": meta_hint.get("company_intro","원문이 정제되지 않았습니다."),
            "job_title": meta_hint.get("job_title",""),
            "responsibilities": [],
            "qualifications": [],
            "preferences": [],
            "error": str(e),
        }

# ============== UI ==============
st.header("1) 채용 공고 URL")
url = st.text_input("채용 공고 상세 URL", placeholder="예: https://www.wanted.co.kr/wd/123456")
run = st.button("원문 수집 → LLM 정제", type="primary")

if run:
    if not url.strip():
        st.warning("URL을 입력하세요.")
        st.stop()

    with st.spinner("원문 수집 중..."):
        raw, meta, soup = fetch_all_text(url)
        hint = extract_company_meta(soup)

    with st.expander("디버그: 수집 메타/힌트", expanded=False):
        st.json({"fetch_meta": meta, "meta_hint": hint})

    if not raw:
        st.error("원문을 가져오지 못했습니다. (로그인/동적 렌더링/봇 차단 가능)")
        st.stop()

    with st.spinner("LLM으로 정제 중..."):
        clean = llm_structurize(raw, hint, CHAT_MODEL)

    st.header("2) 정제된 회사 요약 / 채용 요건")
    st.markdown("### 회사명")
    st.write(clean.get("company_name") or "N/A")

    st.markdown("### 간단한 회사 소개(요약)")
    st.write(clean.get("company_intro") or "N/A")

    st.markdown("### 모집 분야(직무명)")
    st.write(clean.get("job_title") or "N/A")

    st.markdown("### 주요 업무")
    resp = clean.get("responsibilities", [])
    if resp:
        for b in resp: st.markdown(f"- {b}")
    else:
        st.write("—")

    st.markdown("### 자격 요건")
    qual = clean.get("qualifications", [])
    if qual:
        for b in qual: st.markdown(f"- {b}")
    else:
        st.write("—")

    st.markdown("### 우대 사항")
    pref = clean.get("preferences", [])
    if pref:
        for b in pref: st.markdown(f"- {b}")
    else:
        st.write("—")

    st.divider()
    st.subheader("다운로드")
    st.download_button("원문 전체 다운로드", data=raw.encode("utf-8"),
                       file_name="job_posting_raw.txt", mime="text/plain")
    st.download_button("정제 결과(JSON) 다운로드",
                       data=json.dumps(clean, ensure_ascii=False, indent=2).encode("utf-8"),
                       file_name="job_posting_clean.json", mime="application/json")

st.caption("팁) ‘상세 더보기’가 필요한 페이지는 Jina 프록시를 우선 사용하여 최대한 텍스트를 확보합니다.")
