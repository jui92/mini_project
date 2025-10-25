# -*- coding: utf-8 -*-
import os, re, json, urllib.parse
from typing import Optional, Tuple, Dict, List

import requests
from bs4 import BeautifulSoup
import html2text
import streamlit as st

# -----------------------------
# Streamlit page
# -----------------------------
st.set_page_config(page_title="채용 공고 파서 (직접 URL → 구조화 요약)", page_icon="🧾", layout="wide")
st.title("채용 공고 파서 · 직접 URL → 회사 요약/채용 요건")

# -----------------------------
# HTTP helpers
# -----------------------------
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
        if r.status_code == 200 and "text/html" in r.headers.get("content-type", ""):
            return r
    except Exception:
        pass
    return None

# -----------------------------
# Text extraction (Jina → WebBase → BS4)
# -----------------------------
def fetch_jina_text(url: str, timeout: int = 15) -> str:
    """프리렌더 텍스트(더보기/동적 포함 가능)."""
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

    # 긴 블록 위주로
    blocks = []
    for sel in ["article", "section", "main", "div", "ul", "ol"]:
        for el in soup.select(sel):
            txt = el.get_text(" ", strip=True)
            if txt and len(txt) > 300:
                txt = re.sub(r"\s+", " ", txt)
                blocks.append(txt)
    if not blocks:
        all_txt = soup.get_text(" ", strip=True)
        return all_txt[:120000], soup

    seen, out = set(), []
    for b in blocks:
        if b not in seen:
            seen.add(b); out.append(b)
    return ("\n\n".join(out)[:120000], soup)

def fetch_all_text(url: str) -> Tuple[str, Dict, Optional[BeautifulSoup]]:
    """최대치 텍스트와 디버그 메타, soup 반환"""
    url = normalize_url(url)
    if not url:
        return "", {"error":"invalid_url"}, None

    jina = fetch_jina_text(url)
    if jina:
        # soup는 별도로
        _, soup = fetch_bs4_text(url)
        return jina, {"source":"jina","len":len(jina),"url_final":url}, soup

    webbase = fetch_webbase_text(url)
    if webbase:
        _, soup = fetch_bs4_text(url)
        return webbase, {"source":"webbase","len":len(webbase),"url_final":url}, soup

    bs, soup = fetch_bs4_text(url)
    return bs, {"source":"bs4","len":len(bs),"url_final":url}, soup

# -----------------------------
# Section parsing
# -----------------------------
H_ROLE = [r"모집\s*분야", r"채용\s*분야", r"Position", r"Role", r"직무\s*명", r"Job\s*Title"]
H_RESP = [r"주요\s*업무", r"담당\s*업무", r"업무(?!\S)", r"Responsibilities?", r"What you will do"]
H_QUAL = [r"자격\s*요건", r"지원\s*자격", r"필수\s*요건", r"Requirements?", r"Qualifications?"]
H_PREF = [r"우대\s*사항", r"우대\s*조건", r"Preferred", r"Nice to have", r"Plus"]

HEADER_PATTERNS = [
    ("role", H_ROLE),
    ("resp", H_RESP),
    ("qual", H_QUAL),
    ("pref", H_PREF),
]

BULLET_RX = re.compile(r"^\s*(?:[-*•·▪▶]|[0-9]+\.)\s+")

def split_lines(text: str) -> List[str]:
    lines = [re.sub(r"\s+", " ", l).strip() for l in text.splitlines()]
    return [l for l in lines if l]

def pick_first(lines: List[str], patterns: List[str]) -> Optional[int]:
    for i, ln in enumerate(lines):
        s = ln.lower()
        for pat in patterns:
            if re.search(pat, s, re.I):
                return i
    return None

def extract_sections_from_text(text: str) -> Dict[str, List[str]]:
    """
    텍스트에서 [role, resp, qual, pref] 구간을 찾아 불릿 리스트로 정리
    """
    lines = split_lines(text)
    idx = {}
    for key, pats in HEADER_PATTERNS:
        pos = pick_first(lines, pats)
        if pos is not None: idx[key] = pos

    if not idx:
        return {"role":[], "resp":[], "qual":[], "pref":[]}

    # 섹션 경계 계산
    order = sorted(idx.items(), key=lambda x: x[1])
    bounds = []
    for i, (k, start) in enumerate(order):
        end = order[i+1][1] if i+1 < len(order) else len(lines)
        bounds.append((k, start, end))

    out = {"role":[], "resp":[], "qual":[], "pref":[]}
    for k, s, e in bounds:
        chunk = lines[s+1:e]   # 헤더 다음부터
        bullets = []
        cur = ""
        for ln in chunk:
            if BULLET_RX.match(ln):
                if cur: bullets.append(cur.strip()); cur = ""
                bullets.append(BULLET_RX.sub("", ln).strip())
            else:
                # 문장이 길면 이어붙이기
                if cur:
                    cur += " " + ln
                else:
                    cur = ln
        if cur: bullets.append(cur.strip())

        # 너무 짧은 라인은 제거
        bullets = [b for b in bullets if len(b) > 3]
        out[k] = bullets[:20]
    return out

def extract_company_meta(soup: Optional[BeautifulSoup]) -> Dict[str, str]:
    meta = {"company_name":"", "company_intro": "", "job_title":""}
    if not soup: return meta

    # 회사명 후보
    # og:site_name, application-name, title 분리
    cand = []
    og = soup.find("meta", {"property":"og:site_name"})
    if og and og.get("content"): cand.append(og["content"])
    app = soup.find("meta", {"name":"application-name"})
    if app and app.get("content"): cand.append(app["content"])
    if soup.title and soup.title.string: cand.append(soup.title.string)

    # 간단 정제
    cand = [re.split(r"[\-\|\·\—]", c)[0].strip() for c in cand if c]
    cand = [c for c in cand if 2 <= len(c) <= 40]
    meta["company_name"] = cand[0] if cand else ""

    # 소개 = meta description 우선
    md = soup.find("meta", {"name":"description"}) or soup.find("meta", {"property":"og:description"})
    if md and md.get("content"):
        intro = md["content"].strip()
        meta["company_intro"] = re.sub(r"\s+", " ", intro)[:500]

    # 직무명 후보: h1/h2/og:title
    jt = ""
    ogt = soup.find("meta", {"property":"og:title"})
    if ogt and ogt.get("content"): jt = ogt["content"]
    if not jt:
        h1 = soup.find("h1")
        if h1 and h1.get_text(): jt = h1.get_text(strip=True)
    if not jt:
        h2 = soup.find("h2")
        if h2 and h2.get_text(): jt = h2.get_text(strip=True)

    jt = re.sub(r"\s+", " ", jt).strip()
    meta["job_title"] = jt[:120]
    return meta

# -----------------------------
# UI — Direct URL mode only
# -----------------------------
st.header("1) 채용 공고 URL 입력")
url = st.text_input("채용 공고 상세 URL", placeholder="예: https://www.wanted.co.kr/wd/123456")

col_btn = st.columns(2)
with col_btn[0]:
    run = st.button("원문 가져오기", type="primary")
with col_btn[1]:
    show_raw = st.checkbox("원문 미리보기 표시", value=True)

if run:
    if not url.strip():
        st.warning("URL을 입력하세요.")
    else:
        with st.spinner("원문 수집 및 구조화 중..."):
            text_all, meta, soup = fetch_all_text(url.strip())
            company_meta = extract_company_meta(soup)
            sections = extract_sections_from_text(text_all)

        # 디버그
        with st.expander("디버그: 원문 수집 상태/메타"):
            st.json({"fetch_meta": meta, "company_meta": company_meta})
            st.write(f"원문 길이: {len(text_all)}")

        # 레이아웃
        st.header("2) 회사 요약 / 채용 요건 (구조화 출력)")

        st.markdown("### 회사명")
        st.write(company_meta.get("company_name") or "N/A")

        st.markdown("### 간단한 회사 소개(요약)")
        st.write(company_meta.get("company_intro") or "메타 설명을 찾지 못했습니다.")

        st.markdown("### 모집 분야(직무명)")
        job_title = company_meta.get("job_title") or (sections["role"][0] if sections["role"] else "")
        st.write(job_title if job_title else "본문에서 직무명을 확정하지 못했습니다.")

        st.markdown("### 주요 업무")
        resp = sections.get("resp") or []
        if resp:
            for b in resp: st.markdown(f"- {b}")
        else:
            st.write("본문에서 '주요 업무' 섹션을 찾지 못했습니다.")

        st.markdown("### 자격 요건")
        qual = sections.get("qual") or []
        if qual:
            for b in qual: st.markdown(f"- {b}")
        else:
            st.write("본문에서 '자격 요건' 섹션을 찾지 못했습니다.")

        st.markdown("### 우대 사항")
        pref = sections.get("pref") or []
        if pref:
            for b in pref: st.markdown(f"- {b}")
        else:
            st.write("본문에서 '우대 사항' 섹션을 찾지 못했습니다.")

        # 원문 보기/다운로드
        if show_raw:
            st.divider()
            st.subheader("원문 텍스트(전체)")
            st.text_area("원문", value=text_all[:30000], height=300)
        st.download_button(
            "원문 텍스트 다운로드",
            data=text_all.encode("utf-8"),
            file_name="job_posting_raw.txt",
            mime="text/plain",
        )

        # JSON 결과 다운로드
        result = {
            "source_url": meta.get("url_final"),
            "fetch_source": meta.get("source"),
            "company_name": company_meta.get("company_name"),
            "company_intro": company_meta.get("company_intro"),
            "job_title": job_title,
            "responsibilities": resp,
            "qualifications": qual,
            "preferences": pref,
        }
        st.download_button(
            "구조화 결과(JSON) 다운로드",
            data=json.dumps(result, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="job_posting_structured.json",
            mime="application/json",
        )

# 안내
st.info(
    "⚠️ 동적 렌더링/로그인/봇차단 페이지는 일부 누락될 수 있습니다.\n"
    "- 우선순위: **Jina Reader → 정적 HTML → BS4**\n"
    "- 섹션 헤더 키워드: 주요업무/담당업무, 자격요건/지원자격, 우대사항/Preferred 등을 기준으로 자동 분리합니다."
)
