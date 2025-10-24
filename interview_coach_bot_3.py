# -*- coding: utf-8 -*-
# 파일명 예: interview_job_fulltext_app.py

import os, re, io, json, html, textwrap, urllib.parse
from typing import Tuple, Dict, List, Optional

import streamlit as st
import requests
from bs4 import BeautifulSoup

# 선택 의존성 (없어도 동작)
try:
    from langchain_community.document_loaders import WebBaseLoader
    LC_OK = True
except Exception:
    LC_OK = False

# =========================
# Page / Secrets
# =========================
st.set_page_config(page_title="회사 요약 · 채용 요건 원문 수집/요약", page_icon="🧲", layout="wide")

def _get(key: str) -> Optional[str]:
    v = os.getenv(key)
    if v: return v
    try:
        return st.secrets.get(key, None)
    except Exception:
        return None

OPENAI_API_KEY = _get("OPENAI_API_KEY")
NAVER_ID        = _get("NAVER_CLIENT_ID")
NAVER_SECRET    = _get("NAVER_CLIENT_SECRET")

# =========================
# Utils
# =========================
def _clean(s: str) -> str:
    if not s: return ""
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _abs(url: str) -> str:
    u = url.strip()
    if not u.startswith("http"): u = "https://" + u
    return u

# =========================
# Portal-specific collectors
# =========================
def wanted_full_text(url: str) -> Tuple[str, Dict]:
    """원티드 상세(/wd/<id>)의 펼쳐진 본문을 JSON API로 수집."""
    m = re.search(r"/wd/(\d+)", url)
    if not m:
        return "", {"wanted": "no_id"}

    jid = m.group(1)
    endpoints = [
        f"https://www.wanted.co.kr/api/v4/jobs/{jid}?locale=ko-KR",
        f"https://www.wanted.co.kr/api/v2/jobs/{jid}?locale=ko-KR",
        f"https://www.wanted.co.kr/api/v4/jobs/{jid}",
        f"https://www.wanted.co.kr/api/v2/jobs/{jid}",
    ]
    hdr = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": url,
        "X-Wanted-Language": "ko-KR",
        "Accept-Language": "ko-KR,ko;q=0.9",
    }
    for ep in endpoints:
        try:
            r = requests.get(ep, headers=hdr, timeout=10)
            if r.status_code != 200 or "application/json" not in r.headers.get("content-type",""):
                continue
            data = r.json()
            texts = []
            keys = [
                "detail","description","intro","qualification","prefer","requirements",
                "mainTasks","responsibility","responsibilities","summary","content",
                "job_detail","jobDescription"
            ]
            def walk(obj):
                if isinstance(obj, dict):
                    for k,v in obj.items():
                        if isinstance(v,(dict,list)):
                            walk(v)
                        else:
                            if isinstance(k,str) and isinstance(v,str):
                                ks = k.lower()
                                if any(sub in ks for sub in keys):
                                    s = _clean(v)
                                    if len(s)>3: texts.append(s)
                elif isinstance(obj, list):
                    for it in obj: walk(it)
            walk(data)

            if not texts:
                blob = json.dumps(data, ensure_ascii=False)
                cand = re.findall(r'["\'](?:detail|description|qualification|prefer|requirements)["\']\s*:\s*"(.*?)"', blob, flags=re.S)
                texts += [_clean(x) for x in cand]

            if texts:
                return "\n\n".join(dict.fromkeys(texts)), {"source":"wanted+json","url_final":url}
        except Exception:
            continue
    return "", {"wanted":"fail"}

def saramin_full_text(url: str) -> Tuple[str, Dict]:
    """사람인 상세(SSR + 일부 접힘). 대표 컨테이너 모아 원문 조립."""
    if "saramin.co.kr" not in url:
        return "", {"saramin":"skip"}
    try:
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
        if r.status_code != 200: return "", {"saramin":"http_"+str(r.status_code)}
        soup = BeautifulSoup(r.text, "html.parser")
        sections = []
        for sel in ["#job_summary",".user_content",".wrap_jview",".cont",".content"]:
            for c in soup.select(sel):
                txt=_clean(c.get_text(" "))
                if len(txt)>50: sections.append(txt)
        if not sections:
            for h in soup.select("h2,h3,h4"):
                title=_clean(h.get_text(" "))
                if not title: continue
                buf=[title]
                sib=h.find_next_sibling()
                stop={"h2","h3","h4"}
                while sib and getattr(sib,"name",None) not in stop:
                    if getattr(sib,"name","") in {"p","ul","ol","li","div","section"}:
                        s=_clean(sib.get_text(" "))
                        if len(s)>2: buf.append(s)
                    sib=getattr(sib,"next_sibling",None)
                if len(" ".join(buf))>50: sections.append("\n".join(buf))
        if sections:
            return "\n\n".join(dict.fromkeys(sections)), {"source":"saramin+raw","url_final":url}
    except Exception:
        pass
    return "", {"saramin":"fail"}

def jobkorea_full_text(url: str) -> Tuple[str, Dict]:
    """잡코리아 상세(SSR). 대표 컨테이너 수집."""
    if "jobkorea.co.kr" not in url:
        return "", {"jobkorea":"skip"}
    try:
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
        if r.status_code != 200: return "", {"jobkorea":"http_"+str(r.status_code)}
        soup = BeautifulSoup(r.text, "html.parser")
        sections = []
        for sel in [".recruit-info",".detail",".tbCompanyInfo",".readSum",".section",".devView"]:
            for c in soup.select(sel):
                txt=_clean(c.get_text(" "))
                if len(txt)>50: sections.append(txt)
        if sections:
            return "\n\n".join(dict.fromkeys(sections)), {"source":"jobkorea+raw","url_final":url}
    except Exception:
        pass
    return "", {"jobkorea":"fail"}

# =========================
# Generic loaders (Jina → WebBase → BS4)
# =========================
def get_full_page_text(url: str) -> Tuple[str, Dict]:
    u = _abs(url)
    meta = {"url_final": u}

    # 1) 포털 전용
    if "wanted.co.kr/wd/" in u:
        t, _ = wanted_full_text(u)
        if t:
            meta.update({"source":"wanted+raw","lens":{"jina":0,"webbase":len(t),"bs4":len(t)}})
            return t, meta
    if "saramin.co.kr" in u:
        t, _ = saramin_full_text(u)
        if t:
            meta.update({"source":"saramin+raw","lens":{"jina":0,"webbase":len(t),"bs4":len(t)}})
            return t, meta
    if "jobkorea.co.kr" in u:
        t, _ = jobkorea_full_text(u)
        if t:
            meta.update({"source":"jobkorea+raw","lens":{"jina":0,"webbase":len(t),"bs4":len(t)}})
            return t, meta

    # 2) Jina 프리렌더
    try:
        ep = "https://r.jina.ai/http://" + u.replace("https://","").replace("http://","")
        r = requests.get(ep, headers={"User-Agent":"Mozilla/5.0"}, timeout=12)
        if r.status_code == 200 and len(r.text.strip())>200:
            t = _clean(r.text)
            meta.update({"source":"jina","lens":{"jina":len(t)}})
            return t, meta
    except Exception:
        pass

    # 3) WebBaseLoader
    if LC_OK:
        try:
            docs = WebBaseLoader(u).load()
            txt = "\n\n".join(d.page_content for d in docs)
            if len(txt.strip())>50:
                meta.update({"source":"webbase","lens":{"webbase":len(txt)}})
                return _clean(txt), meta
        except Exception:
            pass

    # 4) BS4
    try:
        r = requests.get(u, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
        if r.status_code == 200 and "text/html" in r.headers.get("content-type",""):
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script","style","noscript"]): tag.extract()
            txt = _clean(soup.get_text(" "))
            if len(txt)>50:
                meta.update({"source":"bs4","lens":{"bs4":len(txt)}})
                return txt, meta
    except Exception:
        pass

    meta.update({"source":"none","lens":{"jina":0,"webbase":0,"bs4":0}})
    return "", meta

# =========================
# Search (Naver → DuckDuckGo)
# =========================
JOB_SITES = ["wanted.co.kr","saramin.co.kr","jobkorea.co.kr","rocketpunch.com","linkedin.com","indeed.com"]

def naver_search_web(query: str, display: int = 5) -> List[str]:
    if not (NAVER_ID and NAVER_SECRET): return []
    url = "https://openapi.naver.com/v1/search/webkr.json"
    headers = {"X-Naver-Client-Id": NAVER_ID, "X-Naver-Client-Secret": NAVER_SECRET}
    params = {"query": query, "display": display, "sort": "date"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=6)
        if r.status_code!=200: return []
        js = r.json()
        out=[]
        for it in js.get("items",[]):
            link = it.get("link")
            if link and link not in out: out.append(link)
        return out
    except Exception:
        return []

def duckduck_search(query: str, display: int = 10) -> List[str]:
    url = f"https://duckduckgo.com/html/?q={urllib.parse.quote(query)}"
    try:
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=8)
        if r.status_code!=200: return []
        soup = BeautifulSoup(r.text, "html.parser")
        out=[]
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/l/?kh=-1&uddg="):
                href = urllib.parse.unquote(href.split("/l/?kh=-1&uddg=")[-1])
            dom = urllib.parse.urlparse(href).netloc.lower()
            if any(s in dom for s in JOB_SITES):
                out.append(href)
            if len(out)>=display: break
        return out
    except Exception:
        return []

def discover_job_url(company: str, role: str, limit: int = 6) -> List[str]:
    q1 = f"{company} {role} 채용"
    site_part = " OR ".join([f"site:{s}" for s in JOB_SITES])
    q2 = f"{company} {role} ({site_part})"
    urls=[]
    if NAVER_ID and NAVER_SECRET:
        urls += naver_search_web(q1, display=6)
        urls += naver_search_web(q2, display=6)
    if not urls:
        urls += duckduck_search(q2, display=10)
    seen=set(); out=[]
    for u in urls:
        try:
            d = urllib.parse.urlparse(u).netloc.lower()
            if any(s in d for s in JOB_SITES):
                if u not in seen:
                    seen.add(u); out.append(u)
        except Exception: pass
        if len(out)>=limit: break
    return out

# =========================
# OpenAI summarizer (optional)
# =========================
OPENAI_READY = False
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY, timeout=40.0)
        OPENAI_READY = True
    except Exception:
        OPENAI_READY = False

def llm_summarize_sections(raw_text: str, company: str) -> Dict[str, List[str] | str]:
    """원문에서 회사소개/주요업무/자격요건/우대사항을 요약으로 생성."""
    if not (OPENAI_READY and raw_text.strip()):
        return {"intro":"", "resp":[], "qual":[], "pref":[]}
    sys = ("너는 채용공고를 읽고 섹션별 핵심을 한국어로 요약하는 도우미다. "
           "가능하면 공고의 문구를 그대로 따르지 말고 간결히 재서술하되, 의미는 유지한다.")
    user = f"""
[회사명] {company}

[채용공고 원문]
{raw_text[:12000]}

[요청]
1) 간단한 회사 소개: 2~3문장
2) 주요 업무: 불릿 5~8개
3) 자격 요건: 불릿 5~8개
4) 우대 사항: 불릿 5~8개
JSON으로만 답하라. 키는 intro(resp/qual/pref)이며 resp/qual/pref는 리스트.
"""
    try:
        r = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[{"role":"system","content":sys},{"role":"user","content":user}]
        )
        content = r.choices[0].message.content.strip()
        js = json.loads(content)
        intro = _clean(js.get("intro",""))
        resp  = [_clean(x) for x in js.get("resp",[]) if _clean(x)]
        qual  = [_clean(x) for x in js.get("qual",[]) if _clean(x)]
        pref  = [_clean(x) for x in js.get("pref",[]) if _clean(x)]
        return {"intro":intro,"resp":resp[:12],"qual":qual[:12],"pref":pref[:12]}
    except Exception:
        return {"intro":"","resp":[],"qual":[],"pref":[]}

# =========================
# UI
# =========================
st.title("🧲 회사 요약 · 채용 요건 (원문 수집 + 요약)")

with st.sidebar:
    st.header("입력")
    company = st.text_input("회사명", placeholder="예: 마이리얼트립 / 화해 / 카카오뱅크")
    role    = st.text_input("지원 직무명", placeholder="예: Data Analyst / ML Engineer")
    job_url = st.text_input("채용 공고 URL(선택)", placeholder="상세 URL을 모르면 비워두세요")
    st.caption("URL이 없으면 검색→후보 중 첫 번째를 시도합니다.")
    btn_go  = st.button("회사/직무 정보 불러오기", type="primary", use_container_width=True)

# 세션 상태
if "raw_job_text" not in st.session_state:
    st.session_state.raw_job_text = ""
if "job_url_final" not in st.session_state:
    st.session_state.job_url_final = ""
if "meta_collect" not in st.session_state:
    st.session_state.meta_collect = {}

# 실행
if btn_go:
    urls = [job_url] if job_url.strip() else discover_job_url(company, role, limit=6)
    chosen = None
    for u in urls:
        if not u: continue
        chosen = u; break
    if not chosen:
        st.warning("공고 URL을 찾지 못했습니다. URL을 직접 입력해 주세요.")
    else:
        with st.spinner("원문 수집 중…(포털 전용 → Jina → WebBase → BS4)"):
            txt, meta = get_full_page_text(chosen)
            st.session_state.raw_job_text  = txt
            st.session_state.job_url_final = meta.get("url_final") or chosen
            st.session_state.meta_collect  = meta
        if not st.session_state.raw_job_text:
            st.warning("원문 텍스트를 가져오지 못했습니다. 로그인/봇차단/동적 렌더링 가능성이 있습니다.")
        else:
            st.success("원문 수집 완료!")

# 표시: 원문 전체
if st.session_state.raw_job_text:
    st.info("아래는 채용 상세 페이지에서 추출한 **원문 전체 텍스트**입니다. (접힘 포함, 가능한 한 모두)")
else:
    st.warning("원문이 아직 없습니다. 좌측에서 회사/직무를 입력하고 불러오기를 실행하세요.")

c1, c2 = st.columns(2)
with c1:
    st.subheader("회사 요약 (원문 전체)")
    st.text_area("회사 요약 원문", value=st.session_state.raw_job_text, height=420)
    st.download_button("회사 요약 원문 다운로드",
                       data=st.session_state.raw_job_text.encode("utf-8-sig"),
                       file_name="company_fulltext.txt", use_container_width=True)
with c2:
    st.subheader("채용 요건 (원문 전체)")
    st.text_area("채용 요건 원문", value=st.session_state.raw_job_text, height=420)
    st.download_button("채용 요건 원문 다운로드",
                       data=st.session_state.raw_job_text.encode("utf-8-sig"),
                       file_name="job_requirements_fulltext.txt", use_container_width=True)

# 표시: 요약 섹션
st.divider()
st.subheader("요약 섹션 (회사 소개 / 주요업무 / 자격요건 / 우대사항)")

if st.session_state.raw_job_text and OPENAI_READY:
    with st.spinner("요약 생성 중…"):
        summ = llm_summarize_sections(st.session_state.raw_job_text, company or "")
    intro = summ.get("intro","")
    resp  = summ.get("resp",[])
    qual  = summ.get("qual",[])
    pref  = summ.get("pref",[])
    st.markdown(f"**회사명:** {company or '-'}")
    if intro: st.markdown(f"**간단한 회사 소개(요약)**\n\n{intro}")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.markdown("**주요업무(요약)**")
        if resp: st.markdown("\n".join([f"- {x}" for x in resp]))
        else: st.caption("요약 가능한 주요업무가 없습니다.")
    with cc2:
        st.markdown("**자격요건(요약)**")
        if qual: st.markdown("\n".join([f"- {x}" for x in qual]))
        else: st.caption("요약 가능한 자격요건이 없습니다.")
    with cc3:
        st.markdown("**우대사항(요약)**")
        if pref: st.markdown("\n".join([f"- {x}" for x in pref]))
        else: st.caption("요약 가능한 우대사항이 없습니다.")
else:
    if not OPENAI_READY:
        st.info("요약 섹션을 사용하려면 OPENAI_API_KEY가 필요합니다.")
    else:
        st.info("먼저 원문을 수집하세요.")

# 디버그: 경로/상태
st.divider()
with st.expander("디버그: 원문 수집 경로/상태"):
    st.write({
        "url_final": st.session_state.get("job_url_final",""),
        "source":    st.session_state.get("meta_collect",{}).get("source",""),
        "lens":      st.session_state.get("meta_collect",{}).get("lens",{}),
    })
    st.caption("source가 wanted+raw/saramin+raw/jobkorea+raw면 포털 전용 수집기가 동작하여 접힌 본문까지 포함합니다.")

# 자가진단: 직접 URL 테스트
st.divider()
with st.expander("🧪 원문 테스트(직접 URL)"):
    test_url = st.text_input("테스트할 채용 상세 URL을 입력하세요", key="test_url")
    if st.button("테스트 실행"):
        if not test_url.strip():
            st.warning("URL을 입력하세요.")
        else:
            txt, meta = get_full_page_text(test_url.strip())
            st.write(meta)
            st.write(f"텍스트 길이: {len(txt)}")
            st.text_area("미리보기(앞 3000자)", value=txt[:3000], height=300)

st.caption("Tip) URL이 없으면 검색 후보가 부정확할 수 있습니다. 가능하면 상세 공고 URL을 직접 넣어주세요.")
