# -*- coding: utf-8 -*-
import os, re, json, urllib.parse, random, time
from typing import Optional, Tuple, Dict, List

import requests
from bs4 import BeautifulSoup
import html2text
import streamlit as st
import pandas as pd
import numpy as np

# ============== 기본 설정 ==============
st.set_page_config(page_title="회사 맞춤 면접 코치 (URL→정제→질문→채점→레이더)", page_icon="🎯", layout="wide")
st.title("회사 맞춤 면접 코치 · 채용 URL → 정제 → 질문 생성 → 채점/코칭 → 레이더")

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

with st.sidebar:
    st.subheader("모델 설정")
    CHAT_MODEL = st.selectbox("LLM 모델", ["gpt-4o-mini","gpt-4o"], index=0)

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

# ============== LLM 정제 (채용 공고 → 구조 JSON) ==============
PROMPT_SYSTEM_STRUCT = (
    "너는 채용 공고를 깔끔하게 구조화하는 보조원이다. "
    "입력 텍스트는 포털 광고 문구, UI잔재, 복수 직무가 섞여 있을 수 있다. "
    "한국어로 간결하고 중복없이 정제하라."
)

def llm_structurize(raw_text: str, meta_hint: Dict[str,str], model: str) -> Dict:
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
            model=model, temperature=0.2,
            response_format={"type": "json_object"},
            messages=[{"role":"system","content":PROMPT_SYSTEM_STRUCT}, user_msg],
        )
        data = json.loads(resp.choices[0].message.content)
        # 후처리
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

# ============== 질문 생성 ==============
PROMPT_SYSTEM_Q = (
    "너는 채용담당자다. 회사/직무 맥락과 채용요건을 반영해 면접 질문을 한국어로 생성한다. "
    "질문은 서로 형태·관점·키워드가 겹치지 않게 다양화하고, 수치/지표/기간/규모/리스크 등도 섞어라."
)

def llm_generate_questions(clean: Dict, q_type: str, level: str, model: str, num: int = 8, seed: int = 0) -> List[str]:
    # seed를 컨텍스트에 반영(샘플 변동)
    ctx = json.dumps(clean, ensure_ascii=False)
    user_msg = {
        "role": "user",
        "content": (
            f"[회사/직무/요건]\n{ctx}\n\n"
            f"[요청]\n- 질문 유형: {q_type}\n- 난이도/연차: {level}\n"
            f"- 총 {num}개, 한 줄씩\n- 중복/유사도 최소화\n- 랜덤시드: {seed}"
        ),
    }
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.9,
            messages=[{"role":"system","content":PROMPT_SYSTEM_Q}, user_msg],
        )
        txt = resp.choices[0].message.content.strip()
        lines = [re.sub(r'^\s*\d+[\).\s-]*','', l).strip() for l in txt.splitlines() if l.strip()]
        # 한 줄 질문만 남기기
        lines = [l for l in lines if len(l.split()) > 2][:num]
        # 다양도 향상: 최근 5개와 유사한 문장 제거(간단 기준)
        if "q_hist" in st.session_state:
            hist = st.session_state.q_hist[-10:]
            def sim(a,b):
                a_set=set(a.lower().split()); b_set=set(b.lower().split())
                inter=len(a_set&b_set); denom=max(1,len(a_set|b_set))
                return inter/denom
            uniq=[]
            for q in lines:
                if all(sim(q,h)<0.4 for h in hist):
                    uniq.append(q)
            if uniq: lines = uniq
        return lines[:num]
    except Exception:
        return []

# ============== 채점/코칭(정합성 보장) ==============
PROMPT_SYSTEM_SCORE = (
    "너는 톱티어 면접 코치다. 아래 형식의 JSON만 출력하라. "
    "각 기준은 0~20 정수, 총점은 기준 합계(최대 100)와 반드시 일치해야 한다. "
    "각 기준에 대해 짧은 코멘트(강점/감점요인/개선포인트 포함)를 제공하라."
)

CRITERIA = [
    "문제정의", "데이터/지표", "실행력/주도성", "협업/커뮤니케이션", "고객가치"
]

def llm_score_and_coach(clean: Dict, question: str, answer: str, model: str) -> Dict:
    ctx = json.dumps(clean, ensure_ascii=False)
    # 출력 JSON 스키마를 명시
    schema = {
        "overall_score": 0,
        "criteria": [{"name": "", "score": 0, "comment": ""} for _ in range(5)],
        "strengths": [],           # 2~3개
        "risks": [],               # 2~3개
        "improvements": [],        # 3개
        "revised_answer": ""       # STAR 기반
    }
    user_msg = {
        "role":"user",
        "content": (
            f"[회사/직무/채용요건]\n{ctx}\n\n"
            f"[면접 질문]\n{question}\n\n"
            f"[지원자 답변]\n{answer}\n\n"
            "다음 JSON 스키마로만 한국어 응답:\n"
            "{"
            "\"overall_score\": 0~100 정수,"
            "\"criteria\": [{\"name\":\"문제정의\",\"score\":0~20,\"comment\":\"...\"},"
            "{\"name\":\"데이터/지표\",\"score\":0~20,\"comment\":\"...\"},"
            "{\"name\":\"실행력/주도성\",\"score\":0~20,\"comment\":\"...\"},"
            "{\"name\":\"협업/커뮤니케이션\",\"score\":0~20,\"comment\":\"...\"},"
            "{\"name\":\"고객가치\",\"score\":0~20,\"comment\":\"...\"}],"
            "\"strengths\": [\"...\", \"...\"],"
            "\"risks\": [\"...\", \"...\"],"
            "\"improvements\": [\"...\", \"...\", \"...\"],"
            "\"revised_answer\": \"STAR 구조로 간결히\""
            "}"
        )
    }
    try:
        resp = client.chat.completions.create(
            model=model, temperature=0.2,
            response_format={"type":"json_object"},
            messages=[{"role":"system","content":PROMPT_SYSTEM_SCORE}, user_msg]
        )
        data = json.loads(resp.choices[0].message.content)

        # 방어적 정합화: 기준 5개 강제/합계=총점
        crit = data.get("criteria", [])
        # 이름 보정/누락 채우기
        fixed=[]
        # 이름 맵
        keymap = {c: c for c in CRITERIA}
        for name in CRITERIA:
            found = None
            for it in crit:
                n = str(it.get("name","")).strip()
                if n in keymap and keymap[n]==name:
                    found = it; break
            if not found:
                found = {"name": name, "score": 0, "comment": ""}
            # 범위보정
            sc = int(found.get("score",0))
            sc = max(0, min(20, sc))
            found["score"] = sc
            found["comment"] = str(found.get("comment","")).strip()
            fixed.append(found)
        total = sum(x["score"] for x in fixed)
        data["criteria"] = fixed
        data["overall_score"] = total  # 총점=합계로 강제
        # 리스트 클린
        for k in ["strengths","risks","improvements"]:
            arr = data.get(k, [])
            if not isinstance(arr, list): arr=[]
            data[k] = [str(x).strip() for x in arr if str(x).strip()][:5]
        data["revised_answer"] = str(data.get("revised_answer","")).strip()
        return data
    except Exception as e:
        return {
            "overall_score": 0,
            "criteria": [{"name": n, "score": 0, "comment": ""} for n in CRITERIA],
            "strengths": [],
            "risks": [],
            "improvements": [],
            "revised_answer": "",
            "error": str(e),
        }

# ============== 세션 상태 ==============
if "clean_struct" not in st.session_state:
    st.session_state.clean_struct = None
if "q_hist" not in st.session_state:
    st.session_state.q_hist = []
if "records" not in st.session_state:
    st.session_state.records = []  # [{question, answer, overall, criteria:[..]}]
if "current_question" not in st.session_state:
    st.session_state.current_question = ""
if "answer_text" not in st.session_state:
    st.session_state.answer_text = ""

# ============== 1) 채용 공고 URL → 정제 ==============
st.header("1) 채용 공고 URL 입력 → 정제")
url = st.text_input("채용 공고 상세 URL", placeholder="예: https://www.wanted.co.kr/wd/123456")
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
            with st.spinner("LLM으로 정제 중..."):
                clean = llm_structurize(raw, hint, CHAT_MODEL)
            st.session_state.clean_struct = clean
            st.success("정제 완료!")

# ============== 2) 회사 요약 섹션 ==============
st.header("2) 회사 요약 (정제 결과)")
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

# ============== 3) 질문 생성 ==============
st.header("3) 질문 생성")
q_type = st.selectbox("질문 유형", ["행동(STAR)","기술 심층","핵심가치 적합성","역질문"], index=0)
level  = st.selectbox("난이도/연차", ["주니어","미들","시니어"], index=0)
seed   = st.number_input("랜덤시드", value=int(time.time())%1_000_000, step=1)
num    = st.slider("질문 개수", 4, 10, 8, 1)

if st.button("새 질문 받기", type="primary"):
    if not st.session_state.clean_struct:
        st.warning("먼저 URL을 정제하세요.")
    else:
        qs = llm_generate_questions(st.session_state.clean_struct, q_type, level, CHAT_MODEL, num=num, seed=int(seed))
        if qs:
            st.session_state.q_hist.extend(qs)
            # 가장 다양한 한 문장 선택
            st.session_state.current_question = random.choice(qs)
            st.session_state.answer_text = ""  # ✅ 이전 답변 초기화
            st.success("질문 생성 완료!")
        else:
            st.error("질문 생성 실패")

st.text_area("질문", value=st.session_state.current_question, height=100)

# ============== 4) 답변/채점/코칭 ==============
st.header("4) 나의 답변 → 채점 & 코칭")
ans = st.text_area("여기에 답변을 작성하세요 (STAR 권장: 상황-과제-행동-성과)", height=180, key="answer_text")

if st.button("채점 & 코칭", type="primary"):
    if not st.session_state.current_question:
        st.warning("먼저 질문을 생성하세요.")
    elif not st.session_state.answer_text.strip():
        st.warning("답변을 작성해 주세요.")
    else:
        with st.spinner("채점/코칭 중..."):
            res = llm_score_and_coach(st.session_state.clean_struct, st.session_state.current_question, st.session_state.answer_text, CHAT_MODEL)
        # 기록 저장 (총점=기준 합계)
        st.session_state.records.append({
            "question": st.session_state.current_question,
            "answer": st.session_state.answer_text,
            "overall": res.get("overall_score", 0),
            "criteria": res.get("criteria", []),
            "strengths": res.get("strengths", []),
            "risks": res.get("risks", []),
            "improvements": res.get("improvements", []),
            "revised_answer": res.get("revised_answer","")
        })

# ============== 5) 피드백 결과 표시 (정합성 보장) ==============
st.header("5) 피드백 결과")
if st.session_state.records:
    last = st.session_state.records[-1]
    left, right = st.columns([1,3])
    with left:
        st.metric("총점(/100)", last["overall"])
    with right:
        # 기준별 코멘트 + 감점/아쉬움/개선포인트 반영
        st.markdown("**기준별 점수 & 코멘트**")
        for it in last["criteria"]:
            st.markdown(f"- **{it['name']}**: {it['score']}/20 — {it.get('comment','')}")
        if last["strengths"]:
            st.markdown("**강점**")
            for s in last["strengths"]: st.markdown(f"- {s}")
        if last["risks"]:
            st.markdown("**감점 요인/리스크**")
            for r in last["risks"]: st.markdown(f"- {r}")
        if last["improvements"]:
            st.markdown("**개선 포인트**")
            for im in last["improvements"]: st.markdown(f"- {im}")
        if last["revised_answer"]:
            st.markdown("**수정본 답변 (STAR)**")
            st.write(last["revised_answer"])
else:
    st.info("아직 채점 결과가 없습니다.")

st.divider()

# ============== 6) 역량 레이더 (누적 + 평균) ==============
st.header("6) 역량 레이더 (세션 누적)")
def build_comp_table(records):
    rows=[]
    for idx, r in enumerate(records, 1):
        crit = r.get("criteria", [])
        row={"#": idx, "question": r.get("question",""), "overall": r.get("overall",0)}
        # 기준 맵 쉽게
        cm = {c["name"]: c["score"] for c in crit if "name" in c}
        for k in CRITERIA:
            row[k] = cm.get(k, 0)
        rows.append(row)
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["#","question","overall"]+CRITERIA)

df = build_comp_table(st.session_state.records)
if not df.empty:
    # 평균 / 누적합
    avg = [df[k].mean() for k in CRITERIA]
    cum = [df[k].sum() for k in CRITERIA]

    try:
        import plotly.graph_objects as go
        radar = go.Figure()
        radar.add_trace(go.Scatterpolar(
            r=avg + [avg[0]], theta=CRITERIA + [CRITERIA[0]],
            fill='toself', name='평균(0~20)'
        ))
        radar.add_trace(go.Scatterpolar(
            r=cum + [cum[0]], theta=CRITERIA + [CRITERIA[0]],
            fill='toself', name='누적(합계)'
        ))
        radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True, height=420)
        st.plotly_chart(radar, use_container_width=True)
    except Exception:
        st.bar_chart(pd.DataFrame({"평균":avg,"누적":cum}, index=CRITERIA))

    st.markdown("**세션 표(질문별 기준 점수)**")
    st.dataframe(df, use_container_width=True)
else:
    st.caption("아직 누적 데이터가 없습니다. 질문 생성→답변→채점을 진행하세요.")

st.divider()

# ============== 7) CSV 다운로드 ==============
st.header("7) 세션 리포트 다운로드")
def export_csv(records):
    rows=[]
    for r in records:
        base = {
            "question": r.get("question",""),
            "answer": r.get("answer",""),
            "overall": r.get("overall",0),
        }
        cm = {c["name"]: c["score"] for c in r.get("criteria",[])}
        for k in CRITERIA:
            base[f"comp_{k}"] = cm.get(k, 0)
        rows.append(base)
    return pd.DataFrame(rows).to_csv(index=False, encoding="utf-8-sig")
csv_data = export_csv(st.session_state.records)
st.download_button("CSV 다운로드", data=csv_data, file_name="interview_session.csv", mime="text/csv")

st.caption("‘새 질문 받기’ 클릭 시 답변 입력란은 자동 초기화됩니다. 총점은 기준(5×20) 합계와 항상 일치하도록 강제했습니다.")
