# -*- coding: utf-8 -*-
# interview_coach_bot_2.py
# -----------------------------------------------------------
# 변경 요약
# - 총점 일원화: 역량 5축(0~20)의 평균×5 → 0~100으로 산출하여
#   우측 피드백 본문 첫 줄 '총점:'을 강제 치환(좌/우 항상 동일)
# - 레이더 표: '합계'(5축 합, 0~100) 컬럼 추가 + 값 보정(0~20)
# - 속도 개선: 임베딩 캐시(@st.cache_data), 질문/답변 토큰 다이어트
# - 앱 구조: 회사/직무 입력 → (선택)RAG 인덱싱 → 질문 생성 → 채점/코칭 → 시각화/CSV
# -----------------------------------------------------------

import os, re, io, difflib, random, textwrap
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# Optional: Plotly(레이더 그래프), 없으면 bar 차트로 대체
try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# OpenAI SDK
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# -----------------------------
# 페이지 설정
# -----------------------------
st.set_page_config(page_title="회사 특화 모의 면접 코치", page_icon="🎯", layout="wide")

# -----------------------------
# Key 로딩
# -----------------------------
def load_api_key() -> Optional[str]:
    k = os.getenv("OPENAI_API_KEY")
    if k: return k
    try:
        return st.secrets.get("OPENAI_API_KEY", None)
    except Exception:
        return None

# -----------------------------
# 유틸
# -----------------------------
def _clean(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip()

def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()

def chunk_text(text: str, size: int = 900, overlap: int = 150) -> List[str]:
    """간단 청크(업로드 문서용)."""
    t = re.sub(r"\s+", " ", text or "").strip()
    if not t: return []
    out, i = [], 0
    while i < len(t):
        j = min(len(t), i + size)
        out.append(t[i:j])
        if j == len(t): break
        i = max(0, j - overlap)
    return out

# -----------------------------
# 사이드바: 설정
# -----------------------------
with st.sidebar:
    st.title("⚙️ 설정")
    api_key = load_api_key()
    if not api_key:
        st.info("환경변수 또는 secrets.toml에 OPENAI_API_KEY가 없으면 아래에 입력하세요.")
        api_key = st.text_input("OPENAI_API_KEY", type="password")

    MODEL = st.selectbox("챗 모델", ["gpt-4o-mini", "gpt-4o"], index=0)
    EMBED_MODEL = st.selectbox("임베딩 모델", ["text-embedding-3-small", "text-embedding-3-large"], index=0)

    with st.expander("디버그: 상태/버전"):
        st.write({"api_key_loaded": bool(api_key), "model": MODEL, "embed": EMBED_MODEL})

if not api_key or OpenAI is None:
    st.error("OpenAI API Key가 필요합니다. (requirements.txt에 openai 포함)")
    st.stop()

client = OpenAI(api_key=api_key)

# -----------------------------
# 캐시(속도 개선)
# -----------------------------
@st.cache_data(ttl=3600)
def cached_embeddings(model: str, texts: List[str]) -> np.ndarray:
    """동일 텍스트 재임베딩 방지."""
    if not texts: return np.zeros((0, 3), dtype=np.float32)
    r = client.embeddings.create(model=model, input=texts)
    return np.array([d.embedding for d in r.data], dtype=np.float32)

def embed_texts(texts: List[str]) -> np.ndarray:
    return cached_embeddings(EMBED_MODEL, texts)

def cosine_topk(mat: np.ndarray, q: np.ndarray, k: int = 4):
    if mat.size == 0: return np.array([]), np.array([], dtype=int)
    mn = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
    qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    sims = mn @ qn.T
    sims = sims.reshape(-1)
    idx = np.argsort(-sims)[:k]
    return sims[idx], idx

# -----------------------------
# 세션 상태 초기화
# -----------------------------
default_state = {
    "company": {"name": "", "homepage": "", "role": ""},
    "rag_store": {"chunks": [], "embeds": None},
    "current_question": "",
    "answer_text": "",
    "history": [],  # [{ts, question, user_answer, score, feedback, competencies(list[int]|None)}]
}
for k, v in default_state.items():
    if k not in st.session_state:
        st.session_state[k] = v

# -----------------------------
# ① 회사/직무 입력
# -----------------------------
st.subheader("① 회사/직무 입력")
c_col, r_col = st.columns([2, 1])
with c_col:
    company_name = st.text_input("회사 이름", placeholder="예: 네이버 / 카카오 / 삼성SDS",
                                 value=st.session_state["company"]["name"])
with r_col:
    role_title = st.text_input("지원 직무명", placeholder="데이터 애널리스트 / ML 엔지니어 ...",
                               value=st.session_state["company"]["role"])

home_url = st.text_input("홈페이지 URL(선택)", placeholder="https://...",
                         value=st.session_state["company"]["homepage"])

if st.button("회사/직무 정보 불러오기", type="primary"):
    st.session_state["company"] = {
        "name": company_name.strip(),
        "homepage": home_url.strip(),
        "role": role_title.strip()
    }
    # ↓ 결과 초기화
    st.session_state["rag_store"] = {"chunks": [], "embeds": None}
    st.session_state["current_question"] = ""
    st.session_state["answer_text"] = ""
    st.session_state["history"] = []
    st.success("회사 정보 갱신 및 결과 초기화 완료")

if st.session_state["company"]["name"]:
    st.markdown(f"- **회사명**: {st.session_state['company']['name']}"
                f" / **직무**: {st.session_state['company']['role'] or '—'}")
    if st.session_state["company"]["homepage"]:
        st.markdown(f"- **홈페이지**: {st.session_state['company']['homepage']}")

# -----------------------------
# ② RAG 옵션(선택)
# -----------------------------
st.subheader("② RAG 옵션(선택)")
with st.expander("회사 문서 업로드 / 인덱싱"):
    rag_on = st.toggle("회사 문서 기반 질문/코칭 사용", value=True, key="rag_on")
    topk = st.slider("검색 상위 K", 1, 8, 4, 1)
    ups = st.file_uploader("회사 문서 업로드 (TXT/MD/PDF)", type=["txt", "md", "pdf"], accept_multiple_files=True)
    size = st.slider("청크 길이", 400, 2000, 900, 100)
    ovlp = st.slider("오버랩", 0, 400, 150, 10)

    if ups:
        chunks = []
        for u in ups:
            raw = u.read()
            name = u.name.lower()
            text = ""
            if name.endswith((".txt", ".md")):
                for enc in ("utf-8", "cp949", "euc-kr"):
                    try:
                        text = raw.decode(enc)
                        break
                    except Exception:
                        text = ""
                if not text:
                    text = raw.decode("utf-8", errors="ignore")
            elif name.endswith(".pdf"):
                try:
                    import pypdf
                    reader = pypdf.PdfReader(io.BytesIO(raw))
                    text = "\n\n".join([(reader.pages[i].extract_text() or "") for i in range(len(reader.pages))])
                except Exception:
                    text = ""
            if text:
                chunks += chunk_text(text, size=size, overlap=ovlp)
        if chunks:
            embs = embed_texts(chunks)
            st.session_state["rag_store"] = {"chunks": chunks, "embeds": embs}
            st.success(f"인덱싱 완료: 청크 {len(chunks)}개")

# -----------------------------
# ③ 질문 생성
# -----------------------------
st.subheader("③ 질문 생성")
TYPE_INSTR = {
    "행동(STAR)": "과거 실무 사례를 이끌어내는 STAR 행동 질문",
    "기술 심층": "성능/비용/지연/정확도/운영까지 파고드는 기술 심층 질문",
    "핵심가치 적합성": "가치관/태도/협업 스타일을 검증하는 질문",
    "역질문": "지원자가 회사를 평가하는 역질문",
}
q_type = st.selectbox("질문 유형", list(TYPE_INSTR.keys()))
level = st.selectbox("난이도/연차", ["주니어", "미들", "시니어"])
hint = st.text_input("질문 생성 힌트(선택)", placeholder="예: 전환 퍼널 / 모델 성능-비용 / 데이터 품질")

def retrieve_supports(qtext: str, k: int) -> List[Tuple[str, float, str]]:
    store = st.session_state["rag_store"]
    chs, embs = store.get("chunks", []), store.get("embeds")
    if not st.session_state.get("rag_on") or embs is None or not chs:
        return []
    qv = embed_texts([qtext])
    s, idx = cosine_topk(embs, qv, k=k)
    return [("회사자료", float(sc), chs[int(i)]) for sc, i in zip(s, idx)]

def choose_diverse(cands: List[str], history: List[str]) -> str:
    """이전 질문과 유사하지 않게 한 개 선택."""
    if not cands: return ""
    if not history: return random.choice(cands)
    best, best_s = None, 1e9
    for q in cands:
        sims = [similarity(q, h) for h in history] or [0.0]
        s = (sum(sims)/len(sims)) + 0.35*np.std(sims)
        if s < best_s:
            best_s, best = s, q
    return best

if st.button("새 질문 받기", type="primary", use_container_width=True):
    st.session_state["answer_text"] = ""  # 이전 답변 비우기(요청 반영)
    try:
        company = st.session_state["company"]
        ctx_lines = [
            f"[회사명] {company.get('name','')}",
            f"[직무] {company.get('role','')}",
        ]
        ctx = "\n".join(ctx_lines)

        focuses = []
        if hint.strip(): focuses.append(hint.strip())
        if company.get("role"): focuses.append(company["role"])

        # 선택적 RAG 기반 키워드 보강
        supports = []
        if st.session_state.get("rag_on"):
            supports = retrieve_supports(hint or company.get("role", ""), k=topk)
            for _, _, txt in supports[:3]:
                for frag in re.split(r"[•\-\n\.]", txt):
                    frag = frag.strip()
                    if 6 < len(frag) < 80:
                        focuses.append(frag)
        focuses = list(dict.fromkeys(focuses))[:6]

        sys = f"""너는 '{q_type}' 유형({TYPE_INSTR[q_type]})의 질문 6개를 한국어로 생성한다.
각 질문은 관점/형태/키워드를 다르게 하고, 난이도는 {level}에 맞춘다.
포맷: 1) ... 2) ... 3) ... (한 줄씩)"""
        user = f"[컨텍스트]\n{ctx}\n[포커스]\n- " + "\n- ".join(focuses) if focuses else f"[컨텍스트]\n{ctx}"

        r = client.chat.completions.create(model=MODEL, temperature=0.95,
                                           messages=[{"role":"system","content":sys},
                                                     {"role":"user","content":user}])
        raw = r.choices[0].message.content.strip()
        cands = [re.sub(r'^\s*\d+\)\s*', '', ln).strip()
                 for ln in raw.splitlines() if re.match(r'^\s*\d+\)', ln)]
        if not cands:
            cands = [ln.strip("- ").strip() for ln in raw.splitlines() if ln.strip()][:6]

        hist_q = [h["question"] for h in st.session_state["history"]][-10:]
        st.session_state["current_question"] = choose_diverse(cands, hist_q) or cands[0]
        st.session_state["last_supports_q"] = supports
    except Exception as e:
        st.error(f"질문 생성 오류: {e}")

st.text_area("질문", height=110, value=st.session_state.get("current_question", ""))

# -----------------------------
# ④ 답변/코칭 & 점수 일원화
# -----------------------------
st.subheader("④ 나의 답변 / 코칭")
AXES = ["문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치"]

def coach_answer(company: dict, question: str, answer: str,
                 supports: List[Tuple[str, float, str]]) -> dict:
    # 토큰 다이어트: 너무 길면 잘라서 사용
    q_trim = (question or "")[:500]
    a_trim = (answer or "")[:1200]

    rag = ""
    if supports:
        rag = "\n[회사 근거(발췌)]\n" + "\n".join([f"- ({s:.3f}) {txt[:300]}" for _, s, txt in supports])

    sys = f"""너는 한국어 면접 코치다. 아래 형식만 출력:
1) 총점: NN/100
2) 강점: • 2~3개
3) 리스크: • 2~3개
4) 개선 포인트: • 3개 (행동·지표·임팩트 중심)
5) 수정본 답변: STAR(상황-과제-행동-성과)로 간결히
6) 역량 점수(각 0~20, 비적용은 '-' 그대로): [{', '.join(AXES)}] — 5개 값을 쉼표로 출력
"""
    user = f"""[회사/직무] {company.get('name','')} / {company.get('role','')}
{rag}
[면접 질문]
{q_trim}

[후보자 답변]
{a_trim}
"""

    resp = client.chat.completions.create(model=MODEL, temperature=0.35,
                                          messages=[{"role":"system","content":sys},
                                                    {"role":"user","content":user}])
    content = resp.choices[0].message.content.strip()

    # (A) 모델이 본문에 쓴 총점(있으면 파싱)
    score = None
    m = re.search(r'총점\s*[:：]\s*(\d{1,3})', content)
    if m:
        score = max(0, min(100, int(m.group(1))))

    # (B) 역량 5개(0~20) 파싱
    last_line = content.splitlines()[-1] if content.splitlines() else ""
    nums = re.findall(r'\b(\d{1,2})\b', last_line)
    if len(nums) < 5:
        nums = re.findall(r'\b(\d{1,2})\b', content)
    competencies = None
    if len(nums) >= 5:
        cand = [int(x) for x in nums[:5]]
        # 0~5 또는 0~10로 나올 때 보정
        if all(0 <= x <= 5 for x in cand):
            cand = [x*4 for x in cand]
        elif all(0 <= x <= 10 for x in cand) and any(x > 5 for x in cand):
            cand = [x*2 for x in cand]
        competencies = [max(0, min(20, x)) for x in cand]

    # ✅ 최종 총점: 역량 평균×5가 우선, 없으면 모델 총점, 둘 다 없으면 0
    if competencies and len(competencies) == 5:
        final_score = int(round(sum(competencies) / 5.0 * 5))
    else:
        final_score = score if score is not None else 0

    # ✅ 우측 본문 첫 줄의 '총점:' 강제 치환 → 좌/우 항상 동일
    lines = content.splitlines()
    replaced = False
    for i, L in enumerate(lines[:4]):  # 상단 3~4줄만 변경
        if "총점" in L:
            lines[i] = re.sub(r"총점\s*[:：]\s*\d{1,3}(?:\s*/\s*100)?",
                              f"총점: {final_score}/100", L)
            replaced = True
            break
    if not replaced:
        lines.insert(0, f"총점: {final_score}/100")
    content_fixed = "\n".join(lines)

    return {
        "raw": content_fixed,         # 우측에 보여줄 피드백(총점 교체 완료)
        "score": final_score,         # 좌측 metric에 보여줄 총점
        "competencies": competencies  # [0..20]*5 또는 None
    }

ans = st.text_area("여기에 답변을 작성하세요 (STAR 권장: 상황-과제-행동-성과)",
                   height=180, key="answer_text")

if st.button("채점 & 코칭", type="primary", use_container_width=True):
    if not st.session_state.get("current_question"):
        st.warning("먼저 '새 질문 받기'로 질문을 생성하세요.")
    elif not st.session_state["answer_text"].strip():
        st.warning("답변을 작성해 주세요.")
    else:
        with st.spinner("코칭 중..."):
            # RAG 근거(질문+답변 일부 기반)
            sups = []
            if st.session_state.get("rag_on"):
                q_for_rag = (st.session_state["current_question"][:500]
                             + "\n" + st.session_state["answer_text"][:800])
                sups = retrieve_supports(q_for_rag, k=topk)

            res = coach_answer(st.session_state["company"],
                               st.session_state["current_question"],
                               st.session_state["answer_text"],
                               sups)

            st.session_state["history"].append({
                "ts": pd.Timestamp.now(),
                "question": st.session_state["current_question"],
                "user_answer": st.session_state["answer_text"],
                "score": res["score"],
                "feedback": res["raw"],
                "competencies": res["competencies"],  # [0..20]*5 or None
            })

# -----------------------------
# ⑤ 피드백 결과(좌/우 총점 동일)
# -----------------------------
st.divider()
st.subheader("피드백 결과")

if st.session_state["history"]:
    last = st.session_state["history"][-1]
    total = last["score"]
    c1, c2 = st.columns([1, 3])
    with c1:
        st.metric("총점(/100)", total)
    with c2:
        # 우측 본문 첫 줄의 총점도 위와 동일하게 치환되어 있음
        st.markdown(last["feedback"])
else:
    st.info("아직 결과가 없습니다.")

# -----------------------------
# ⑥ 역량 레이더 (세션 누적) + 합계 컬럼
# -----------------------------
st.divider()
st.subheader("역량 레이더 (세션 누적)")

COMP_AXES = ["문제정의","데이터/지표","실행력/주도성","협업/커뮤니케이션","고객가치"]

def comp_df(history):
    rows = []
    for h in history:
        cs = h.get("competencies")
        if not cs or len(cs) != 5:
            continue
        fixed = []
        for v in cs:
            try:
                x = int(v)
            except Exception:
                x = 0
            fixed.append(max(0, min(20, x)))  # 0~20 보정
        rows.append(fixed)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=COMP_AXES)
    df["합계"] = df[COMP_AXES].sum(axis=1)  # 5축 합: 0~100
    return df

cdf = comp_df(st.session_state["history"])
if cdf is not None:
    avg = cdf[COMP_AXES].mean().tolist()  # 0~20
    avg = [float(x) for x in avg]
    if PLOTLY_OK:
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=avg + [avg[0]], theta=COMP_AXES + [COMP_AXES[0]], fill='toself'))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 20])),
                          showlegend=False, height=420)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(pd.DataFrame({"score": avg}, index=COMP_AXES))

    # ✅ 합계 포함 테이블
    st.dataframe(cdf, use_container_width=True)
else:
    st.caption("아직 역량 점수가 파싱된 코칭 결과가 없습니다.")

# -----------------------------
# ⑦ 세션 리포트(CSV)
# -----------------------------
st.divider()
st.subheader("세션 리포트 (CSV)")

def build_report(history):
    rows = []
    for h in history:
        row = {
            "timestamp": h["ts"],
            "question": h["question"],
            "user_answer": h["user_answer"],
            "score": h["score"],
            "feedback_raw": h["feedback"]
        }
        cs = h.get("competencies") or []
        for k, v in zip(COMP_AXES, cs[:5]):
            row[f"comp_{k}"] = v
        row["comp_sum"] = sum([int(v) for v in cs[:5] if isinstance(v, (int, float))])
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["timestamp","question","user_answer","score","feedback_raw"])
    return pd.DataFrame(rows)

rep = build_report(st.session_state["history"])
st.download_button("CSV 다운로드",
                   data=rep.to_csv(index=False).encode("utf-8-sig"),
                   file_name="interview_session_report.csv",
                   mime="text/csv")

st.caption("총점 일원화(역량 평균×5) 적용, 레이더 합계 컬럼 추가, 캐시/토큰 다이어트로 속도 개선")
