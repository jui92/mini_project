# -*- coding: utf-8 -*-
# ============================================
# 회사 특화 가상 면접 코치 (텍스트 전용, RAG+레ーダ+CSV)
# ============================================
# 의존: streamlit, openai(>=1.x), faiss-cpu, pypdf, plotly, pandas, numpy
# 실행:
#   pip install streamlit openai faiss-cpu pypdf plotly pandas numpy
#   streamlit run interview_coach_rag.py
# API 키: .streamlit/secrets.toml 또는 환경변수 OPENAI_API_KEY
# ============================================

import os, json, io, re, textwrap, time, tempfile
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# OpenAI SDK
try:
    from openai import OpenAI
except ImportError:
    st.error("`openai` 패키지가 필요합니다. 터미널에서 `pip install openai` 후 다시 실행하세요.")
    st.stop()

# 벡터 검색: FAISS
try:
    import faiss
except ImportError:
    faiss = None

# PDF 텍스트 추출
try:
    import pypdf
except ImportError:
    pypdf = None

# Radar chart (Plotly)
import plotly.graph_objects as go

# =========================
# Streamlit 기본 설정
# =========================
st.set_page_config(page_title="회사 특화 가상 면접 코치 (RAG+Radar+CSV)", page_icon="🎯", layout="wide")

# ===== 사이드바: 설정 =====
with st.sidebar:
    st.title("🎯 가상 면접 코치 (텍스트 전용)")
    st.caption("회사 맞춤 질문 + 문서 근거 RAG + 역량 레이더 + CSV 리포트")

    # API Key
    API_KEY = (
        st.secrets.get("OPENAI_API_KEY", None)
        if hasattr(st, "secrets") else None
    ) or os.getenv("OPENAI_API_KEY")

    if not API_KEY:
        API_KEY = st.text_input("OpenAI API Key", type="password")

    MODEL = st.selectbox("모델", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
    EMBED_MODEL = st.selectbox("임베딩 모델", ["text-embedding-3-small", "text-embedding-3-large"], index=0)

    st.divider()
    st.markdown("#### 회사/직무 설정")

    # 회사 요약 JSON 불러오기/업로드
    data_dir = "data/companies"
    os.makedirs(data_dir, exist_ok=True)

    # 기본 예시 파일 자동 생성
    defaults = {
        "acme.json": {
            "company_name": "ACME",
            "values": ["고객집착", "데이터기반", "주도적 실행"],
            "role": "데이터 애널리스트",
            "role_requirements": [
                "SQL/EDA 숙련", "비즈니스 임팩트 지표 설계", "대시보드/리포트 커뮤니케이션"
            ],
            "recent_projects": ["구독 전환 퍼널 최적화", "이탈 예측 모델 파일럿"],
            "language": "ko"
        },
        "contoso.json": {
            "company_name": "Contoso",
            "values": ["소유감", "협업", "고객 성공"],
            "role": "머신러닝 엔지니어",
            "role_requirements": [
                "모델링/서빙 파이프라인", "관측성/모니터링", "성능-비용 최적화"
            ],
            "recent_projects": ["추천 시스템 리랭킹", "A/B 테스트 플랫폼 개선"],
            "language": "ko"
        }
    }
    for fn, payload in defaults.items():
        path = os.path.join(data_dir, fn)
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

    files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    company_file = st.selectbox("회사 프로필 파일", files, index=0)

    uploaded_company = st.file_uploader("또는 회사 프로필 JSON 업로드", type=["json"])
    if uploaded_company is not None:
        try:
            company = json.load(uploaded_company)
        except Exception as e:
            st.error(f"회사 JSON 파싱 오류: {e}")
            st.stop()
    else:
        company = json.load(open(os.path.join(data_dir, company_file), "r", encoding="utf-8"))

    st.markdown("#### 질문 옵션")
    q_type = st.selectbox("질문 유형", ["행동(STAR)", "기술 심층", "핵심가치 적합성", "역질문"], index=0)
    level = st.selectbox("난이도/연차", ["주니어", "미들", "시니어"], index=0)

    st.markdown("#### RAG (선택)")
    rag_enabled = st.toggle("회사 문서 기반 질문/코칭 사용 (RAG)", value=True)
    chunk_size = st.slider("청크 길이(문자)", min_value=400, max_value=2000, value=900, step=100)
    chunk_overlap = st.slider("오버랩(문자)", min_value=0, max_value=400, value=150, step=10)
    top_k = st.slider("검색 상위 K", min_value=1, max_value=8, value=4, step=1)

    st.caption("※ TXT/MD/PDF 업로드 가능(PDF는 텍스트 추출). 파일은 세션 메모리에만 저장됩니다.")
    docs = st.file_uploader("회사 문서 업로드 (여러 파일 가능)", type=["txt", "md", "pdf"], accept_multiple_files=True)

# =========================
# 공용 함수
# =========================
def build_company_context(c: dict) -> str:
    return textwrap.dedent(f"""
    [회사명] {c.get('company_name','')}
    [핵심가치] {", ".join(c.get('values', []))}
    [직무] {c.get('role','')}
    [주요 요구역량] {", ".join(c.get('role_requirements', []))}
    [최근 프로젝트] {", ".join(c.get('recent_projects', []))}
    """)

def read_file_to_text(uploaded) -> str:
    name = uploaded.name.lower()
    data = uploaded.read()
    if name.endswith((".txt", ".md")):
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return data.decode("cp949", errors="ignore")
    elif name.endswith(".pdf"):
        if pypdf is None:
            st.warning("pypdf가 설치되어야 PDF 텍스트 추출이 가능합니다. `pip install pypdf`")
            return ""
        try:
            reader = pypdf.PdfReader(io.BytesIO(data))
            pages = []
            for i in range(len(reader.pages)):
                try:
                    pages.append(reader.pages[i].extract_text() or "")
                except Exception:
                    pages.append("")
            return "\n\n".join(pages)
        except Exception as e:
            st.warning(f"PDF 파싱 실패({uploaded.name}): {e}")
            return ""
    else:
        return ""

def chunk_text(text: str, size: int = 900, overlap: int = 150) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(text):
            break
    return chunks

def embed_texts(client: OpenAI, embed_model: str, texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    # OpenAI SDK 1.x: embeddings.create
    resp = client.embeddings.create(model=embed_model, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    return vecs

def build_faiss_index(embeds: np.ndarray):
    if faiss is None:
        return None
    d = embeds.shape[1]
    index = faiss.IndexFlatIP(d)  # Inner Product (cosine 위해 정규화 전제)
    # 정규화
    norms = np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-12
    normed = embeds / norms
    index.add(normed)
    return index, normed

def faiss_search(index, normed_embeds: np.ndarray, query_vec: np.ndarray, k: int = 4):
    qn = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-12)
    D, I = index.search(qn, k)
    return D[0], I[0]

def cosine_topk(matrix: np.ndarray, query: np.ndarray, k: int = 4):
    # fallback (faiss 없는 경우)
    qn = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-12)
    mn = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    sims = mn @ qn.T
    sims = sims.reshape(-1)
    idx = np.argsort(-sims)[:k]
    scores = sims[idx]
    return scores, idx

# =========================
# OpenAI 클라이언트
# =========================
if not API_KEY:
    st.warning("API Key를 입력하면 바로 사용 가능합니다.")
    st.stop()
client = OpenAI(api_key=API_KEY)

# =========================
# RAG 준비: 문서→청크→임베딩→인덱스
# =========================
if "rag_store" not in st.session_state:
    st.session_state.rag_store = {
        "chunks": [],          # List[str]
        "embeds": None,        # np.ndarray
        "index": None          # FAISS index (or None)
    }

if rag_enabled and docs:
    with st.spinner("문서 처리 중..."):
        all_chunks, src_map = [], []  # src_map: (filename, chunk_id)
        for up in docs:
            text = read_file_to_text(up)
            if not text:
                continue
            chs = chunk_text(text, chunk_size, chunk_overlap)
            all_chunks.extend(chs)
            src_map.extend([(up.name, i) for i in range(len(chs))])

        if all_chunks:
            embeds = embed_texts(client, EMBED_MODEL, all_chunks)
            st.session_state.rag_store["chunks"] = all_chunks
            st.session_state.rag_store["embeds"] = embeds
            if faiss is not None and embeds.shape[0] > 0:
                index, normed = build_faiss_index(embeds)
                st.session_state.rag_store["index"] = (index, normed)
            else:
                st.session_state.rag_store["index"] = None
            st.success(f"RAG 준비 완료: 청크 {len(all_chunks)}개")
        else:
            st.info("업로드 문서에서 추출된 텍스트가 없습니다.")

# =========================
# 프롬프트 빌더
# =========================
def gen_question(company: dict, qtype: str, level: str, supports: List[Tuple[str, float, str]]) -> str:
    # supports: [(source_name, score, text_chunk)]
    ctx = build_company_context(company)
    rag_note = ""
    if supports:
        joined = "\n\n".join([f"- ({s:.3f}) {src} :: {txt[:400]}" for (src, s, txt) in supports])
        rag_note = f"\n[회사 근거 문서 발췌]\n{joined}\n"

    sys = f"""너는 '{company.get('company_name','')}'의 '{company.get('role','')}' 면접관이다.
회사 맥락과 (있다면) 아래 근거 문서를 반영하여 {qtype} 유형의 **구체적 질문 1개만** 한국어로 생성하라.
지원자가 행동/지표/임팩트를 드러내도록 하며, 난이도/연차는 {level}에 맞춘다.
서론/부연 금지. 결과는 질문 문장 한 줄만."""
    user = f"""[회사 컨텍스트]
{ctx}
{rag_note}"""

    resp = client.chat.completions.create(
        model=MODEL, temperature=0.7,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}]
    )
    return resp.choices[0].message.content.strip()

def coach_answer(company: dict, question: str, user_answer: str, supports: List[Tuple[str, float, str]]) -> Dict:
    ctx = build_company_context(company)
    rag_note = ""
    if supports:
        joined = "\n\n".join([f"- ({s:.3f}) {src} :: {txt[:600]}" for (src, s, txt) in supports])
        rag_note = f"\n[회사 근거 문서 발췌]\n{joined}\n"

    # 역량 레이더용 카테고리: 0~5점씩 요청 (정수)
    competencies = [
        "문제정의", "데이터/지표", "실행력/주도성", "협업/커뮤니케이션", "고객가치"
    ]
    comp_str = ", ".join(competencies)

    sys = f"""너는 톱티어 면접 코치다. 한국어로 아래 형식에 맞춰 답하라:
1) 총점: 0~10 정수 1개
2) 강점: 2~3개 불릿
3) 리스크: 2~3개 불릿
4) 개선 포인트: 3개 불릿 (행동·지표·임팩트 중심)
5) 수정본 답변: STAR(상황-과제-행동-성과) 구조로 자연스럽고 간결하게
6) 역량 점수: [{comp_str}] 각각 0~5 정수 (한 줄에 쉼표로 구분)
추가 설명은 금지. 형식을 유지하라."""
    user = f"""[회사 컨텍스트]
{ctx}
{rag_note}
[면접 질문]
{question}

[후보자 답변]
{user_answer}
"""

    resp = client.chat.completions.create(
        model=MODEL, temperature=0.4,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}]
    )
    content = resp.choices[0].message.content.strip()

    # 총점 파싱 (첫 번째 0-10 정수)
    m = re.search(r'([0-9]{1,2})\s*(?:/10|점|$)', content)
    score = None
    if m:
        try:
            score = int(m.group(1))
            score = max(0, min(10, score))
        except:
            pass

    # 역량 점수 파싱: "역량 점수:" 라인이 있다면 숫자 5개 추출
    comp_scores = None
    comp_line = None
    for line in content.splitlines():
        if "역량" in line and any(k in line for k in ["점수", "점"]):
            comp_line = line
            break
    if comp_line is None:
        # 마지막 줄 가정
        comp_line = content.splitlines()[-1]
    nums = re.findall(r'\b([0-5])\b', comp_line)
    if len(nums) >= 5:
        comp_scores = [int(x) for x in nums[:5]]

    return {"raw": content, "score": score, "competencies": comp_scores}

# =========================
# 세션 상태
# =========================
if "history" not in st.session_state:
    st.session_state.history = []  # [{q, a, score, feedback, supports, comp_scores, ts}]

if "current_question" not in st.session_state:
    st.session_state.current_question = ""

# =========================
# 검색 유틸
# =========================
def retrieve_supports(query_text: str, k: int) -> List[Tuple[str, float, str]]:
    """질문 생성/코칭 전에 RAG 컨텍스트로 쓸 근거 청크를 반환
       리턴: [(source, score, chunk_text), ...]"""
    store = st.session_state.rag_store
    chunks = store.get("chunks", [])
    embeds = store.get("embeds", None)
    if not rag_enabled or embeds is None or len(chunks) == 0:
        return []
    qv = embed_texts(client, EMBED_MODEL, [query_text])
    if store["index"] and faiss is not None:
        index, normed = store["index"]
        scores, idxs = faiss_search(index, normed, qv, k=k)
    else:
        scores, idxs = cosine_topk(embeds, qv, k=k)
    out = []
    for s, i in zip(scores, idxs):
        out.append(("업로드문서", float(s), chunks[int(i)]))
    return out

# =========================
# UI 본문
# =========================
left, right = st.columns([1, 1])

with left:
    st.header("① 질문 생성")
    st.markdown("**선택한 회사 요약**")
    st.json(company, expanded=False)

    prompt_hint = st.text_input("질문 생성 힌트(선택)", placeholder="예: 구독 전환 퍼널 관련 경험 위주로 물어봐줘")
    if st.button("새 질문 받기", use_container_width=True):
        try:
            supports = []
            if rag_enabled and (docs or st.session_state.rag_store.get("chunks")):
                # 힌트가 있으면 힌트로 검색, 없으면 회사 role과 values로 검색
                base_q = prompt_hint.strip() or f"{company.get('role','')} {', '.join(company.get('values', []))}"
                supports = retrieve_supports(base_q, top_k)
            q = gen_question(company, q_type, level, supports)
            st.session_state.current_question = q
            st.session_state.last_supports_q = supports
        except Exception as e:
            st.error(f"질문 생성 오류: {e}")

    st.text_area("질문", height=110, value=st.session_state.get("current_question",""))

    # 근거 표시 (질문 생성에 사용)
    if rag_enabled and st.session_state.get("last_supports_q"):
        with st.expander("질문 생성에 사용된 근거 보기"):
            for i, (src, sc, txt) in enumerate(st.session_state.last_supports_q, 1):
                st.markdown(f"**[{i}] {src} (sim={sc:.3f})**\n\n{txt[:600]}{'...' if len(txt)>600 else ''}")
                st.markdown("---")

with right:
    st.header("② 나의 답변")
    answer = st.text_area("여기에 답변을 작성하세요 (가능하면 STAR: 상황-과제-행동-성과)", height=160)

    # 코칭 실행
    if st.button("채점 & 코칭", type="primary", use_container_width=True):
        if not st.session_state.get("current_question"):
            st.warning("먼저 '새 질문 받기'로 질문을 생성하세요.")
        elif not answer.strip():
            st.warning("답변을 작성해 주세요.")
        else:
            with st.spinner("코칭 중..."):
                try:
                    # 코칭용 RAG: 질문+답변을 합쳐 질의로 사용
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

st.divider()
st.subheader("③ 피드백 결과")
if st.session_state.history:
    last = st.session_state.history[-1]
    c1, c2 = st.columns([1, 3])
    with c1:
        st.metric("총점(/10)", last.get("score", "—"))
    with c2:
        st.markdown(last.get("feedback", ""))

    # 코칭에 사용된 근거
    if rag_enabled and last.get("supports"):
        with st.expander("코칭에 사용된 근거 보기"):
            for i, (src, sc, txt) in enumerate(last["supports"], 1):
                st.markdown(f"**[{i}] {src} (sim={sc:.3f})**\n\n{txt[:800]}{'...' if len(txt)>800 else ''}")
                st.markdown("---")

# =========================
# 역량 레이더 차트 (누적)
# =========================
st.divider()
st.subheader("④ 역량 레이더 (세션 누적)")
competencies = ["문제정의", "데이터/지표", "실행력/주도성", "협업/커뮤니케이션", "고객가치"]

def compute_competency_df(hist):
    rows = []
    for h in hist:
        if h.get("competencies") and len(h["competencies"]) == 5:
            rows.append(h["competencies"])
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=competencies)
    return df

comp_df = compute_competency_df(st.session_state.history)
if comp_df is not None:
    avg_scores = comp_df.mean().values.tolist()
    # Radar
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=avg_scores + [avg_scores[0]],
        theta=competencies + [competencies[0]],
        fill='toself',
        name='평균 점수(0~5)'
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,5])), showlegend=False, height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(comp_df, use_container_width=True)
else:
    st.info("아직 역량 점수를 파싱할 수 있는 코칭 결과가 없습니다. (코칭 실행 시 자동 수집)")

# =========================
# 세션 리포트 (CSV)
# =========================
st.divider()
st.subheader("⑤ 세션 리포트 다운로드 (CSV)")

def build_report_df(hist):
    out = []
    for h in hist:
        row = {
            "timestamp": h.get("ts"),
            "question": h.get("question"),
            "user_answer": h.get("user_answer"),
            "score": h.get("score"),
            "feedback_raw": h.get("feedback")
        }
        comps = h.get("competencies")
        if comps and len(comps) == 5:
            for k, v in zip(competencies, comps):
                row[f"comp_{k}"] = v
        # 간단히 근거 텍스트 앞부분만 묶어서 기록
        sups = h.get("supports") or []
        row["supports_preview"] = " || ".join([s[2][:120].replace("\n"," ") for s in sups])
        out.append(row)
    if not out:
        return pd.DataFrame(columns=["timestamp","question","user_answer","score","feedback_raw","supports_preview"])
    return pd.DataFrame(out)

report_df = build_report_df(st.session_state.history)
csv_bytes = report_df.to_csv(index=False).encode("utf-8-sig")
st.download_button("CSV로 다운로드", data=csv_bytes, file_name="interview_session_report.csv", mime="text/csv")

# =========================
# 푸터
# =========================
st.caption("Tip: 회사 JSON과 문서를 풍부하게 넣을수록 질문/코칭의 회사 정합성이 올라갑니다. (RAG 근거는 상단에서 확인 가능)")
