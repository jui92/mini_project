# -*- coding: utf-8 -*-
# ============================================
# 회사 특화 가상 면접 코치 (텍스트 전용 / RAG + 레이더 + CSV)
# - Streamlit Cloud 호환 (faiss 미사용 / plotly 선택적)
# - 안전한 시크릿 로더(환경변수 → secrets → 사이드바 입력)
# - "회사 이름만"으로 자동 프로필 생성 (Wikipedia + 홈페이지 요약) + RAG 자동 투입(옵션)
# ============================================

import os, io, re, json, textwrap
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------
# PDF 텍스트 추출 (optional)
# ------------------------------
try:
    import pypdf
except Exception:
    pypdf = None

# ------------------------------
# Plotly (optional, 없으면 막대그래프 fallback)
# ------------------------------
try:
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# ------------------------------
# OpenAI SDK (>=1.x)
# ------------------------------
try:
    from openai import OpenAI
except ImportError:
    st.error("`openai` 패키지가 필요합니다. requirements.txt에 openai를 추가했는지 확인하세요.")
    st.stop()

# ------------------------------
# Auto profile helpers (Wikipedia + simple site scrape)
# ------------------------------
import requests
from bs4 import BeautifulSoup
try:
    import wikipedia
    try:
        wikipedia.set_lang("ko")
    except Exception:
        pass
except Exception:
    wikipedia = None  # wikipedia 패키지가 없으면 Wikipedia 경로는 건너뜀

VAL_KEYWORDS = [
    "핵심가치","가치","미션","비전","문화","원칙","철학","고객","데이터","혁신",
    "values","mission","vision","culture","principles","philosophy","customer","data","innovation"
]

def _clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()

def fetch_wikipedia_summary(company_name: str) -> dict | None:
    if wikipedia is None:
        return None
    try:
        page = wikipedia.page(company_name, auto_suggest=True, redirect=True)
        summary = _clean_text((page.summary or "").split("\n")[0])
        title = page.title
        return {
            "company_name": title,
            "values": [],
            "role": "",
            "role_requirements": [],
            "recent_projects": [],
            "wiki_summary": summary,
            "homepage": None
        }
    except Exception:
        return None

def fetch_site_snippets(base_url: str | None) -> dict:
    """회사 홈페이지의 /about, /values, /mission 등에서 짧은 문장 추출"""
    if not base_url:
        return {"values": [], "recent": []}
    url0 = base_url.strip()
    if not url0.startswith("http"):
        url0 = "https://" + url0
    cand_paths = ["", "/", "/about", "/company", "/about-us", "/mission", "/values", "/culture", "/kr/about"]
    values_found, recent_found = [], []
    for path in cand_paths:
        url = url0.rstrip("/") + path
        try:
            r = requests.get(url, timeout=6)
            if r.status_code != 200 or "text/html" not in r.headers.get("content-type", ""):
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup.find_all(["h1","h2","h3","p","li"]):
                txt = _clean_text(tag.get_text(separator=" "))
                if 10 <= len(txt) <= 240:
                    if any(k.lower() in txt.lower() for k in VAL_KEYWORDS):
                        values_found.append(txt)
                    if any(k in txt for k in ["프로젝트","개발","출시","성과","project","launched","release","delivered","improved"]):
                        recent_found.append(txt)
        except Exception:
            continue
    # 중복 제거 + 길이 정리
    def dedup(lst): 
        seen, out = set(), []
        for x in lst:
            if x not in seen:
                seen.add(x); out.append(x)
        return out
    values_found = dedup(values_found)[:5]
    recent_found = dedup(recent_found)[:5]

    # values 길면 콜론/쉼표로 쪼개 키워드 추출
    trimmed = []
    for v in values_found:
        v2 = v
        if ":" in v2: v2 = v2.split(":",1)[1]
        if len(v2) > 60 and "," in v2:
            parts = [p.strip() for p in v2.split(",") if 2 <= len(p.strip()) <= 24]
            if parts:
                trimmed.extend(parts[:6]); continue
        trimmed.append(v2[:60])
    values_found = trimmed[:6]

    return {"values": values_found, "recent": recent_found}

def build_auto_company(company_name: str, homepage: str | None) -> dict:
    base = fetch_wikipedia_summary(company_name) or {
        "company_name": company_name,
        "values": [],
        "role": "",
        "role_requirements": [],
        "recent_projects": [],
        "wiki_summary": None,
        "homepage": homepage or None
    }
    site = fetch_site_snippets(homepage) if homepage else {"values": [], "recent": []}
    values = site["values"] or base.get("values", [])
    recents = site["recent"] or base.get("recent_projects", [])
    return {
        "company_name": base.get("company_name") or company_name,
        "values": values,
        "role": "",
        "role_requirements": [],
        "recent_projects": recents,
        "language": "ko",
        "wiki_summary": base.get("wiki_summary"),
        "homepage": homepage or base.get("homepage")
    }

# ============================================
# Streamlit 기본 설정
# ============================================
st.set_page_config(page_title="회사 특화 가상 면접 코치", page_icon="🎯", layout="wide")

# ============================================
# 유틸: 안전한 시크릿 로더
# ============================================
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

# ============================================
# 공용 함수 (텍스트 처리 / 임베딩 / 검색)
# ============================================
def build_company_context(c: dict) -> str:
    base = textwrap.dedent(f"""
    [회사명] {c.get('company_name','')}
    [핵심가치] {", ".join(c.get('values', []))}
    [직무] {c.get('role','')}
    [주요 요구역량] {", ".join(c.get('role_requirements', []))}
    [최근 프로젝트] {", ".join(c.get('recent_projects', []))}
    """).strip()
    if c.get("wiki_summary"):
        base += f"\n[위키 요약] {c['wiki_summary']}"
    if c.get("homepage"):
        base += f"\n[홈페이지] {c['homepage']}"
    return base

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
            pages = []
            for i in range(len(reader.pages)):
                try: pages.append(reader.pages[i].extract_text() or "")
                except Exception: pages.append("")
            return "\n\n".join(pages)
        except Exception as e:
            st.warning(f"PDF 파싱 실패({uploaded.name}): {e}")
            return ""
    return ""

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

def embed_texts(client: OpenAI, embed_model: str, texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    resp = client.embeddings.create(model=embed_model, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

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

# ============================================
# 사이드바: 모델/키/버전 디버그
# ============================================
with st.sidebar:
    st.title("🎯 가상 면접 코치")

    API_KEY = load_api_key_from_env_or_secrets()
    if not API_KEY:
        st.info("환경변수/Secrets에서 키를 못 찾았습니다. 아래에 입력하면 즉시 사용 가능해요.")
        API_KEY = st.text_input("OPENAI_API_KEY", type="password")

    MODEL = st.selectbox("챗 모델", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
    EMBED_MODEL = st.selectbox("임베딩 모델", ["text-embedding-3-small", "text-embedding-3-large"], index=0)

    # 버전 디버그
    _openai_ver = None
    _httpx_ver = None
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

# 안전한 클라이언트 생성
try:
    client = OpenAI(api_key=API_KEY, timeout=30.0)
except TypeError:
    st.error("OpenAI 클라이언트 초기화 TypeError. requirements.txt에서 openai==1.44.0, httpx==0.27.2로 고정 후 Clear cache → Reboot 해주세요.")
    st.stop()
except Exception as e:
    st.error(f"OpenAI 클라이언트 초기화 오류: {e}")
    st.stop()

# ============================================
# 사이드바: 회사/직무 설정 + 자동 프로필 생성 + 문서 업로드(RAG)
# ============================================
with st.sidebar:
    st.markdown("---")
    st.markdown("#### 회사/직무 설정")

    data_dir = "data/companies"
    os.makedirs(data_dir, exist_ok=True)
    defaults = {
        "acme.json": {
            "company_name": "ACME",
            "values": ["고객집착", "데이터기반", "주도적 실행"],
            "role": "데이터 애널리스트",
            "role_requirements": ["SQL/EDA", "지표 설계", "리포팅/커뮤니케이션"],
            "recent_projects": ["구독 전환 퍼널 최적화", "이탈 예측 모델 파일럿"]
        },
        "contoso.json": {
            "company_name": "Contoso",
            "values": ["소유감", "협업", "고객 성공"],
            "role": "머신러닝 엔지니어",
            "role_requirements": ["모델 서빙/모니터링", "성능-비용 최적화", "A/B테스트"],
            "recent_projects": ["추천 시스템 리랭킹", "실시간 피드 랭킹"]
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

    # 🔎 자동 프로필 생성
    st.markdown("#### 🔎 자동 프로필 생성 (이름/홈페이지)")
    auto_name = st.text_input("회사 이름 (예: 네이버, Kakao, Samsung SDS)")
    auto_home = st.text_input("홈페이지 URL (선택, 예: https://www.kakaoenterprise.com)")
    auto_add_to_rag = st.checkbox("홈페이지 텍스트를 RAG에 자동 추가", value=True)
    if st.button("자동 프로필 만들기"):
        if not auto_name.strip():
            st.warning("회사 이름을 입력해 주세요.")
        else:
            with st.spinner("회사 정보 수집 중... (Wikipedia + 홈페이지 요약)"):
                auto_company = build_auto_company(auto_name.strip(), auto_home.strip() or None)
            st.session_state["company_override"] = auto_company
            st.success("자동 프로필 생성 완료! (회사 요약에 적용됨)")

            # 선택: 홈페이지 텍스트를 RAG에 넣기
            if auto_add_to_rag and auto_home.strip():
                try:
                    pages = [auto_home.strip(), auto_home.strip().rstrip("/")+"/about",
                             auto_home.strip().rstrip("/")+"/values",
                             auto_home.strip().rstrip("/")+"/mission"]
                    texts = []
                    for u in pages:
                        try:
                            r = requests.get(u, timeout=6)
                            if r.status_code == 200 and "text/html" in r.headers.get("content-type",""):
                                s = BeautifulSoup(r.text, "html.parser")
                                txts = [_clean_text(t.get_text(" ")) for t in s.find_all(["h1","h2","h3","p","li"])]
                                texts.append("\n".join([t for t in txts if len(t) > 10]))
                        except Exception:
                            pass
                    full = "\n\n".join(texts)
                    if full:
                        chs = chunk_text(full, 900, 150)
                        embs = embed_texts(client, EMBED_MODEL, chs)
                        if "rag_store" not in st.session_state:
                            st.session_state.rag_store = {"chunks": [], "embeds": None}
                        st.session_state.rag_store["chunks"] = (st.session_state.rag_store.get("chunks", []) or []) + chs
                        if st.session_state.rag_store.get("embeds") is None or st.session_state.rag_store["embeds"].size == 0:
                            st.session_state.rag_store["embeds"] = embs
                        else:
                            st.session_state.rag_store["embeds"] = np.vstack([st.session_state.rag_store["embeds"], embs])
                        st.info(f"RAG에 홈페이지 텍스트 {len(chs)}개 청크 추가")
                except Exception:
                    st.warning("홈페이지 텍스트 RAG 추가 중 오류가 있어 일부만 반영되었습니다.")

    st.markdown("#### 질문 옵션")
    q_type = st.selectbox("질문 유형", ["행동(STAR)", "기술 심층", "핵심가치 적합성", "역질문"], index=0)
    level = st.selectbox("난이도/연차", ["주니어", "미들", "시니어"], index=0)

    st.markdown("#### RAG (선택)")
    rag_enabled = st.toggle("회사 문서 기반 질문/코칭 사용", value=True)
    chunk_size = st.slider("청크 길이(문자)", 400, 2000, 900, 100)
    chunk_overlap = st.slider("오버랩(문자)", 0, 400, 150, 10)
    top_k = st.slider("검색 상위 K", 1, 8, 4, 1)
    st.caption("TXT/MD/PDF 업로드 가능 (세션 메모리 내 처리)")
    docs = st.file_uploader("회사 문서 업로드 (여러 파일 가능)", type=["txt", "md", "pdf"], accept_multiple_files=True)

# ============================================
# 회사 프로필 결정 (우선순위: 자동 생성 > 업로드 > 선택 파일)
# ============================================
if uploaded_company is not None:
    try:
        company = json.load(uploaded_company)
    except Exception as e:
        st.error(f"회사 JSON 파싱 오류: {e}")
        st.stop()
else:
    company = json.load(open(os.path.join("data/companies", company_file), "r", encoding="utf-8"))

if "company_override" in st.session_state:
    company = st.session_state["company_override"]

# ============================================
# 세션 상태
# ============================================
if "rag_store" not in st.session_state:
    st.session_state.rag_store = {"chunks": [], "embeds": None}

if "history" not in st.session_state:
    st.session_state.history = []  # [{ts, question, user_answer, score, feedback, competencies, supports}]

if "current_question" not in st.session_state:
    st.session_state.current_question = ""

# ============================================
# 업로드 문서 → RAG 인덱스 준비
# ============================================
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

# ============================================
# 검색/프롬프트
# ============================================
def retrieve_supports(query_text: str, k: int) -> List[Tuple[str, float, str]]:
    store = st.session_state.rag_store
    chunks, embeds = store.get("chunks", []), store.get("embeds", None)
    if not rag_enabled or embeds is None or len(chunks) == 0:
        return []
    qv = embed_texts(client, EMBED_MODEL, [query_text])
    scores, idxs = cosine_topk(embeds, qv, k=k)
    return [("업로드문서/홈페이지", float(s), chunks[int(i)]) for s, i in zip(scores, idxs)]

def gen_question(company: dict, qtype: str, level: str, supports: List[Tuple[str, float, str]]) -> str:
    ctx = build_company_context(company)
    rag_note = ""
    if supports:
        joined = "\n".join([f"- ({s:.3f}) {txt[:300]}" for (_, s, txt) in supports])
        rag_note = f"\n[회사 근거 문서 발췌]\n{joined}\n"
    sys = f"""너는 '{company.get('company_name','')}'의 '{company.get('role','')}' 면접관이다.
회사 맥락과 (있다면) 아래 근거 문서를 반영하여 {qtype} 유형의 **구체적 질문 1개만** 한국어로 생성하라.
지원자가 행동/지표/임팩트를 드러내도록 하고, 난이도는 {level}에 맞춘다.
서론/부연 금지. 결과는 질문 문장 한 줄만."""
    user = f"[회사 컨텍스트]\n{ctx}\n{rag_note}"
    resp = client.chat.completions.create(
        model=MODEL, temperature=0.7,
        messages=[{"role":"system","content":sys},{"role":"user","content":user}]
    )
    return resp.choices[0].message.content.strip()

def coach_answer(company: dict, question: str, user_answer: str, supports: List[Tuple[str, float, str]]) -> Dict:
    ctx = build_company_context(company)
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
추가 설명 금지. 형식 유지."""
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

# ============================================
# UI 본문
# ============================================
left, right = st.columns([1, 1])

with left:
    st.header("① 질문 생성")
    st.markdown("**회사 요약**")
    st.json(company, expanded=False)

    prompt_hint = st.text_input("질문 생성 힌트(선택)", placeholder="예: 프로젝트 관련 경험 위주로")
    if st.button("새 질문 받기", use_container_width=True):
        try:
            supports = []
            if rag_enabled and (docs or st.session_state.rag_store.get("chunks")):
                base_q = prompt_hint.strip() or f"{company.get('role','')} {', '.join(company.get('values', []))}"
                supports = retrieve_supports(base_q, top_k)
            q = gen_question(company, q_type, level, supports)
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
    st.header("② 나의 답변")
    answer = st.text_area("여기에 답변을 작성하세요 (STAR 권장: 상황-과제-행동-성과)", height=160)
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

# ============================================
# 결과 / 레이더 / 리포트
# ============================================
st.divider()
st.subheader("③ 피드백 결과")
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
st.subheader("④ 역량 레이더 (세션 누적)")
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
st.subheader("⑤ 세션 리포트 (CSV)")
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

st.caption("Tip: Cloud에서는 App → Settings → Secrets 에 OPENAI_API_KEY를 넣어 주세요. 자동 프로필은 위키/홈페이지 가용성에 따라 정확도가 달라질 수 있습니다.")
