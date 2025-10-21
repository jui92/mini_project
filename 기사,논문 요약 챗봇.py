import os, io, time, contextlib, tempfile, hashlib
import requests
from bs4 import BeautifulSoup

import streamlit as st

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI
# (로컬 LLM 쓰면) from langchain_ollama import ChatOllama

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# --- OpenAI 키 로딩 ---
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY가 없습니다. .streamlit/secrets.toml 또는 환경변수를 설정하세요.")
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# --- LLM 선택 (둘 중 택1) ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)     # 클라우드
# llm = ChatOllama(model="solar:10.7b-instruct", temperature=0.2)  # 로컬(Ollama)

# --- 유틸: 간단 타이머 ---
@contextlib.contextmanager
def timer(store: dict, key: str):
    t0 = time.perf_counter()
    yield
    store[key] = (time.perf_counter() - t0) * 1000  # ms

st.set_page_config(page_title="뉴스/논문 요약 챗봇", page_icon="📰")
st.title("📰 뉴스/논문 요약 챗봇")

# --- 본문 추출: 뉴스 URL ---
def fetch_article_text(url: str) -> str:
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    html = BeautifulSoup(r.text, "html.parser")
    # 최소화 버전: <p>만 긁어 모으기 (언론사/블로그마다 구조가 달라 엄격 파싱은 생략)
    paragraphs = [p.get_text(" ", strip=True) for p in html.find_all("p")]
    text = "\n".join([p for p in paragraphs if len(p) > 0])
    return text

# --- 본문 추출: PDF ---
def load_pdf_text(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        path = tmp.name
    pages = PyPDFLoader(path).load_and_split()
    text = "\n".join(p.page_content for p in pages)
    return text

# --- 텍스트 분할 ---
def split_text(text: str, chunk_size=1200, chunk_overlap=150):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

# --- Map 프롬프트: 청크 요약 ---
map_prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 뉴스/논문 요약가다. 주어진 텍스트에서 핵심 사실만 한국어로 3~5문장으로 요약하라. 과장/추측 금지."),
    ("human", "{chunk}")
])

map_chain = map_prompt | llm | StrOutputParser()

# --- Reduce 프롬프트: 전체 통합 요약 ---
reduce_prompt = ChatPromptTemplate.from_messages([
    ("system", "너는 편집자다. 여러 부분 요약을 읽고 중복/군더더기를 제거해 5~7문장으로 최종 한국어 요약을 만든다. 핵심 수치/인용 유지."),
    ("human", "{partial_summaries}")
])

reduce_chain = reduce_prompt | llm | StrOutputParser()

def summarize_long_text(text: str, chunk_size=1200, chunk_overlap=150, metrics: dict | None=None) -> str:
    with timer(metrics, "split_ms"):
        chunks = split_text(text, chunk_size, chunk_overlap)
    with timer(metrics, "map_ms"):
        partials = [map_chain.invoke({"chunk": c}) for c in chunks]
    with timer(metrics, "reduce_ms"):
        final_summary = reduce_chain.invoke({"partial_summaries": "\n\n".join(partials)})
    return final_summary

# --- QA 프롬프트: 요약을 근거로 답하기 ---
qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "너는 뉴스/논문 요약 기반 Q&A 어시스턴트다. 다음 요약 내용에 근거해 한국어로 간결히 답하라. "
     "요약에 없으면 '요약에 근거가 없습니다'라고 답하라.\n\n요약:\n{summary}"),
    MessagesPlaceholder("history"),
    ("human", "{question}")
])

qa_chain = qa_prompt | llm | StrOutputParser()

# --- 간단 세션 메모리(요약 유지 + 대화 유지) ---
if "summary" not in st.session_state:
    st.session_state.summary = None
if "history" not in st.session_state:
    st.session_state.history = []   # [{"role":"user","content":...},{"role":"assistant","content":...}]

with st.sidebar:
    st.markdown("### ⏱ 성능")
    perf = {}

st.markdown("#### 데이터 입력")
col1, col2 = st.columns(2)
with col1:
    url = st.text_input("뉴스/블로그/기사 URL 붙여넣기")
with col2:
    pdf = st.file_uploader("논문/보고서 PDF 업로드", type=["pdf"])

if st.button("요약 생성", type="primary"):
    raw_text = ""
    try:
        if url:
            with st.spinner("URL에서 본문 추출 중..."):
                with timer(perf, "fetch_url_ms"):
                    raw_text = fetch_article_text(url)
        elif pdf is not None:
            with st.spinner("PDF에서 텍스트 추출 중..."):
                with timer(perf, "fetch_pdf_ms"):
                    raw_text = load_pdf_text(pdf.getvalue())
        else:
            st.warning("URL 또는 PDF 중 하나를 제공하세요.")
            st.stop()

        if len(raw_text.strip()) < 50:
            st.warning("본문을 충분히 추출하지 못했습니다. 다른 URL/PDF로 시도하세요.")
            st.stop()

        with st.spinner("요약 생성 중..."):
            summary = summarize_long_text(raw_text, metrics=perf)
            st.session_state.summary = summary
            st.success("요약 생성 완료!")

    except Exception as e:
        st.error(f"요약 생성 실패: {e}")

# 요약 미리보기
if st.session_state.summary:
    with st.expander("📌 최종 요약 보기", expanded=True):
        st.write(st.session_state.summary)

# 챗 UI
st.markdown("#### 질의응답")
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role":"assistant","content":"요약이 준비되면 자유롭게 질문해 주세요!"}]

for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

user_q = st.chat_input("요약에 대해 무엇이 궁금한가요?")
if user_q:
    if not st.session_state.summary:
        st.warning("먼저 URL 또는 PDF로 요약을 생성하세요.")
    else:
        st.session_state.messages.append({"role":"user","content":user_q})
        st.chat_message("user").write(user_q)
        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                # history를 최근 N개만 유지해도 됨
                ans = qa_chain.invoke({
                    "summary": st.session_state.summary,
                    "history": st.session_state.history[-6:],
                    "question": user_q
                })
                st.session_state.messages.append({"role":"assistant","content":ans})
                st.session_state.history.extend([
                    {"role":"user","content":user_q},
                    {"role":"assistant","content":ans},
                ])
                st.write(ans)

# 성능 지표 출력
with st.sidebar:
    if perf:
        for k,v in perf.items():
            st.write(f"- {k}: {v:,.0f} ms")