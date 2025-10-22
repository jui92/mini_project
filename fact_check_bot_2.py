import os, time, contextlib, tempfile, requests, feedparser, arxiv
from bs4 import BeautifulSoup
from newspaper import Article
import streamlit as st
from openai import OpenAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.docstore.document import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ======================
# 0) API Key 안전 로드
# ======================
def load_openai_key():
    k = os.getenv("OPENAI_API_KEY")
    if k: return k
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return None

OPENAI_API_KEY = load_openai_key()
if not OPENAI_API_KEY:
    st.error("❌ OPENAI_API_KEY가 없습니다. 환경변수 또는 .streamlit/secrets.toml에 설정하세요.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)


# ======================
# 1) 공통 유틸
# ======================
st.set_page_config(page_title="뉴스·논문 기반 팩트체크 챗봇 (GPT-4o)", page_icon="🧾", layout="wide")
st.title("🧾 뉴스·논문 기반 팩트체크 챗봇 (GPT-4o)")

@contextlib.contextmanager
def timer(store, key):
    t0 = time.perf_counter(); yield
    store[key] = (time.perf_counter() - t0) * 1000

def split_text(text, chunk_size=2000, overlap=200):
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap).split_text(text)

def chat_gpt4o(messages, temperature=0.2):
    resp = client.chat.completions.create(model="gpt-4o", messages=messages, temperature=temperature)
    return resp.choices[0].message.content


# ======================
# 2) 데이터 수집
# ======================
def fetch_article_text(url):
    # 우선 newspaper3k로 시도
    try:
        art = Article(url, language='ko'); art.download(); art.parse()
        if len(art.text.strip()) > 400: return art.text
    except Exception: pass
    # fallback
    r = requests.get(url, timeout=15); r.raise_for_status()
    html = BeautifulSoup(r.text, "html.parser")
    return "\n".join(p.get_text(" ", strip=True) for p in html.find_all("p"))

def load_pdf_text(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes); path = tmp.name
    pages = PyPDFLoader(path).load_and_split()
    return "\n".join(p.page_content for p in pages)

def fetch_news_rss(keyword, max_items=10):
    rss = f"https://news.google.com/rss/search?q={requests.utils.quote(keyword)}"
    feed = feedparser.parse(rss)
    rows=[]
    for e in feed.entries[:max_items]:
        summary = BeautifulSoup(e.summary, "html.parser").get_text(" ", strip=True)
        rows.append({"title": e.title, "summary": summary, "link": e.link})
    return rows

def fetch_arxiv(keyword, max_items=10):
    search = arxiv.Search(query=keyword, max_results=max_items, sort_by=arxiv.SortCriterion.Relevance)
    return [{"title":r.title,"summary":r.summary,"link":r.entry_id} for r in search.results()]


# ======================
# 3) 요약 (Map-Reduce)
# ======================
def summarize_long_text(text, metrics):
    chunks = split_text(text)
    partials=[]
    for ch in chunks:
        partials.append(chat_gpt4o([
            {"role":"system","content":"너는 요약가다. 핵심 사실만 3~5문장 한국어 요약. 과장·추측 금지."},
            {"role":"user","content":ch}
        ]))
    merged="\n\n".join(partials)
    final = chat_gpt4o([
        {"role":"system","content":"넌 편집자다. 중복 제거, 수치·인용 유지, 5~7문장으로 최종 한국어 요약."},
        {"role":"user","content":merged}
    ])
    return final


# ======================
# 4) 벡터스토어 (RAG)
# ======================
EMBED = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

@st.cache_resource(show_spinner=False)
def build_vectorstore(texts):
    docs=[]
    for t in texts:
        for ch in SPLITTER.split_text(t):
            docs.append(Document(page_content=ch))
    return Chroma.from_documents(docs, EMBED)

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system","너는 근거 기반 어시스턴트다. 아래 컨텍스트 범위 내에서만 한국어로 대답하라. "
              "없으면 '문서에서 근거를 찾지 못했습니다'라 말하라. 답변 끝에 [출처 N] 형태로 표시."),
    ("human","질문: {question}\n\n컨텍스트:\n{context}")
])

def join_ctx(ds):
    return "\n\n".join(f"[출처 {i+1}] {d.page_content[:1200]}" for i,d in enumerate(ds))

def rag_answer(question, retriever):
    hits = retriever.get_relevant_documents(question)
    ctx = join_ctx(hits)
    msg = RAG_PROMPT.format_messages(question=question, context=ctx)
    res = client.chat.completions.create(model="gpt-4o",
                                         messages=[m.to_dict() for m in msg],
                                         temperature=0.2)
    return res.choices[0].message.content, hits


# ======================
# 5) Streamlit UI
# ======================
with st.sidebar:
    st.markdown("### ⏱ 성능")
    perf={}

st.subheader("입력 방식 선택")
mode = st.radio("모드", ["자동 수집(추천)", "직접 입력(수동)"], horizontal=True)

if mode=="자동 수집(추천)":
    kw = st.text_input("키워드 (예: 'AI regulation', '인공지능 규제')")
    n_news = st.slider("뉴스 개수",1,20,8)
    n_paper= st.slider("논문 개수",1,20,8)

    if st.button("뉴스·논문 검색", type="primary"):
        with st.spinner("검색 중..."):
            news = fetch_news_rss(kw,n_news)
            papers= fetch_arxiv(kw,n_paper)
        st.session_state["news"]=news; st.session_state["papers"]=papers
        st.success(f"뉴스 {len(news)}개, 논문 {len(papers)}개 수집!")

    selected=[]
    if st.session_state.get("news"):
        st.write("**뉴스 선택**")
        for i,r in enumerate(st.session_state["news"]):
            if st.checkbox(f"뉴스 {i+1}. {r['title']}", key=f"news_{i}"):
                try: selected.append(fetch_article_text(r["link"]))
                except: selected.append(r["summary"])
    if st.session_state.get("papers"):
        st.write("**논문 선택**")
        for i,r in enumerate(st.session_state["papers"]):
            if st.checkbox(f"논문 {i+1}. {r['title']}", key=f"paper_{i}"):
                selected.append(r["summary"])

    if st.button("요약 생성", type="primary"):
        if not selected:
            st.warning("최소 1개 선택하세요.")
        else:
            with st.spinner("요약 중..."):
                bundle="\n\n---\n\n".join(selected)
                summary=summarize_long_text(bundle,metrics=perf)
                st.session_state.summary=summary
                st.session_state.texts=selected
                st.session_state.history=[]
                st.success("요약 완료!")

else:
    url=st.text_input("뉴스/블로그 URL")
    pdf=st.file_uploader("PDF 업로드", type=["pdf"])
    free=st.text_area("텍스트 직접 입력",height=150)
    if st.button("요약 생성", type="primary"):
        raw=""
        if url: raw=fetch_article_text(url)
        elif pdf is not None: raw=load_pdf_text(pdf.getvalue())
        elif free.strip(): raw=free
        else: st.warning("입력 필요."); st.stop()
        summary=summarize_long_text(raw,metrics=perf)
        st.session_state.summary=summary
        st.session_state.texts=[raw]
        st.session_state.history=[]
        st.success("요약 완료!")

# ---------- 요약 보기 ----------
if st.session_state.get("summary"):
    with st.expander("📌 요약 결과", expanded=True):
        st.write(st.session_state.summary)

# ---------- 질의응답 ----------
st.subheader("질문·응답")
if "messages" not in st.session_state:
    st.session_state["messages"]=[{"role":"assistant","content":"요약이 준비되면 질문해 주세요!"}]
for m in st.session_state.messages:
    st.chat_message(m["role"]).write(m["content"])

user_q=st.chat_input("요약이나 문서에 대해 질문하세요")
if user_q:
    st.session_state.messages.append({"role":"user","content":user_q})
    st.chat_message("user").write(user_q)
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            # 1) 요약 기반 1차 답
            ans = chat_gpt4o([
                {"role":"system","content":"요약 내용에 근거해 한국어로 간결히 답하라. "
                                           "없으면 '요약에 근거가 없습니다'라 말하라."},
                {"role":"system","content":f"요약:\n{st.session_state.summary}"},
                {"role":"user","content":user_q}
            ])
            # 2) 필요시 RAG 백업
            if ("근거가 없습니다" in ans) or (len(ans.strip())<10):
                if "vs" not in st.session_state:
                    st.session_state["vs"]=build_vectorstore(st.session_state.get("texts",[]))
                retriever=st.session_state["vs"].as_retriever(search_kwargs={"k":5})
                ans,_=rag_answer(user_q,retriever)
            st.session_state.messages.append({"role":"assistant","content":ans})
            st.session_state.history=st.session_state.get("history",[])+[
                {"role":"user","content":user_q},
                {"role":"assistant","content":ans}
            ]
            st.write(ans)

with st.sidebar:
    if perf:
        st.markdown("### ⏱ 처리시간(ms)")
        for k,v in perf.items():
            st.write(f"- {k}: {v:,.0f} ms")
