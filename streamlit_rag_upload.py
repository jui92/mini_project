import os, hashlib, tempfile
import streamlit as st

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# (선택) Chroma 캐시 초기화가 꼭 필요할 때만 사용
# import chromadb
# chromadb.api.client.SharedSystemClient.clear_system_cache()

# sqlite 대체 (필요한 경우에만)
import sys
__import__("pysqlite3"); sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

# --- OpenAI 키 안전 설정 ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY가 설정되지 않았습니다. 환경변수 또는 .streamlit/secrets.toml을 확인하세요.")
    st.stop()
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 파일 바이트 해시 (캐시/컬렉션 분리에 사용)
def file_hash(uploaded_file) -> str:
    m = hashlib.sha256()
    m.update(uploaded_file.getvalue())
    return m.hexdigest()[:16]

@st.cache_data(show_spinner=False)
def load_pdf_bytes(_bytes: bytes):
    with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp:
        tmp.write(_bytes)
        path = tmp.name
    loader = PyPDFLoader(file_path=path)
    return loader.load_and_split()

def make_vectorstore_key(fhash: str, embed_model: str) -> str:
    return f"chroma_{embed_model.replace('-', '_')}_{fhash}"

# persist를 쓰고 싶지 않다면 persist_directory=None (메모리)로 두세요.
def create_vector_store(pages, collection_name: str):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = splitter.split_documents(pages)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # 1536차원
    vs = Chroma.from_documents(
        docs, embeddings,
        collection_name=collection_name,
        persist_directory=None  # 디스크 보존 원하면 "./.chroma/<collection_name>" 등으로
    )
    return vs

@st.cache_resource(show_spinner=False)
def chaining(_pages, collection_name: str):
    vectorstore = create_vector_store(_pages, collection_name)
    retriever = vectorstore.as_retriever()

    qa_system_prompt = (
        "당신은 문서 기반 QA 어시스턴트입니다. 아래 컨텍스트만 사용해 한국어로 공손하고 간결하게 답하세요. "
        "모르면 모른다고 말하세요. 이모지는 딱 1개만 사용하세요.\n\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        ("human", "{input}")
    ])

    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
    rag_chain = (
        {"context": retriever | (lambda d: "\n\n".join(doc.page_content for doc in d)),
         "input": RunnablePassthrough()}
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# ---- UI ----
st.header("ChatPDF 💬 📚")
uploaded = st.file_uploader("PDF를 업로드하세요", type=["pdf"])

if uploaded is not None:
    fhash = file_hash(uploaded)
    pages = load_pdf_bytes(uploaded.getvalue())
    coll = make_vectorstore_key(fhash, "text-embedding-3-small")
    rag_chain = chaining(pages, coll)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "무엇이든 물어보세요!"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input("질문을 입력해주세요 :)"):
        st.chat_message("user").write(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_chain.invoke(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.write(response)
