import os
import streamlit as st
import tempfile
import re # 글자 수 카운트를 위해 추가

# LangChain 모듈 임포트
from langchain_community.document_loaders import Docx2txtLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

# 환경 변수에서 OpenAI API 키 가져오기
# LangChain_openai는 환경 변수 'OPENAI_API_KEY'를 자동으로 찾습니다.

st.set_page_config(page_title="자소서 작성 챗봇", layout="centered")
# 글자 수 (공백 제외) 카운트 함수
def count_korean_chars(text):
    """공백을 제외한 순수 한글, 영문, 숫자, 특수문자 등의 글자 수를 계산합니다."""
    if not text:
        return 0
    # 모든 공백 문자 (스페이스, 탭, 개행) 제거
    cleaned_text = re.sub(r'\s+', '', text) 
    return len(cleaned_text)

# 파일 로드 (Word) 처리 함수
@st.cache_data(show_spinner=False)
def load_file_docs(_file):
    try:
        # Word 파일을 임시로 저장
        with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp_file:
            tmp_file.write(_file.getvalue())
            tmp_file_path = tmp_file.name
            
            # Word 파일 로드
            loader = Docx2txtLoader(file_path=tmp_file_path)
            pages = loader.load()
        
        # 임시 파일 삭제
        os.unlink(tmp_file_path)
        return pages
    except Exception as e:
        st.error(f"Word 파일 '{_file.name}' 처리 중 오류 발생: {e}")
        return []

# URL 로딩 및 처리 함수
@st.cache_data(show_spinner=False)
def load_url_docs(_url):
    try:
        # WebBaseLoader를 사용하여 웹페이지 콘텐츠 로드
        loader = WebBaseLoader(_url)
        pages = loader.load()
        return pages
    except Exception as e:
        # 웹페이지 로딩 실패 시 구체적인 메시지 출력
        st.error(f"URL: {_url} 로딩 중 오류 발생. 접근할 수 없거나 콘텐츠를 추출할 수 없습니다.")
        return []

# 텍스트 청크들을 Chroma 안에 임베딩 벡터로 저장
@st.cache_resource(show_spinner="텍스트 임베딩 및 벡터 저장소 생성 중...")
def create_vector_store(_docs):
    # 단일화된 문서 분할(Text Splitting)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(_docs)
    
    st.info(f"문서가 총 {len(split_docs)}개의 청크로 분할되었습니다. 쪼개진 문서를 기반으로 벡터 저장소를 생성합니다.")
    
    # Chroma DB에 저장.
    vectorstore = Chroma.from_documents(split_docs, OpenAIEmbeddings(model='text-embedding-3-small'))
    return vectorstore

# 검색된 문서를 하나의 텍스트로 합치는 헬퍼 함수 (출처 정보 강화)
def format_docs(docs):
    formatted_content = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get('source', '업로드 파일')
        
        # 문서 출처를 명확히 표시하여 LLM이 참고할 수 있도록 함
        if source.startswith(('http://', 'https://')):
            source_display = f"Source URL: {source}"
        else:
            source_display = "Source File"

        formatted_content.append(f"**[참고 문서 {i+1}]** ({source_display})\n{doc.page_content}")
        
    # LLM이 참고할 수 있도록 출처가 포함된 문자열을 반환
    return "\n\n".join(formatted_content)

# RAG 체인 구축
@st.cache_resource(show_spinner="RAG 체인 구성 중...")
def chaining(_vectorstore):
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 5}) # 검색 문서 개수 5개로 설정

    qa_system_prompt = """
당신은 사용자가 제공한 [검색된 문서] (경력기술서, 성격, 장단점, 직무기술서, 인재상, 조직문화, 산업정보, 기업정보, 뉴스 등) 내용을 분석하여, \
사용자의 요청(자기소개서 문항)에 대한 **자기소개서 초안을 직접 작성**해주는 전문 AI 작가입니다.

## 작성 지침
1. **반드시** 제공된 [검색된 문서]의 **경험, 역량, 사실**만을 활용하여 답변을 구성합니다.
2. 답변은 사용자의 요청(질문)에 대한 자기소개서 문항의 **본문 초안** 형태로 작성되어야 합니다.
3. 문체는 **논리적이고, 전문적이며, 구체적인 서술**을 사용합니다.
4. 답변의 분량은 최소 400자 (공백 제외) 이상이 되도록 작성하되, 핵심 내용이 잘 드러나도록 합니다.
5. [검색된 문서]에 답변을 위한 정보(경험, 사례)가 충분하지 않다면, '정보가 부족하여 해당 문항의 초안을 작성할 수 없습니다. 관련 경험이나 사례를 먼저 업로드하거나 질문을 바꿔주세요.'라고 응답합니다.
6. 기업, 직무와 관련된 경험, 지식, 역량을 잘 설명하도록 작성.
7. 입사하려는 의지가 명확하게 보이도록 작성.

## 필수 출력 형식 (신규 지침)
- 사용자의 요청에 포함된 **회사 명**과 **직무 명**을 답변 최상단에 명확하게 표시해야 합니다.
- [검색된 문서]에 포함된 Source URL 중 답변 구성에 가장 핵심적으로 사용된 **하나의 URL**을 찾아서 '참고 URL'로 명시해야 합니다. (URL이 모두 파일 출처이거나 답변에 URL 출처가 사용되지 않았다면 '제시된 URL 없음'으로 명시)

[예시 출력 형식]
회사 명: [사용자가 제시한 회사 명]
직무 명: [사용자가 제시한 직무 명]
참고 URL: [가장 중요한 하나의 URL, 또는 '제시된 URL 없음']
---
[여기에 400자 이상의 자기소개서 초안 본문 작성]

\n\n[검색된 문서]\n{context}
    """

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            ("user", "{input}"),
        ]
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    
    # 1. Context retrieval chain (검색 및 문서 형식 지정)
    retrieval_chain = retriever | format_docs

    # 2. Answer generation chain (답변 생성). context와 input을 필요로 함.
    answer_chain = (
        RunnableParallel(
            context=retrieval_chain,      # 검색된 context를 프롬프트에 제공
            input=RunnablePassthrough()   # 사용자 입력(질문)을 프롬프트에 제공
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    # 3. Final output chain: 최종적으로 context와 answer를 모두 반환하는 체인 구성
    # (RunnableParallel을 사용하여 최종 출력 딕셔너리 구조 명확화)
    rag_chain = RunnableParallel(
        context=retrieval_chain, # 검색된 context 결과를 최종 출력에 포함
        answer=answer_chain      # 생성된 답변 결과를 최종 출력에 포함
    )
    return rag_chain # {"context": str, "answer": str} 반환

# Streamlit UI
st.set_page_config(page_title="자소서 작성 챗봇", layout="centered")
st.title("자소서 작성 챗봇 💬✨")
st.markdown("자소서 초안 작성에 필요한 **여러 개의 이력서 파일(Word)** 또는 **기업 정보를 담고 있는 URL**을 업로드/입력해주세요.")

# ------------------------------------
# 1. 데이터 소스 입력 섹션 (다중 처리)
# ------------------------------------
st.subheader("1. 데이터 소스 입력")

# 다중 파일 업로드
uploaded_files = st.file_uploader(
    "1-1. 이력서(Word 파일)를 여러 개 업로드하세요.", 
    type=[".docx"], 
    accept_multiple_files=True
)

st.markdown("---")

# 다중 URL 입력
url_input_area = st.text_area(
    "1-2. 기업 홈페이지 URL을 여러 줄로 입력하세요. (각 줄마다 'http://' 또는 'https://'로 시작)",
    height=150
)

# ------------------------------------
# 2. 문서 로딩 및 벡터스토어 생성
# ------------------------------------
vectorstore = None
all_docs = []

if uploaded_files or url_input_area:
    with st.spinner("모든 데이터 소스를 로드 중입니다..."):
        
        # A. 파일 로드
        if uploaded_files:
            file_progress = st.progress(0, text="Word 파일 로딩 중...")
            for i, file in enumerate(uploaded_files):
                docs_from_file = load_file_docs(file)
                all_docs.extend(docs_from_file)
                file_progress.progress((i + 1) / len(uploaded_files), text=f"Word 파일 로딩 중... ({file.name})")
            file_progress.empty()
            if uploaded_files:
                st.success(f"총 {len(uploaded_files)}개의 Word 파일 로딩 완료. 📁")

        # B. URL 로드
        if url_input_area:
            urls = [u.strip() for u in url_input_area.split('\n') if u.strip()]
            valid_urls = [u for u in urls if u.startswith(("http://", "https://"))]
            invalid_urls = [u for u in urls if not u.startswith(("http://", "https://"))]
            
            if valid_urls:
                url_progress = st.progress(0, text="URL 웹페이지 로딩 중...")
                for i, url in enumerate(valid_urls):
                    docs_from_url = load_url_docs(url)
                    all_docs.extend(docs_from_url)
                    url_progress.progress((i + 1) / len(valid_urls), text=f"URL 로딩 중... ({url})")
                url_progress.empty()
                st.success(f"총 {len(valid_urls)}개의 URL 웹페이지 로딩 완료. 🌐")

            if invalid_urls:
                st.error(f"다음 {len(invalid_urls)}개 URL은 형식이 유효하지 않아 건너뜁니다: {', '.join(invalid_urls[:3])}...") 
                st.info("URL은 'http://' 또는 'https://'로 시작해야 합니다.")

        # C. 벡터 저장소 생성
        if all_docs:
            vectorstore = create_vector_store(all_docs)
            st.success("벡터 저장소 생성 완료. 이제 챗봇을 사용할 수 있습니다.")
        else:
            st.warning("로딩된 문서 내용이 비어 있거나 처리할 수 있는 콘텐츠가 없습니다.")

# ------------------------------------
# 3. 챗봇 섹션
# ------------------------------------
st.subheader("3. 챗봇과 대화 시작")

if vectorstore:
    # vectorstore를 인수로 전달하여 체인 구성
    rag_chain = chaining(vectorstore)

    # 세션 상태 초기화 및 초기 메시지 업데이트 (사용자 안내 강화)
    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": "데이터 준비가 완료되었습니다. **회사 명, 직무 명, 그리고 자기소개서 항목**을 명확하게 포함하여 작성을 요청해 주세요.\n\n**예시 요청:**\n`회사 명: 구글 코리아, 직무 명: 소프트웨어 엔지니어, 자소서 항목: 당신이 구글에 기여할 수 있는 핵심 역량은 무엇입니까?`"}
        ]

    # 이전 대화 표시
    for msg in st.session_state.messages:
        st.chat_message(msg['role']).write(msg['content'])

    # 사용자 입력 처리
    if prompt_message := st.chat_input("질문을 입력해주세요 :)"):
        # 사용자 메시지 표시 및 저장
        st.chat_message("user").write(prompt_message)
        st.session_state.messages.append({"role": "user", "content": prompt_message})

        # LLM 응답 생성 및 표시
        with st.chat_message("assistant"):
            with st.spinner("생각 중... 잠시만 기다려 주세요."):
                try:
                    # RAG 체인 호출 및 딕셔너리 응답 수신 ({"context": str, "answer": str})
                    response_dict = rag_chain.invoke(prompt_message)
                    response_text = response_dict["answer"]
                    retrieved_context = response_dict["context"]
                except Exception as e:
                    # 오류 발생 시 디버깅 정보 출력
                    response_text = f"죄송합니다. RAG 체인 실행 중 오류가 발생했습니다: {e}"
                    retrieved_context = "검색 실패"
                    st.error(response_text)
                
                # 1. LLM 응답 출력
                st.write(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

                # 2. 글자 수 카운트 피드백
                char_count = count_korean_chars(response_text)
                
                # 시스템 프롬프트에 정의된 최소 글자 수
                MIN_CHARS = 400 
                
                st.markdown(f"**📝 글자 수:** 공백 제외 **{char_count}자** (최소 {MIN_CHARS}자 권장)")

                if char_count < MIN_CHARS:
                    st.warning(f"⚠️ **{MIN_CHARS - char_count}자 부족!** 글자 수 요구사항을 충족시키기 위해 내용을 더 구체화하거나 보강해보세요.")
                else:
                    st.success("✅ 충분한 분량입니다.")

                # 3. RAG 투명성 확보
                # 검색된 컨텍스트를 Expander를 사용하여 보여줍니다.
                with st.expander("🔍 LLM이 참고한 검색된 문서 (Retrieval Context) 보기"):
                    st.code(retrieved_context, language='markdown')
else:
    st.info("먼저 1단계에서 이력서(Word) 파일이나 기업 URL을 입력하여 데이터를 준비해주세요.")
