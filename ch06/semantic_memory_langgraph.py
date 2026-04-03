# 타입 힌트를 위한 모듈 (LangGraph에서 상태 정의할 때 사용)
from typing_extensions import TypedDict

# LLM 초기화 유틸 (OpenAI 등 다양한 모델을 동일 방식으로 사용 가능)
from langchain.chat_models import init_chat_model

# 시스템 메시지 (LLM에게 역할/지침을 주는 메시지)
from langchain_core.messages import SystemMessage

# LangGraph 핵심 구성 요소
from langgraph.graph import StateGraph, MessagesState, START

# -----------------------------
# ✅ 벡터DB 관련 (RAG 핵심)
# -----------------------------
# FAISS: 벡터 유사도 검색 엔진
from langchain_community.vectorstores import FAISS

# OpenAI 임베딩 모델 (텍스트 → 벡터 변환)
from langchain_openai import OpenAIEmbeddings


# -----------------------------
# ✅ 환경변수 로드 (.env 파일 지원)
# -----------------------------
import os
try:
    from dotenv import load_dotenv
    load_dotenv()  # .env 파일에 있는 환경변수 로드
except ImportError:
    pass

# OpenAI API Key 없으면 실행 중단
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY가 설정되지 않았습니다."
        "환경변수 또는 .env 파일에서 설정해주세요."
    )


# -----------------------------
# ✅ LLM 초기화
# -----------------------------
# gpt-5-mini 모델 사용 (속도 빠르고 비용 저렴)
llm = init_chat_model(model="gpt-5-mini", temperature=0)
# temperature=0 → 항상 일관된 답변 (결정적 출력)


# -----------------------------
# ✅ 데이터 준비 (지식 저장)
# -----------------------------
# 머신러닝 설명 텍스트
text = """
머신러닝은 분석 모델 구축을 자동화하는 데이터 분석 방법이다.
이는 시스템이 데이터로부터 학습하고 패턴을 식별하며 최소한의 인간 개입으로 의사결정을 내릴 수 있다는 개념에 기반한 인공지능의 한 분야이다.
머신러닝 알고리즘은 원하는 출력 사례를 포함한 데이터셋으로 학습된다.
예를 들어, 이미지를 분류하는 머신러닝 알고리즘은 고양이와 개의 이미지를 포함한 데이터셋으로 훈련될 수 있다.
알고리즘이 학습을 마치면 새로운 데이터에 대한 예측에 사용될 수 있다.
"""

# 메타데이터 (출처 정보 등)
metadata = {
    "title": "Introduction to Machine Learning",
    "url": "https://learn.microsoft.com/en-us/training/modules/introduction-to-machine-learning"
}

# 인공지능 설명 텍스트
text2 = """
인공지능(AI)은 인간처럼 사고하고 행동을 모방하도록 프로그래밍된 기계에서 인간 지능을 시뮬레이션하는 것을 의미한다.
이 용어는 학습과 문제 해결과 같이 인간의 정신과 연관된 특성을 보이는 모든 기계에 적용될 수 있다.
AI 연구는 게임 플레이부터 의료 진단에 이르기까지 매우 다양한 문제를 해결하기 위한 효과적인 기법을 개발하는 데 큰 성공을 거두었다.
"""

metadata2 = {
    "title": "Introduction to Artificial Intelligence",
    "url": "https://microsoft.github.io/AI-For-Beginners/"
}


# -----------------------------
# ✅ 벡터DB 생성 (핵심 RAG 단계)
# -----------------------------
# 텍스트를 벡터로 변환하는 모델
embeddings = OpenAIEmbeddings()

# 저장할 텍스트 리스트
texts = [text, text2]

# 각 텍스트에 대한 메타데이터
metadatas = [metadata, metadata2]

# FAISS 벡터DB 생성
# 내부적으로:
# 1. 텍스트 → 임베딩 벡터 변환
# 2. 벡터를 FAISS 인덱스에 저장
vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)


# -----------------------------
# ✅ 사용자 질문 (검색 쿼리)
# -----------------------------
query = "AI와 머신러닝은 어떤 관계가 있나요?"


# -----------------------------
# ✅ LangGraph 노드 정의
# -----------------------------
def call_model(state: MessagesState):
    """
    LangGraph에서 실행되는 핵심 노드 함수

    역할:
    1. 벡터DB에서 관련 문서 검색
    2. 검색 결과를 context로 구성
    3. LLM에 전달하여 답변 생성
    """

    # 🔍 1. 유사도 검색
    # query와 가장 유사한 문서 3개 찾기
    docs = vectorstore.similarity_search(query, k=3)

    # 📚 2. context 생성
    # 검색된 문서 내용을 하나의 문자열로 합침
    context = "\n\n".join([doc.page_content for doc in docs])

    # 🧠 3. LLM 입력 구성
    # SystemMessage로 "참고자료"를 먼저 주고
    # 그 뒤에 사용자 질문 추가
    messages = [SystemMessage(content="참고:\n" + context)] + list(state["messages"])

    # 🤖 4. LLM 호출
    response = llm.invoke(messages)

    # LangGraph는 dict 형태로 상태 반환해야 함
    return {"messages": response}


# -----------------------------
# ✅ LangGraph 구성
# -----------------------------
# 상태 기반 그래프 생성 (메시지 기반 상태 사용)
builder = StateGraph(MessagesState)

# 노드 추가
builder.add_node("call_model", call_model)

# 시작 → call_model로 연결
builder.add_edge(START, "call_model")

# 그래프 컴파일 (실행 가능한 상태로 변환)
graph = builder.compile()


# -----------------------------
# ✅ 실행
# -----------------------------
# 사용자 입력 메시지
input_message = {
    "type": "user",
    "content": "AI와 머신러닝은 어떤 관계가 있나요?"
}

# 그래프 실행 (stream 방식)
for chunk in graph.stream({"messages": [input_message]}, {}, stream_mode="values"):
    # 마지막 메시지 출력
    chunk["messages"][-1].pretty_print()