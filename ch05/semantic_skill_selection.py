# ================================
# 기본 라이브러리 import
# ================================
import os
import requests
import logging

# ================================
# LangChain 관련 모듈 import
# ================================
from langchain_core.tools import tool                  # 함수를 Tool로 등록하기 위한 데코레이터
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # LLM 및 임베딩 모델
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage  # 메시지 객체
from langchain_community.vectorstores import FAISS     # 벡터스토어 래퍼
import faiss                                           # 실제 벡터 검색 라이브러리
import numpy as np                                     # 수치 계산 및 벡터 처리

# ================================
# 환경변수 로드 (.env 지원)
# ================================
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ================================
# OpenAI API 키 존재 여부 확인
# ================================
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY가 설정되지 않았습니다."
        "환경변수 또는 .env 파일에서 설정해주세요."
    )

# ================================
# Wolfram Alpha Tool 정의
# ================================
@tool
def query_wolfram_alpha(expression: str) -> str:
    """
    Wolfram Alpha에 질의를 보내 식을 계산하거나 정보를 조회합니다.
    Args: expression (str): 계산하거나 평가할 수식 또는 질의입니다.
    Returns: str: 계산 결과 또는 조회된 정보입니다.
        """

    # API 호출 URL 구성 (수식 URL 인코딩)
    api_url = f'''https://api.wolframalpha.com/v1/result?i={requests.utils.quote(expression)}&appid={os.getenv("WOLFRAM_ALPHA_APP_ID")}'''

    try:
        # 외부 API 호출
        response = requests.get(api_url)

        # 정상 응답 처리
        if response.status_code == 200:
            return response.text
        else:
            # API 오류 처리
            raise ValueError(f'''Wolfram Alpha API 오류: 
            {response.status_code} - {response.text}''')

    except requests.exceptions.RequestException as e:
        # 네트워크 오류 처리
        raise ValueError(f"Wolfram Alpha 질의에 실패했습니다: {e}")


# ================================
# OpenAI 임베딩 및 LLM 초기화
# ================================
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))  # 텍스트 → 벡터 변환
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))                      # LLM (파라미터 생성용)

# ================================
# 도구 설명 정의 (Semantic Search 대상)
# ================================
tool_descriptions = {
    "query_wolfram_alpha": "Wolfram Alpha에 질의를 보내 식을 계산하거나 정보를 조회합니다.",
    "trigger_zapier_webhook": "미리 정의된 Zap을 실행하기 위해 Zapier 웹훅을 트리거합니다.",
    "send_slack_message": "지정한 Slack 채널에 메시지를 보냅니다."
}

# ================================
# 각 도구 설명 → 임베딩 벡터 생성
# ================================
tool_embeddings = []
tool_names = []

for tool_name, description in tool_descriptions.items():
    # 텍스트 설명을 벡터로 변환
    embedding = embeddings.embed_query(description)
    tool_embeddings.append(embedding)
    tool_names.append(tool_name)

# ================================
# FAISS 벡터 인덱스 생성
# ================================
dimension = len(tool_embeddings[0])  # 벡터 차원 수
index = faiss.IndexFlatL2(dimension) # L2 거리 기반 인덱스

# ================================
# 코사인 유사도 사용을 위한 정규화
# ================================
# (FAISS는 기본적으로 L2 거리 기반이므로, 정규화하면 코사인 유사도처럼 동작)
faiss.normalize_L2(np.array(tool_embeddings).astype('float32'))

# ================================
# numpy 배열로 변환 후 인덱스에 추가
# ================================
tool_embeddings_np = np.array(tool_embeddings).astype('float32')
index.add(tool_embeddings_np)

# ================================
# FAISS 인덱스 → 도구 이름 매핑
# ================================
index_to_tool = {
    0: "query_wolfram_alpha",
    1: "trigger_zapier_webhook",
    2: "send_slack_message"
}

# ================================
# 1. 도구 선택 (Semantic Search 핵심)
# ================================
def select_tool(query: str, top_k: int = 1) -> list:
    """
    벡터 기반 검색을 사용하여 사용자 질의에 가장 적합한 도구(들)를 선택합니다.

    Args:
        query (str): 사용자의 입력 질의.
        top_k (int): 검색할 상위 도구의 수.

    Returns:
        list: 선택된 도구 함수 이름의 리스트.
    """

    # 사용자 질의를 임베딩
    query_embedding = np.array(embeddings.embed_query(query)).astype('float32')

    # 코사인 유사도 계산을 위한 정규화
    faiss.normalize_L2(query_embedding.reshape(1, -1))

    # 유사한 벡터 검색
    D, I = index.search(query_embedding.reshape(1, -1), top_k)

    # 인덱스를 실제 도구 이름으로 변환
    selected_tools = [index_to_tool[idx] for idx in I[0] if idx in index_to_tool]

    return selected_tools

# ================================
# 2. 파라미터 생성 (LLM 활용)
# ================================
def determine_parameters(query: str, tool_name: str) -> dict:
    """
    LLM을 사용하여 질의를 분석하고 호출할 도구의 파라미터를 결정합니다.
    """

    # LLM에게 "이 도구를 쓰려면 어떤 파라미터가 필요해?"라고 질문
    messages = [
        HumanMessage(content=f"사용자의 질의: '{query}'를 바탕으로, 도구 '{tool_name}'에 사용할 파라미터는 무엇입니까?")
    ]

    # LLM 호출
    response = llm.invoke(messages)

    # 실제로는 JSON 파싱 필요 (여기서는 단순 처리)
    parameters = {}

    if tool_name == "query_wolfram_alpha":
        parameters["expression"] = query  # 전체 쿼리를 그대로 사용
    elif tool_name == "trigger_zapier_webhook":
        parameters["zap_id"] = "123456"
        parameters["payload"] = {"data": query}
    elif tool_name == "send_slack_message":
        parameters["channel"] = "#general"
        parameters["message"] = query

    return parameters

# ================================
# 3. 실행 흐름 (End-to-End)
# ================================

# 예제 사용자 질의
user_query = "2x + 3 = 7"

# 1) 도구 선택 (Semantic 기반)
selected_tools = select_tool(user_query, top_k=1)
tool_name = selected_tools[0] if selected_tools else None

if tool_name:
    # 2) 파라미터 생성
    args = determine_parameters(user_query, tool_name)

    # 3) 도구 실행
    try:
        # globals()에서 해당 도구 함수 찾기
        if tool_name in globals():
            tool_result = globals()[tool_name].invoke(args)
            print(f"도구 '{tool_name}' 결과: {tool_result}")
        else:
            print(f"도구 '{tool_name}'가 정의되지 않았습니다. (실제 실행을 위해서는 도구 함수 구현이 필요합니다)")
            print(f"선택된 도구: {tool_name}")
            print(f"파라미터: {args}")

    except ValueError as e:
        print(f"도구 '{tool_name}' 호출 중 오류 발생: {e}")
else:
    print("선택된 도구가 없습니다.")