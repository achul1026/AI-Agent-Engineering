# ================================
# 1. 라이브러리 import
# ================================

# @tool → 일반 Python 함수를 "LLM이 호출 가능한 도구"로 변환
from langchain_core.tools import tool

# LLM(Chat Model) 생성 함수
from langchain.chat_models import init_chat_model

# 메시지 객체들 (대화 흐름을 구성하는 핵심 데이터 구조)
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage


# ================================
# 2. 환경 변수 로드 및 검증
# ================================

import os
try:
    # .env 파일에 저장된 환경변수 로드 (개발용)
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv가 없으면 그냥 넘어감
    pass

# OpenAI API 키가 없으면 LLM 호출 자체가 불가능 → 즉시 에러 발생
if not os.getenv("OPENAI_API_KEY"):
    raise Exception(
        "OPENAI_API_KEY가 설정되지 않았습니다."
        "환경변수 또는 .env 파일에서 설정해주세요."
    )


# ================================
# 3. Tool 정의 (LLM이 사용할 기능)
# ================================

# 👉 이 함수들은 이후 "LLM이 직접 선택해서 호출"하게 됨

@tool
def multiply(x: float, y: float) -> float:
    """
    두 숫자를 곱하는 함수
    → LLM은 이 설명(docstring)을 보고 어떤 tool을 쓸지 판단함
    """
    return x * y


@tool
def exponentiate(x: float, y: float) -> float:
    """
    x의 y제곱을 계산
    """
    return x ** y


@tool
def add(x: float, y: float) -> float:
    """
    두 숫자를 더하는 함수
    """
    return x + y


# 👉 LLM에게 제공할 Tool 목록
tools = [multiply, exponentiate, add]


# ================================
# 4. LLM 생성 + Tool 연결
# ================================

# ChatGPT 모델 생성
# temperature=0 → 항상 같은 입력이면 같은 결과 (결정적)
llm = init_chat_model(model="gpt-5-mini", temperature=0)

# 🔥 핵심
# LLM에게 "이 함수들을 필요하면 써도 된다"라고 알려주는 단계
llm_with_tools = llm.bind_tools(tools)


# ================================
# 5. 사용자 입력 → 메시지 생성
# ================================

query = "393 * 12.25는 얼마인가요? 그리고 11 + 49는요?"

# LangChain은 모든 대화를 메시지 리스트로 관리
messages = [
    HumanMessage(query)  # 사용자 입력
]


# ================================
# 6. 1차 LLM 호출 (Tool 사용할지 판단)
# ================================

# 👉 여기서 실제로 OpenAI API 호출 발생
ai_msg = llm_with_tools.invoke(messages)

# 결과:
# - 단순 답변일 수도 있고
# - tool_calls가 포함된 "함수 호출 요청"일 수도 있음

messages.append(ai_msg)


# ================================
# 7. LLM이 요청한 Tool 실행
# ================================

# ai_msg.tool_calls 안에는 이런 구조가 들어있음:
# [
#   {"name": "multiply", "args": {"x": 393, "y": 12.25}, "id": "..."},
#   {"name": "add", "args": {"x": 11, "y": 49}, "id": "..."}
# ]

for tool_call in ai_msg.tool_calls:

    # 👉 LLM이 요청한 tool 이름으로 실제 Python 함수 선택
    selected_tool = {
        "add": add,
        "multiply": multiply,
        "exponentiate": exponentiate,
    }[tool_call["name"]]

    # 👉 실제 Python 함수 실행 (여기가 "진짜 계산 수행 위치")
    result = selected_tool.invoke(tool_call['args'])

    # 디버깅 출력 (어떤 tool이 어떻게 실행됐는지 확인)
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")
    print(f"Result: {result}")
    print()

    # 👉 실행 결과를 다시 LLM에게 전달하기 위한 메시지 생성
    tool_msg = ToolMessage(
        content=str(result),              # 실행 결과 (문자열로 전달)
        tool_call_id=tool_call["id"]     # 어떤 호출에 대한 응답인지 연결
    )

    # 👉 이 메시지가 중요:
    # "너가 요청한 계산 결과 여기 있음" 이라고 LLM에게 알려줌
    messages.append(tool_msg)


# ================================
# 8. 최종 LLM 호출 (결과 종합)
# ================================

# 👉 다시 OpenAI API 호출
# 이번에는 tool 결과까지 포함된 상태
final_response = llm_with_tools.invoke(messages)

# 👉 LLM이 자연어로 최종 답변 생성
print(final_response.content)