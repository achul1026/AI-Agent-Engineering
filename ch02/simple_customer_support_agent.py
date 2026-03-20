from typing import TypedDict, Annotated, Sequence
import operator

# LangChain에서 Tool 정의할 때 사용
from langchain.tools import tool

# LLM 초기화 (OpenAI 모델 사용)
from langchain.chat_models import init_chat_model

# 메시지 타입 (LLM과 주고받는 데이터 구조)
from langchain_core.messages import (
    BaseMessage, SystemMessage, HumanMessage, ToolMessage
)

# LangGraph: 상태 기반 워크플로우 구성
from langgraph.graph import StateGraph


# -------------------------------
# 0) 환경변수 로드
# -------------------------------
import os
try:
    from dotenv import load_dotenv
    load_dotenv()  # 👉 .env 파일 읽어서 환경변수로 등록
except ImportError:
    pass

# 👉 OpenAI API 키 체크
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY가 설정되지 않았습니다."
    )


# -------------------------------
# 1) 상태(State) 정의
# -------------------------------
class AgentState(TypedDict):
    """
    LangGraph에서 사용하는 '공유 데이터 구조'

    order: 주문 정보 (ex. 주문 ID)
    messages: 대화 히스토리

    Annotated[..., operator.add]:
    → 새로운 메시지가 들어올 때 기존 메시지에 이어붙임 (누적)
    """
    order: dict
    messages: Annotated[Sequence[BaseMessage], operator.add]


# -------------------------------
# 2) Tool 정의 (LLM이 호출하는 함수)
# -------------------------------
@tool
def cancel_order(order_id: str) -> str:
    """
    주문 취소 기능

    실제 서비스라면:
    - DB 조회
    - 배송 상태 확인
    - 취소 API 호출
    """
    return f"주문 {order_id}이(가) 취소되었습니다."


# -------------------------------
# 3) 핵심 에이전트 로직
# -------------------------------
def call_model(state):
    """
    에이전트의 핵심 실행 함수

    흐름:
    1. LLM에게 질문 → Tool 쓸지 판단
    2. Tool 실행 (필요 시)
    3. 다시 LLM → 최종 자연어 응답 생성
    """

    # 현재까지의 대화
    msgs = state["messages"]

    # 주문 정보 가져오기
    order = state.get("order", {"order_id": "UNKNOWN"})


    # -------------------------------
    # LLM 생성
    # -------------------------------
    llm = init_chat_model(
        model="gpt-5-mini",
        temperature=0  # 👉 결과를 안정적으로 (결정적)
    )

    # 👉 Tool을 사용할 수 있도록 LLM에 연결
    llm_with_tools = llm.bind_tools([cancel_order])


    # -------------------------------
    # 시스템 프롬프트 (역할 정의)
    # -------------------------------
    prompt = (
        f'''당신은 이커머스 지원 에이전트입니다.
        주문ID: {order["order_id"]}

        고객이 취소를 요청하면
        cancel_order(order_id)를 호출하고

        간단한 확인 메시지를 보내세요.

        그렇지 않으면 일반적으로 응답하세요.'''
    )

    # 👉 System + User 메시지 합치기
    full = [SystemMessage(content=prompt)] + msgs


    # -------------------------------
    # 1차 LLM 호출 (Tool 사용할지 판단)
    # -------------------------------
    first = llm_with_tools.invoke(full)

    # 결과 저장
    out = [first]


    # -------------------------------
    # Tool 호출 여부 확인
    # -------------------------------
    if getattr(first, "tool_calls", None):

        # 👉 LLM이 요청한 tool 정보
        tc = first.tool_calls[0]

        # 👉 실제 Python 함수 실행
        result = cancel_order.invoke(tc["args"])

        # 👉 Tool 실행 결과를 메시지로 추가
        out.append(
            ToolMessage(
                content=result,
                tool_call_id=tc["id"]
            )
        )


        # -------------------------------
        # 2차 LLM 호출 (최종 답변 생성)
        # -------------------------------
        second = llm.invoke(full + out)

        out.append(second)


    # 👉 LangGraph는 반드시 dict 형태로 반환
    return {"messages": out}


# -------------------------------
# 4) Graph 구성 (흐름 정의)
# -------------------------------
def construct_graph():
    """
    LangGraph로 실행 흐름 정의

    지금 구조:
    [START] → call_model → [END]
    """

    g = StateGraph(AgentState)

    # 노드 추가 (하나의 실행 단계)
    g.add_node("assistent", call_model)

    # 시작 노드 지정
    g.set_entry_point("assistent")

    # 실행 가능한 형태로 변환
    return g.compile()


# Graph 생성
graph = construct_graph()


# -------------------------------
# 5) 실행부 (테스트)
# -------------------------------
if __name__ == "__main__":

    # 테스트용 주문 데이터
    example_order = {"order_id": "B73973"}

    # 사용자 입력
    convo = [
        HumanMessage(content="주문 #B73973를 취소해주세요.")
    ]

    # 👉 Graph 실행
    result = graph.invoke({
        "order": example_order,
        "messages": convo
    })

    # 결과 출력
    for msg in result["messages"]:
        print(f"{msg.type}: {msg.content}")