from typing import Annotated
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START

# 환경변수 확인
import os
try:
    from dotenv import load_dotenv
    load_dotenv()  # .env 파일에 정의된 환경변수 로드 (예: OPENAI_API_KEY)
except ImportError:
    pass  # dotenv가 없으면 무시하고 진행

# OPENAI_API_KEY 존재 여부 확인 (없으면 실행 중단)
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY가 설정되지 않았습니다."
        "환경변수 또는 .env 파일에서 설정해주세요."
    )

# LLM 초기화
# gpt-5-mini 모델을 사용하고, temperature=0으로 설정하여 일관된 응답 생성
llm = init_chat_model(model="gpt-5-mini", temperature=0)


def call_model(state: MessagesState):
    # 현재 상태(state)에 포함된 메시지 리스트를 LLM에 전달하여 응답 생성
    response = llm.invoke(state["messages"])

    # LangGraph에서는 상태를 딕셔너리 형태로 반환하며,
    # "messages" 키에 새로운 메시지를 추가하여 다음 단계로 전달
    return {"messages": response}


# StateGraph 생성 (MessagesState를 상태 구조로 사용)
# MessagesState는 기본적으로 "messages" 리스트를 포함하는 상태 타입
builder = StateGraph(MessagesState)

# "call_model"이라는 이름으로 노드 등록 (LLM 호출 노드)
builder.add_node("call_model", call_model)

# 그래프 시작 지점(START)에서 call_model 노드로 흐름 연결
builder.add_edge(START, "call_model")

# 그래프를 실행 가능한 형태로 컴파일
graph = builder.compile()


from langgraph.checkpoint.memory import MemorySaver

# MemorySaver 생성
# → 대화 상태를 저장(체크포인트)하여 이후 요청에서도 이어서 사용할 수 있게 함
# → 즉, "메모리" 역할 수행
memory = MemorySaver()

# checkpointer=memory 옵션을 통해 그래프에 메모리 기능 추가
# → 같은 thread_id를 사용하면 이전 대화가 유지됨
graph = builder.compile(checkpointer=memory)

# 실행 설정
# thread_id는 대화 세션을 구분하는 ID
# 같은 thread_id를 사용하면 동일한 메모리를 공유
config = {"configurable": {"thread_id": "1"}}

# 첫 번째 사용자 입력 (자기 이름을 알려줌)
input_message = {"type": "user", "content": "안녕하세요! 제 이름은 경철입니다."}

# graph.stream:
# → 그래프 실행을 스트리밍 방식으로 수행
# → 각 단계의 결과(chunk)를 순차적으로 받아옴
for chunk in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    # 마지막 메시지를 사람이 읽기 좋은 형태로 출력
    chunk["messages"][-1].pretty_print()


# 두 번째 사용자 입력 (이전 대화 기억 확인)
input_message = {"type": "user", "content": "제 이름이 뭐라고 했죠?"}

# 같은 thread_id를 사용하므로 이전 대화(메모리)가 유지됨
for chunk in graph.stream({"messages": [input_message]}, config, stream_mode="values"):
    # LLM이 이전 메시지를 참고하여 이름을 기억하는지 확인 가능
    chunk["messages"][-1].pretty_print()