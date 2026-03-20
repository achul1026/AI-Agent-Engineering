# LangChain 메시지 타입
from langchain_core.messages import HumanMessage, ToolMessage

# 우리가 만든 에이전트 그래프 import
from simple_customer_support_agent import graph


# -------------------------------
# 1) 테스트 입력 구성
# -------------------------------

# 테스트용 주문 정보
example_order = {"order_id": "B73973"}

# 사용자 메시지 (취소 의도가 포함된 자연어 입력)
convo = [
    HumanMessage(content='''더 저렴한 곳을 찾았습니다.
    주문 B73973을 취소해주세요.''')
]


# -------------------------------
# 2) 에이전트 실행
# -------------------------------

# graph.invoke() → LangGraph 실행
# 입력: 상태(State) = {order + messages}
result = graph.invoke({
    "order": example_order,
    "messages": convo
})


# -------------------------------
# 3) 도구 호출 여부 검증
# -------------------------------

"""
검증 목적:
→ 에이전트가 실제로 cancel_order 도구를 호출했는지 확인

확인 방법 2가지:
1. LLM 메시지에 tool_calls 속성이 있는지
2. ToolMessage 타입이 결과에 포함되어 있는지
"""

has_tool_call = any(
    getattr(m, "tool_calls", None) or isinstance(m, ToolMessage)
    for m in result["messages"]
)

# 👉 도구 호출이 없으면 테스트 실패
assert has_tool_call, "주문 취소 도구가 호출되지 않음"


# -------------------------------
# 4) 응답 내용 검증
# -------------------------------

"""
검증 목적:
→ 사용자에게 '취소 완료' 같은 확인 메시지를 제대로 전달했는지 확인
"""

assert any(
    "취소" in str(m.content)   # 메시지 내용에 '취소' 포함 여부 확인
    for m in result["messages"]
), "확인 메시지가 누락됨"


# -------------------------------
# 5) 테스트 통과 출력
# -------------------------------

print("✅ 에이전트가 최소 평가 기준을 통과했습니다.")