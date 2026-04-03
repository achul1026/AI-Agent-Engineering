from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

# ================================
# 환경 변수 로드 (.env 파일이 있을 경우)
# ================================
try:
    from dotenv import load_dotenv
    load_dotenv()  # .env 파일에 있는 API KEY 등을 환경변수로 로드
except ImportError:
    pass  # dotenv가 없어도 무시하고 진행

# ================================
# LLM 초기화
# ================================
llm = init_chat_model(model="gpt-5-mini")
# LangChain에서 제공하는 방식으로 Chat 모델 초기화
# 이후 llm.invoke()로 모델 호출

# ================================
# State 정의 (그래프 전체에서 공유되는 데이터 구조)
# ================================
class AgentState(TypedDict):
    user_message: str        # 사용자 입력 메시지
    user_id: str             # 사용자 ID
    issue_type: Optional[str]  # 'billing' 또는 'technical' 분류 결과
    step_result: Optional[str] # 각 처리 단계의 결과값
    response: Optional[str]    # 최종 사용자 응답

# ================================
# 1. 노드 정의 (각 단계별 처리 함수)
# ================================

def categorize_issue(state: AgentState) -> AgentState:
    """
    사용자 메시지를 LLM을 통해 billing / technical로 분류하는 노드
    """
    prompt = (
        f"이 지원 요청을 'billing' 또는 'technical'로 분류하세요.\n\n"
        f"메시지: {state['user_message']}"
    )
    # LLM 호출
    response = llm.invoke([HumanMessage(content=prompt)])

    # 결과 문자열 정리
    kind = response.content.strip().lower()

    # 결과 보정 로직 (LLM이 이상한 값 줄 경우 대비)
    if "billing" in kind:
        kind = "billing"
    elif "technical" in kind:
        kind = "technical"
    else:
        kind = "technical" # 기본값은 technical

    # 상태에 issue_type만 업데이트해서 반환
    return {"issue_type": kind}

def handle_invoice(state: AgentState) -> AgentState:
    """
    billing 중 '인보이스 조회' 처리
    (실제 시스템에서는 DB/API 호출 위치)
    """
    return {"step_result": f"Invoice details for {state['user_id']}"}

def handle_refund(state: AgentState) -> AgentState:
    """
    환불 처리 워크플로 시작
    """
    return {"step_result": "Refund process initiated"}

def handle_login(state: AgentState) -> AgentState:
    """
    로그인 문제 해결 (예: 비밀번호 초기화)
    """
    return {"step_result": "Password reset link sent"}

def handle_performance(state: AgentState) -> AgentState:
    """
    성능 문제 분석 처리
    """
    return {"step_result": "Performance metrics analyzed"}

def summarize_response(state: AgentState) -> AgentState:
    """
    각 단계 결과(step_result)를 기반으로
    사용자에게 보여줄 최종 응답 생성
    """
    details = state.get("step_result", "")

    prompt = f"다음 내용을 바탕으로 간결한 고객 응답을 작성하세요: {details}"

    # LLM을 통해 자연어 응답 생성
    response = llm.invoke([HumanMessage(content=prompt)])

    return {"response": response.content.strip()}

# ================================
# 2. 그래프 구성 (LangGraph 핵심 부분)
# ================================

# 상태 스키마를 기반으로 그래프 생성
graph_builder = StateGraph(AgentState)

# ----------------
# 노드 등록
# ----------------
graph_builder.add_node("categorize_issue", categorize_issue)
graph_builder.add_node("handle_invoice", handle_invoice)
graph_builder.add_node("handle_refund", handle_refund)
graph_builder.add_node("handle_login", handle_login)
graph_builder.add_node("handle_performance", handle_performance)
graph_builder.add_node("summarize_response", summarize_response)

# ----------------
# 시작 지점 → 첫 노드 연결
# ----------------
graph_builder.add_edge(START, "categorize_issue")

# ----------------
# 상위 라우터 (billing vs technical)
# ----------------
def top_router(state: AgentState):
    """
    issue_type 값에 따라 다음 경로 결정
    """
    return "billing" if state["issue_type"] == "billing" else "technical"

# categorize_issue 이후 분기
graph_builder.add_conditional_edges(
    "categorize_issue",
    top_router,
    {
        "billing": "handle_invoice",   # billing이면 인보이스 처리로 이동
        "technical": "handle_login"    # technical이면 로그인 처리로 이동
    }
)

# ----------------
# Billing 하위 분기 (invoice → refund 여부 판단)
# ----------------
def billing_router(state: AgentState):
    """
    사용자 메시지에 'refund' 포함 여부로 분기
    """
    msg = state["user_message"].lower()
    return "refund" if "refund" in msg else "invoice_end"

graph_builder.add_conditional_edges(
    "handle_invoice",
    billing_router,
    {
        "refund": "handle_refund",             # 환불 요청이면 환불 노드로
        "invoice_end": "summarize_response"    # 아니면 바로 응답 생성
    }
)

# ----------------
# Technical 하위 분기 (login → performance 여부 판단)
# ----------------
def tech_router(state: AgentState):
    """
    사용자 메시지에 'performance' 포함 여부로 분기
    """
    msg = state["user_message"].lower()
    return "performance" if "performance" in msg else "login_end"

graph_builder.add_conditional_edges(
    "handle_login",
    tech_router,
    {
        "performance": "handle_performance",   # 성능 문제면 성능 분석
        "login_end": "summarize_response"      # 아니면 바로 응답 생성
    }
)

# ----------------
# 하위 처리 후 공통 흐름 (응답 생성으로 수렴)
# ----------------
graph_builder.add_edge("handle_refund", "summarize_response")
graph_builder.add_edge("handle_performance", "summarize_response")

# ----------------
# 최종 종료 지점
# ----------------
graph_builder.add_edge("summarize_response", END)

# 그래프 컴파일 (실행 가능한 형태로 변환)
graph = graph_builder.compile()

# ================================
# 3. 그래프 실행
# ================================
if __name__ == "__main__":
    # 초기 입력 상태
    initial_state = {
        "user_message": "안녕하세요, 인보이스와 (가능하다면) 환불 관련 도움을 받고 싶습니다.",
        "user_id": "U1234"
    }

    # 그래프 실행 (전체 워크플로 자동 수행)
    result = graph.invoke(initial_state)

    # 최종 사용자 응답 출력
    print(result["response"])