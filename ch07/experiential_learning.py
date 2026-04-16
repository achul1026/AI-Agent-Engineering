from typing import Annotated  # 타입 힌트를 위한 Annotated (현재 코드에서는 직접 사용되지는 않음)
from typing_extensions import TypedDict  # TypedDict 타입 정의용 (현재 코드에서는 미사용)
from langchain.chat_models import init_chat_model  # LLM 초기화 함수
from langgraph.graph import StateGraph, MessagesState, START  # LangGraph 상태 기반 실행 구조
from langchain_core.messages import HumanMessage  # 사용자 메시지 객체

# 환경변수 확인
import os
try:
    from dotenv import load_dotenv
    load_dotenv()  # .env 파일을 로드하여 API KEY 등을 환경변수로 등록
except ImportError:
    pass  # dotenv가 없어도 무시 (이미 환경변수 설정되어 있을 수 있음)

# OpenAI API KEY 존재 여부 체크
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY가 설정되지 않았습니다."
        "환경변수 또는 .env 파일에서 설정해주세요."
    )

# LLM 초기화 (gpt-5-mini 모델, temperature=0 → 항상 일관된 출력)
llm = init_chat_model(model="gpt-5-mini", temperature=0)

# LLM 호출 함수 (LangGraph에서 사용하는 노드 함수)
def call_model(state: MessagesState):
    # state["messages"]에 있는 메시지를 그대로 LLM에 전달
    response = llm.invoke(state["messages"])
    # LangGraph는 반드시 dict 형태로 반환해야 함
    return {"messages": response}

class InsightAgent:
    def __init__(self):
        # 생성된 인사이트 (기본 저장소)
        self.insights = []
        # KPI 달성 등으로 중요도가 높은 인사이트
        self.promoted_insights = []
        # KPI 미달 등으로 중요도가 낮은 인사이트
        self.demoted_insights = []
        # 성찰(Reflection) 결과 저장
        self.reflections = []

    def generate_insight(self, observation):
        # 관찰 데이터를 기반으로 LLM을 사용해 인사이트 생성

        # LLM에 전달할 메시지 구성
        messages = [HumanMessage(content=f"다음 관찰을 바탕으로 인사이트를 생성하세요: '{observation}'")]

        # LangGraph 상태 그래프 생성
        builder = StateGraph(MessagesState)

        # "generate_insight"라는 노드 추가 (LLM 호출)
        builder.add_node("generate_insight", call_model)

        # 시작 지점 → generate_insight 노드로 연결
        builder.add_edge(START, "generate_insight")

        # 그래프 컴파일 (실행 가능한 상태로 변환)
        graph = builder.compile()

        # 메시지를 전달하여 그래프 실행
        result = graph.invoke({"messages": messages})

        # 마지막 메시지(AI 응답)에서 생성된 인사이트 추출
        generated_insight = result["messages"][-1].content

        # 인사이트 저장
        self.insights.append(generated_insight)

        # 결과 출력
        print(f"생성된 인사이트: {generated_insight}")
        return generated_insight

    def promote_insight(self, insight):
        # 인사이트를 '중요 인사이트'로 승격
        if insight in self.insights:
            self.insights.remove(insight)  # 기존 리스트에서 제거
            self.promoted_insights.append(insight)  # 승격 리스트에 추가
            print(f"승격된 인사이트: {insight}")
        else:
            print(f"'{insight}'인사이트를 찾을 수 없습니다.")

    def demote_insight(self, insight):
        # 인사이트를 '비중 낮은 인사이트'로 강등
        if insight in self.promoted_insights:
            self.promoted_insights.remove(insight)  # 승격 리스트에서 제거
            self.demoted_insights.append(insight)  # 강등 리스트에 추가
            print(f"강등된 인사이트: {insight}")
        else:
            print(f"'{insight}'인사이트를 찾을 수 없습니다.")

    def edit_insight(self, old_insight, new_insight):
        # 인사이트를 수정하는 함수 (모든 상태 리스트를 탐색)

        if old_insight in self.insights:
            index = self.insights.index(old_insight)
            self.insights[index] = new_insight
        elif old_insight in self.promoted_insights:
            index = self.promoted_insights.index(old_insight)
            self.promoted_insights[index] = new_insight
        elif old_insight in self.demoted_insights:
            index = self.demoted_insights.index(old_insight)
            self.demoted_insights[index] = new_insight
        else:
            print(f"'{old_insight}'인사이트를 찾을 수 없습니다.")
            return

        print(f"수정된 인사이트: '{old_insight}' -> '{new_insight}'")

    def show_insights(self):
        # 현재 인사이트 상태 출력
        print("\n현재 인사이트:")
        print(f"인사이트: {self.insights}")
        print(f"승격된 인사이트: {self.promoted_insights}")
        print(f"강등된 인사이트: {self.demoted_insights}")

    def reflect(self, reflexion_prompt):
        # 성찰(Reflection) 수행

        # LangGraph 상태 그래프 생성
        builder = StateGraph(MessagesState)

        # "reflection" 노드 추가 (LLM 호출)
        builder.add_node("reflection", call_model)

        # 시작 → reflection 노드 연결
        builder.add_edge(START, "reflection")

        # 그래프 컴파일
        graph = builder.compile()

        # 성찰 프롬프트를 전달하여 실행
        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=reflexion_prompt  # 성찰을 위한 입력 프롬프트
                    )
                ]
            }
        )

        # LLM이 생성한 성찰 결과 추출
        reflection = result["messages"][-1].content

        # 성찰 결과 저장
        self.reflections.append(reflection)

        # 출력
        print(f"성찰: {reflection}")

# 에이전트 객체 생성
agent = InsightAgent()

# 시뮬레이션된 관찰 데이터 + KPI 목표 달성 여부
reports = [
    ("웹사이트 트래픽이 15% 증가했지만, 바운스율이 40%에서 55%로 급격히 증가했습니다.",
     False),
    ("이메일 열람률이 25%로 향상되었지만, 20% 목표를 초과했습니다.", True),
    ("장바구니 포기율이 60%에서 68%로 증가했지만, 50% 목표를 놓쳤습니다.",
     False),
    ("평균 주문 가치가 8% 증가했지만, 5% 증가 목표를 놓쳤습니다.", True),
    ("신규 구독자 수가 5% 감소했지만, 10% 성장 목표를 놓쳤습니다.",
     False),
]

# 1) 각 보고서를 기반으로 인사이트 생성 및 분류
for text, hit_target in reports:
    insight = agent.generate_insight(text)  # 인사이트 생성

    # KPI 달성 여부에 따라 승격 / 강등
    if hit_target:
        agent.promote_insight(insight)
    else:
        agent.demote_insight(insight)

# 2) 승격된 인사이트 중 하나를 사람이 직접 수정 (Human-in-the-loop)
if agent.promoted_insights:
    original = agent.promoted_insights[0]
    agent.edit_insight(
        original,
        f'개선된 인사이트: {original} 방문자 경험 개선을 위한 랜딩 페이지 UX 변경 조사'
    )

# 3) 현재 인사이트 상태 출력
agent.show_insights()

# 4) 최상위 인사이트 기반으로 다음 액션(실험 계획) 성찰
reflection_prompt = (
        "승격된 인사이트를 바탕으로, 다음 분기에 실행할 수 있는 하나의 고영향 실험을 제안하세요:"
        + f"\n{agent.promoted_insights}"
)

# 성찰 수행 (LLM 기반 전략 생성)
agent.reflect(reflection_prompt)