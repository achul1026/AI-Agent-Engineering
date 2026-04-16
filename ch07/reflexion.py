from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langchain_core.messages import HumanMessage
import json

# 환경 변수 설정 (로컬 .env 파일을 로드하여 API KEY 등을 사용하기 위함)
import os
try:
    from dotenv import load_dotenv
    load_dotenv()  # .env 파일 로드
except ImportError:
    pass

# OpenAI API Key 존재 여부 체크
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_KPI_KEY가 설정되지 않았습니다."
        "환경변수 또는 .env 파일에서 설정해주세요."
    )

# LLM 초기화 (gpt-5-mini 모델 사용, temperature=0 → 결정론적 응답)
llm = init_chat_model(model="gpt-5-mini", temperature=0)

# reflection 결과를 저장할 리스트
reflections = []

# LangGraph에서 사용할 노드 함수 (모델 호출)
def call_model(state: MessagesState):
    # 현재 state의 messages를 그대로 LLM에 전달
    response = llm.invoke(state["messages"])
    # 반환값은 반드시 dict 형태로 (LangGraph 규칙)
    return {"messages": response}

# Reflexion 기법을 위한 프롬프트
reflexion_prompt = f"""어떤 환경에 놓인 뒤 특정 과제를 수행하라는 지시를 받았던 과거 경험의 기록이 주어질 것이다.
당신은 그 과제를 수행하는 데 실패했습니다. 환경을 요약하려고 하지 말고, 과제를 완수하기 위해 시도했던 전략과 진행 경로에 대해 생각해 보십시오.
당신이 했어야 하지만 하지 못했던 구체적인 행동들을 참고하여, 그 실수를 보완할 수 있는 간결하고 새로운 실행 계획을 세우십시오.
예를 들어, A와 B는 시도했지만 C를 잊어버렸다면, 해당 환경에서 C를 달성하기 위해 어떤 행동을 했어야 하는지 환경에 특화된 행동들로 계획을 세우면 됩니다.
이 계획은 나중에 같은 과제를 다시 풀 때 필요하게 될 것입니다. "Plan"이라는 단어 뒤에 자신의 계획을 작성하십시오.

Instruction:
저는 유제품이 들어 있지 않고 사과 맛이 포함된 칩스 버라이어티 팩을 찾고 있으며, 가격은 30달러 미만이어야 합니다.

[Search]

Action: search[dairy free and apple variety pack of chips]
Observation:
[검색 결과로 돌아가기]
페이지 1 (총 결과:50)
[다음 >]
[B0HRFJWP8]
Enjoy Life Foods Soft Baked Ovals, Breakfast Bars, Nut Free Bars, Soy Free, Dairy Free, Non GMO, Gluten Free, Vegan, Variety Pack, 4 Boxes(20 Bars Total)
 $100.0
[B01KMHY5PG]
Enjoy Life Soft Baked Chewy Bars, Variety Pack, Nut Free Bars, Soy Free, Dairy Free, Gluten Free, 6 Boxes(30 Total Bars)
$21.49
[BOO8D2X8C4]
Enjoy Life Lentil Chips Variety Pack, Dairy Free Chips, Soy Free, Net Free, Non GMO, Vegan, Gluten Free, 24 Bars(0.8xz)
$100

Action: think[앞의 두 상품은 제가 원하는 것이 아닙니다. 세 번째 상품이 제가 찾는, 유제품들이 들어 있지 않고 사과 맛이 포함된 칩스 버라이어티 팩입니다.]
Observation: 알겠습니다.

Action: click[[BOO8D2X8C4]]
Observation:
[검색 결과로 돌아가기]
[< 이전]
flavor name [dill and sour cream][garlic &parmesan][light sea salt][margherita pizza][thai chili lime][vaiety pack]
가격(Price): $100.0
평점(Rating): N.A.
[Description]
[Features]
[Reviews]
[Buy now]

Action: think[이 상품에는 'variety pack'과 '0.8 ounce (pack of 24)' 옵션이 있고, 구매하기에 좋아 보입니다.]
Observation: 알겠습니다.

Action: click[variety pack]
Observation: variety pack을 클릭하셨습니다.

Action: [0.8 ounce (pack of 24)]
Observation: 0.8 ounce (pack of 24)를 클릭하셨습니다.

Action: click[Buy Now]

STATUS: FAIL

Plan:
"""  # 실패한 과정을 기반으로 "다음에 어떻게 해야 하는지" 계획을 생성하도록 유도

# LangGraph 그래프 생성
builder = StateGraph(MessagesState)

# "reflexion"이라는 노드 추가 (LLM 호출)
builder.add_node("reflexion", call_model)

# 시작 → reflexion 노드로 연결
builder.add_edge(START, "reflexion")

# 그래프 컴파일 (실행 가능한 상태로 변환)
graph = builder.compile()

# 그래프 실행 (프롬프트를 메시지로 전달)
result = graph.invoke(
    {
        "messages": [
            HumanMessage(
                reflexion_prompt  # 위에서 정의한 실패 사례 기반 프롬프트
            )
        ]
    }
)

# 결과를 리스트에 저장 (추후 누적 가능)
reflections.append(result)

# ===================== 결과 출력 =====================

print("\n"+ "="*80)
print(" Reflextion 결과")
print("="*80+"\n")

# 메시지 순회하면서 출력
for i, msg in enumerate(result["messages"]):
    msg_type = msg.__class__.__name__  # 메시지 타입 확인 (Human / AI)

    if msg_type == "HumanMessage":
        print(f"입력 메시지:")
        print("-" * 80)
        # 프롬프트가 너무 길면 일부만 표시
        content = msg.content
        if len(content) > 500:
            print(content[:250] + "\n\n ...(중략) ..." + "\n\n" + content[-250:])
        else:
            print(content)
        print()

    elif msg_type == "AIMessage":
        print(f"AI 응답:")
        print("-" * 80)
        print(msg.content)  # 모델이 생성한 Reflection Plan 출력
        print()

print("=" * 80)