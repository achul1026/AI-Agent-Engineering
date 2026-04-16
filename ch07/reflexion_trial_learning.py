from typing import Annotated, List, Dict, Any
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langchain_core.messages import HumanMessage
import json

# 환경변수 확인 (API Key 등을 .env에서 로드)
import os
try:
    from dotenv import load_dotenv
    load_dotenv()  # .env 파일 로드
except ImportError:
    pass

# OpenAI API Key 존재 여부 체크
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY가 설정되지 않았습니다."
        "환경변수 또는 .env 파일에서 설정해주세요."
    )

# LLM 초기화 (temperature=0 → 일관된 응답)
llm = init_chat_model(model="gpt-5-mini", temperature=0)

# reflection 결과 저장용 리스트
reflections = []

# Reflexion 프롬프트 (실패 사례 기반으로 다음 계획 생성 유도)
reflexion_prompt = f"""어떤 환경에 놓인 뒤 특정 과제를 수행하라는 지시를 받았던 과거 경험의 기록이 주어질 것입니다.
당신은 그 과제를 완수하는 데 실패했습니다. 환경을 요약하려고 하지 말고, 과제를 완수하기 위해 시도했던 전략과 진행 경로에 대해 생각해 보십시오.
당신이 했어야 하지만 하지 못했던 구체적인 행동들을 참고하여, 그 실수를 보완할 수 있는 간결하고 새로운 실행 계획을 세우십시오.
예를 들어, A와 B는 시도했지만 C를 잊어버렸다면, 해당 환경에서 C를 달성하기 위해 어떤 행동을 했어야 하는지 환경에 특화된 행동들로 계획을 세우면 됩니다.
이 계획은 나중에 같은 과제를 다시 풀 때 필요하게 될 것입니다. "Plan"이라는 단어 뒤에 자신의 계획을 작성하십시오.

Instruction:
저는 유제품이 들어 있지 않고 사과 맛이 포함된 칩스 버라이어티 팩을 찾고 있으며, 가격은 30달러 미만이어야 합니다.

[Search]

Action: search[dairy free and apple variety pack of chips]
Observation:
[검색 결과로 돌아가기]
페이지 1 (총 결과: 50)
[다음 >]
[B07HRFJWP8]
Enjoy Life Foods Soft Baked Ovals, Breakfast Bars, Nut Free Bars, Soy Free, Dairy Free, Non GMO, Gluten Free, Vegan, Variety Pack, 4 Boxes (20 Bars Total)
$100.0
[B01KMHY5PG]
Enjoy Life Soft Baked Chewy Bars, Variety Pack, Nut Free Bars, Soy Free, Dairy Free, Gluten Free, 6 Boxes (30 Total Bars)
$21.49
[B008D2X8C4]
Enjoy Life Lentil Chips Variety Pack, Dairy Free Chips, Soy Free, Nut Free, Non GMO, Vegan, Gluten Free, 24 Bags (0.8 oz)
$100.0

Action: think[앞의 두 상품은 제가 원하는 것이 아닙니다. 세 번째 상품이 제가 찾는, 유제품이 들어 있지 않고 사과 맛이 포함된 칩스 버라이어티 팩입니다.]
Observation: 알겠습니다.

Action: click[B008D2X8C4]
Observation:
[검색 결과로 돌아가기]
[< 이전]
flavor name [dill and sour cream][garlic & parmesan][light sea salt][margherita pizza][thai chili lime][variety pack]
size [0.8 ounce (pack of 24)][4 ounce (pack of 12)]
가격(Price): $100.0
평점(Rating): N.A.
[Description]
[Features]
[Reviews]
[Buy Now]

Action: think[이 상품에는 'variety pack'과 '0.8 ounce (pack of 24)' 옵션이 있고, 구매하기에 좋아 보입니다.]
Observation: 알겠습니다.

Action: click[variety pack]
Observation: variety pack을 클릭하셨습니다.

Action: click[0.8 ounce (pack of 24)]
Observation: 0.8 ounce (pack of 24)를 클릭하셨습니다.

Action: click[Buy Now]

STATUS: FAIL

Plan:
"""

def get_completion(prompt: str) -> str:
    # 텍스트 완성 헬퍼 함수 (단일 프롬프트 실행)
    result = llm.invoke([HumanMessage(content=prompt)])
    return result.content  # 결과 텍스트 반환

def call_model(state: MessagesState):
    # LangGraph에서 사용하는 노드 함수
    # 현재 state에 있는 메시지를 그대로 모델에 전달
    response = llm.invoke(state["messages"])
    return {"messages": [response]}  # 반드시 dict 형태로 반환

def _generate_reflection_query(trial_log: str, recent_reflections: List[str]):
    # 과거 reflection 기록 + 현재 로그를 합쳐 새로운 reflection 프롬프트 생성
    history = "\n\n".join(recent_reflections)  # 최근 reflection들을 하나의 문자열로 결합
    return f'''{history}
        {trial_log}
        이 경험을 바탕으로 다음 계획을 세우세요. Plan:'''

def update_memory(trial_log_path: str, env_configs: List[Dict[str, Any]]):
    """주어진 env_config에 적절한 성찰(reflection)을 업데이트합니다."""

    # trial log 파일 읽기
    with open(trial_log_path, 'r') as f:
        full_log: str = f.read()

    # 환경별 로그 분리 (구분자 기준)
    env_logs: List[str] = full_log.split('#####\n\n#####')

    # 로그 개수와 env_config 개수가 동일한지 검증
    assert len(env_logs) == len(env_configs), print(f'bad: {env_logs}')

    # 각 환경별로 reflection 생성
    for i, env in enumerate(env_configs):
        # 실패한 경우 + skip이 아닌 경우만 처리
        if not env['is_success'] and not env['skip']:

            # 최근 memory 3개까지만 사용 (컨텍스트 길이 제한)
            if len(env['memory']) > 3:
                memory: List[str] = env['memory'][-3:]
            else:
                memory: List[str] = env['memory']

            # reflection 프롬프트 생성
            reflection_query = _generate_reflection_query(env_logs[i], memory)

            # LLM을 통해 새로운 reflection 생성
            reflection = get_completion(reflection_query)

            # memory에 추가 (지속적으로 학습되는 구조)
            env_configs[i]['memory'] += [reflection]


# LangGraph 구성
builder = StateGraph(MessagesState)

# reflexion 노드 등록 (LLM 호출)
builder.add_node("reflexion", call_model)

# 시작 → reflexion 노드 연결
builder.add_edge(START, "reflexion")

# 그래프 컴파일
graph = builder.compile()

# 그래프 실행 (Reflexion 프롬프트 입력)
result = graph.invoke(
    {
        "messages": [
            HumanMessage(
                reflexion_prompt  # 실패 사례 기반 입력
            )
        ]
    }
)

# 결과 저장
reflections.append(result)

# ===================== 메모리 업데이트 테스트 =====================

trial_log_path = "ch07/data/trial_logs.txt"

# 환경 설정 (실패 상태 → reflection 생성 대상)
env_configs = [
    {
        'is_success': False,
        'skip': False,
        'memory': []  # 과거 경험 저장 공간
    }
]

# memory 업데이트 실행
update_memory(trial_log_path, env_configs)

# 결과 출력
print(f"업데이트된 메모리: {env_configs[0]['memory']}")

# ===================== 결과 출력 =====================

print("\n" + "="*80)
print("🔄 Reflexion 결과")
print("="*80 + "\n")

# 메시지 순회 출력
for i, msg in enumerate(result["messages"]):
    msg_type = msg.__class__.__name__

    if msg_type == "HumanMessage":
        print(f"입력 메시지:")
        print("-" * 80)
        # 프롬프트가 너무 길 경우 일부만 출력
        content = msg.content
        if len(content) > 500:
            print(content[:250] + "\n\n... (중략) ...\n\n" + content[-250:])
        else:
            print(content)
        print()

    elif msg_type == "AIMessage":
        print(f"AI 응답:")
        print("-" * 80)
        print(msg.content)  # 생성된 Plan 출력
        print()

print("="*80)