# LangChain에서 Tool, LLM 초기화, 메시지 클래스 import
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

# 외부 API 호출용
import requests

# =========================
# 1. 환경변수 확인
# =========================
import os
try:
    # .env 파일이 있으면 환경변수 로드
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv 없으면 무시
    pass

# OpenAI API Key 존재 여부 체크
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY가 설정되지 않았습니다."
        "환경변수 또는 .env 파일에서 설정해주세요."
    )

# =========================
# 2. Tool 정의
# =========================
@tool
def get_pokemon_type(pokemon: str) -> str:
    """
    포켓몬 이름을 입력받아 해당 포켓몬의 타입을 반환하는 Tool
    예: pikachu → electric
    """

    # PokeAPI URL (포켓몬 이름은 소문자로 처리)
    api_url = f"https://pokeapi.co/api/v2/pokemon/{pokemon.lower()}"

    try:
        # API 요청
        response = requests.get(api_url)

        # 요청 성공 시
        if response.status_code == 200:
            data = response.json()

            # 타입 정보 추출 (복수 타입 가능)
            type = [t["type"]["name"] for t in data["types"]]

            # 리스트를 문자열로 변환해서 반환
            return ", ".join(type)
        else:
            # API 실패 시
            return f"포켓몬의 타입을 가져오는데 실패했습니다: {pokemon}"

    except requests.exceptions.RequestException:
        # 네트워크 오류 등 예외 처리
        return f"포켓몬의 타입을 가져오는데 실패했습니다: {pokemon}"

# =========================
# 3. LLM 초기화 및 Tool 연결
# =========================

# GPT 모델 초기화 (temperature=0 → 일관된 응답)
llm = init_chat_model(model="gpt-5-mini", temperature=0)

# Tool을 모델에 바인딩
llm_with_tools = llm.bind_tools([get_pokemon_type])

# =========================
# 4. 사용자 질문 생성
# =========================

# 사용자 질문 (피카츄 타입 물어보기)
messages = [HumanMessage("피카츄의 타입은 무엇인가요?")]

# =========================
# 5. 1차 LLM 호출 (Tool 필요 여부 판단)
# =========================

# LLM이 질문을 보고 Tool 호출할지 결정
ai_msg = llm_with_tools.invoke(messages)

# LLM 응답을 메시지 리스트에 추가
messages.append(ai_msg)

# =========================
# 6. Tool 실행
# =========================

# LLM이 요청한 Tool 호출 실행
for tool_call in ai_msg.tool_calls:

    # Tool 실행 (LLM이 전달한 인자 사용)
    tool_msg = get_pokemon_type.invoke(tool_call)

    # 디버깅 출력
    print(tool_msg.name)            # Tool 이름
    print(tool_call['args'])        # 전달된 파라미터 (pokemon 이름)
    print(tool_msg.content)         # Tool 실행 결과 (타입 정보)

    # Tool 실행 결과를 메시지에 추가 (매우 중요)
    messages.append(tool_msg)
    print()

# =========================
# 7. 최종 응답 생성
# =========================

# Tool 결과를 반영해서 최종 자연어 응답 생성
final_response = llm_with_tools.invoke(messages)

# 최종 결과 출력
print(final_response.content)