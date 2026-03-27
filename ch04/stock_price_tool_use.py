# LangChain에서 Tool 데코레이터와 메시지 관련 클래스 import
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

# 외부 API 호출을 위한 requests 라이브러리
import requests

# =========================
# 1. 환경변수 로딩
# =========================
import os
try:
    # .env 파일이 있을 경우 환경변수 로드
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv가 없으면 그냥 패스
    pass

# OPENAI API KEY 존재 여부 체크 (없으면 에러 발생)
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY가 설정되지 않았습니다."
        "환경변수 또는 .env 파일에서 설정해주세요."
    )

# =========================
# 2. Tool 정의
# =========================
@tool
def get_stock_price(ticker: str) -> float:
    """
    주식 티커(symbol)를 입력받아 해당 주식의 현재 가격을 반환하는 Tool
    LLM이 필요 시 자동으로 호출할 수 있음
    """

    # (예시용) 외부 주식 API URL
    api_url = f"https://api_example.com/stocks/{ticker}"

    try:
        # GET 요청으로 주식 데이터 조회
        response = requests.get(api_url)

        # 정상 응답일 경우
        if response.status_code == 200:
            # JSON에서 price 값 추출
            return response.json()["price"]
        else:
            # 실패 시 에러 메시지 반환
            return f"주식 가격을 가져오는데 실패했습니다: {ticker}"

    except requests.exceptions.RequestException:
        # 네트워크 에러 등 예외 처리
        return f"주식 가격을 가져오는데 실패했습니다: {ticker}"

# =========================
# 3. LLM 초기화 및 Tool 연결
# =========================

# GPT 모델 초기화 (temperature=0 → 결정론적 응답)
llm = init_chat_model(model="gpt-5-mini", temperature=0)

# Tool을 LLM에 바인딩 (LLM이 필요 시 자동 호출 가능)
llm_with_tools = llm.bind_tools([get_stock_price])

# =========================
# 4. 사용자 질문 생성
# =========================

# 사용자 메시지 생성
messages = [HumanMessage("애플의 현재 주식 가격은 얼마인가요?")]

# =========================
# 5. 1차 LLM 호출 (Tool 호출 판단)
# =========================

# LLM이 메시지를 보고 Tool 호출 여부 판단
ai_msg = llm_with_tools.invoke(messages)

# AI 응답을 메시지 리스트에 추가
messages.append(ai_msg)

# =========================
# 6. Tool 실행
# =========================

# LLM이 요청한 Tool 호출 실행
for tool_call in ai_msg.tool_calls:

    # Tool 실행 (LLM이 전달한 args 사용)
    tool_msg = get_stock_price.invoke(tool_call)

    # 디버깅 출력
    print(tool_msg.name)            # 사용된 Tool 이름
    print(tool_call['args'])        # Tool에 전달된 인자
    print(tool_msg.content)         # Tool 실행 결과

    # Tool 결과를 다시 메시지에 추가 (LLM이 후속 답변 생성할 수 있게)
    messages.append(tool_msg)
    print()

# =========================
# 7. 최종 응답 생성
# =========================

# Tool 실행 결과를 기반으로 최종 답변 생성
final_response = llm_with_tools.invoke(messages)

# 사용자에게 보여줄 최종 자연어 응답 출력
print(final_response.content)