import os
import requests

from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# -----------------------------
# 환경변수 로드 (.env 지원)
# -----------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# OpenAI API 키 확인 (없으면 실행 중단)
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY가 설정되지 않았습니다."
        "환경변수 또는 .env 파일에서 설정해주세요."
    )

# -----------------------------
# 1. Wolfram Alpha 도구
# -----------------------------
@tool
def query_wolfram_alpha(expression: str) -> str:
    """
    Wolfram Alpha에 질의를 보내 식을 계산하거나 정보를 조회합니다.
    Args: expression (str): 계산하거나 평가할 수식 또는 질의입니다.
    Returns: str: 계산 결과 또는 조회된 정보입니다.
        """

    # 수식을 URL 인코딩하여 API 요청 URL 생성
    api_url = f'''https://api.wolframalpha.com/v1/result?i={requests.utils.quote(expression)}&appid={os.getenv("WOLFRAM_ALPHA_APP_ID")}'''

    try:
        # GET 요청으로 Wolfram Alpha 호출
        response = requests.get(api_url)

        # 성공 시 결과 반환
        if response.status_code == 200:
            return response.text
        else:
            # API 오류 처리
            raise ValueError(f'''Wolfram Alpha API 오류: 
                    {response.status_code} - {response.text}''')

    except requests.exceptions.RequestException as e:
        # 네트워크 오류 처리
        raise ValueError(f"Wolfram Alpha 질의에 실패했습니다: {e}")


# -----------------------------
# 2. Zapier 웹훅 트리거 도구
# -----------------------------
@tool
def trigger_zapier_webhook(zap_id: str, payload: dict) -> str:
    """ 미리 정의된 Zap을 실행하기 위해 Zapier 웹훅을 트리거합니다.
    Args:
    zap_id (str): 트리거할 Zap의 고유 식별자입니다.
    payload (dict): Zapier 웹훅으로 전송할 데이터입니다.
    Returns:
    str: Zap이 성공적으로 트리거되었을 때의 확인 메시지입니다.
    Raises: ValueError: API 요청이 실패하거나 오류를 반환한 경우 발생합니다.
    """

    # Zapier 웹훅 URL 생성
    zapier_webhook_url = f"https://hooks.zapier.com/hooks/catch/{zap_id}/"

    try:
        # POST 요청으로 데이터 전달
        response = requests.post(zapier_webhook_url, json=payload)

        # 성공 여부 확인
        if response.status_code == 200:
            return f"Zapier 웹훅 '{zap_id}'이(가) 성공적으로 트리거되었습니다."

        else:
            raise ValueError(f'''Zapier API 오류: {response.status_code} - 
                         {response.text}''')

    except requests.exceptions.RequestException as e:
        raise ValueError(f"Zapier 웹훅 '{zap_id}' 트리거에 실패했습니다: {e}")


# -----------------------------
# 3. Slack 메시지 전송 도구
# -----------------------------
@tool
def send_slack_message(channel: str, message: str) -> str:
    """ 지정한 Slack 채널에 메시지를 보냅니다.
    Args:
    channel (str): 메시지를 보낼 Slack 채널 ID 또는 이름입니다.
    message (str): 전송할 메시지의 내용입니다.
    Returns:
    str: Slack 메시지가 성공적으로 전송되었을 때의 확인 메시지입니다.
    Raises: ValueError: API 요청이 실패하거나 오류를 반환한 경우 발생합니다.
    """

    api_url = "https://slack.com/api/chat.postMessage"

    # Slack 인증 헤더 (토큰 필요)
    headers = {
        "Authorization": "Bearer YOUR_SLACK_BOT_TOKEN",
        "Content-Type": "application/json"
    }

    # 요청 payload 구성
    payload = {
        "channel": channel,
        "text": message
    }

    try:
        # Slack API 호출
        response = requests.post(api_url, headers=headers, json=payload)
        response_data = response.json()

        # 성공 여부 확인
        if response.status_code == 200 and response_data.get("ok"):
            return f"Slack 채널 '{channel}'에 메시지가 성공적으로 전송되었습니다."
        else:
            error_msg = response_data.get("error", "Unknown error")
            raise ValueError(f"Slack API 오류: {error_msg}")

    except requests.exceptions.RequestException as e:
        raise ValueError(f'''Slack 채널 "{channel}"로 메시지 전송에 실패했습니다: {e}''')


# -----------------------------
# 4. LLM 초기화 및 도구 바인딩
# -----------------------------

# LLM 초기화 (temperature=0 → 항상 동일한 결과 유도)
llm = init_chat_model(model="gpt-4o-mini", temperature=0)

# 사용할 도구 리스트 정의
tools_list = [send_slack_message, query_wolfram_alpha, trigger_zapier_webhook]

# 도구 이름 → 실제 함수 매핑 (선택된 도구 실행용)
tools_by_name = {t.name: t for t in tools_list}

# LLM에 도구 연결 (Tool Calling 활성화)
llm_with_tools = llm.bind_tools(tools_list)


# -----------------------------
# 5. 사용자 입력
# -----------------------------
messages = [HumanMessage("3.15 * 12.25는 얼마인가요?")]


# -----------------------------
# 6. 1차 LLM 호출 (도구 선택 단계)
# -----------------------------
ai_msg = llm_with_tools.invoke(messages)

# LLM 응답을 메시지 히스토리에 추가
messages.append(ai_msg)


# -----------------------------
# 7. 선택된 도구 실행
# -----------------------------
for tool_call in ai_msg.tool_calls:

    # 선택된 도구 이름으로 실제 함수 가져오기
    chosen_tool = tools_by_name[tool_call["name"]]

    # LLM이 생성한 인자로 도구 실행
    result = chosen_tool.invoke(tool_call["args"])

    # 도구 실행 결과를 ToolMessage 형태로 변환
    tool_msg = ToolMessage(
        content=result if isinstance(result, str) else str(result),
        tool_call_id=tool_call["id"],
    )

    # 디버깅 출력
    print(chosen_tool.name)       # 어떤 도구가 선택됐는지
    print(tool_call["args"])      # 어떤 입력으로 호출됐는지
    print(tool_msg.content)       # 실행 결과

    # 결과를 메시지에 추가 (다음 LLM 호출에 사용됨)
    messages.append(tool_msg)
    print()


# -----------------------------
# 8. 최종 응답 생성
# -----------------------------
# 도구 실행 결과까지 포함하여 다시 LLM 호출
final_response = llm_with_tools.invoke(messages)

# 최종 자연어 답변 출력
print(final_response.content)