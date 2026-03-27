# ================================
# 1. 라이브러리 import
# ================================

# LLM(ChatGPT 모델) 생성 함수
from langchain.chat_models import init_chat_model

# Wikipedia 검색을 "Tool 형태"로 제공하는 클래스
from langchain_community.tools import WikipediaQueryRun

# Wikipedia API를 실제로 호출하는 래퍼
from langchain_community.utilities import WikipediaAPIWrapper

# 사용자 메시지 객체
from langchain_core.messages import HumanMessage


# ================================
# 2. 환경 변수 로드 및 검증
# ================================

import os
try:
    # .env 파일에서 API 키 로드 (개발 환경)
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# OpenAI API 키 없으면 실행 중단
if not os.getenv("OPENAI_API_KEY"):
    raise Exception(
        "OPENAI_API_KEY가 설정되지 않았습니다."
        "환경변수 또는 .env 파일에서 설정해주세요."
    )


# ================================
# 3. Wikipedia Tool 생성
# ================================

# 👉 실제 Wikipedia API 호출 설정
api_wrapper = WikipediaAPIWrapper(
    top_k_results=1,            # 검색 결과 1개만 가져옴
    doc_content_chars_max=300   # 최대 300자까지만 내용 반환
)

# 👉 LLM이 사용할 "도구" 생성
tool = WikipediaQueryRun(api_wrapper=api_wrapper)

# 구조:
# LLM → WikipediaQueryRun (tool) → WikipediaAPIWrapper → 실제 Wikipedia


# ================================
# 4. LLM 생성 + Tool 연결
# ================================

# GPT 모델 초기화
llm = init_chat_model(model="gpt-5-mini", temperature=0)

# 👉 LLM에게 "Wikipedia tool을 써도 된다"라고 알려주는 단계
llm_with_tools = llm.bind_tools([tool])


# ================================
# 5. 사용자 질문 생성
# ================================

messages = [
    HumanMessage("Buzz Aldrin의 주요 업적은 무엇인가요?")
]

# 👉 질문 의미:
# "버즈 올드린 업적 알려줘"
# → LLM 입장에서 "검색 필요" 상황


# ================================
# 6. 1차 LLM 호출 (Tool 사용 판단)
# ================================

# 👉 여기서 OpenAI API 호출 발생
ai_msg = llm_with_tools.invoke(messages)

# 결과:
# - 그냥 답변할 수도 있고
# - Wikipedia tool 호출 요청(tool_calls) 생성할 수도 있음

messages.append(ai_msg)


# ================================
# 7. Tool 실행 (Wikipedia 검색)
# ================================

# ai_msg.tool_calls 구조 예시:
# [
#   {
#     "name": "wikipedia",
#     "args": {"query": "Buzz Aldrin"},
#     "id": "call_123"
#   }
# ]

for tool_call in ai_msg.tool_calls:

    # 👉 Wikipedia 검색 실행
    # 내부 동작:
    # 1. tool_call['args']에서 query 추출
    # 2. Wikipedia API 호출
    # 3. 결과 텍스트 반환
    tool_msg = tool.invoke(tool_call)

    # 어떤 tool인지 출력
    print(tool_msg.name)

    # LLM이 어떤 검색어로 요청했는지 확인
    print(tool_call['args'])

    # Wikipedia에서 가져온 실제 내용 출력
    print(tool_msg.content)

    # 👉 Tool 결과를 LLM에게 전달하기 위해 메시지 추가
    messages.append(tool_msg)

    print()


# ================================
# 8. 최종 LLM 호출 (결과 정리)
# ================================

# 👉 다시 OpenAI API 호출
# 이번에는 Wikipedia 결과까지 포함됨
final_response = llm_with_tools.invoke(messages)

# 👉 LLM이 자연어로 정리해서 답변 생성
print(final_response.content)