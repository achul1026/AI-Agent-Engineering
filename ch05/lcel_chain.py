from langchain_core.runnables import RunnableLambda
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
import os

# 환경변수 로드
try:
    from dotenv import load_dotenv
    load_dotenv()  # .env 파일에 정의된 환경변수 로드 (예: OPENAI_API_KEY)
except ImportError:
    pass  # dotenv가 없으면 무시하고 진행

# 함숫값 또는 모델 호출을 Runnable로 감싼다.
# RunnableLambda는 callable을 직접 인자로 받습니다.

# Chat 모델 초기화 (결정적 응답을 위해 temperature=0)
llm_model = init_chat_model(model="gpt-5-mini", temperature=0)

# LLM의 invoke 메서드를 Runnable로 감싸서 체인에서 사용할 수 있게 함
# → 일반 함수처럼 체인에 연결 가능
llm = RunnableLambda(llm_model.invoke)

# prompt 생성용 Runnable
# 입력 텍스트를 받아 PromptTemplate → 메시지 형식으로 변환
prompt = RunnableLambda(lambda text:
                        PromptTemplate.from_template(text)  # 템플릿 생성
                        .format_prompt()                   # 포맷 적용
                        .to_messages())                    # LLM 입력용 메시지로 변환

# 기존 체인과 동일한 형태:
# chain = LLMChain(prompt=prompt, llm=llm)

# 파이프 연산자를 사용하는 LCEL 체인:
# → prompt → llm 순서로 데이터가 흐름
# → prompt의 출력이 llm의 입력으로 전달됨
chain = prompt | llm

# 체인 실행
if __name__ == '__main__':
    # 사용자 입력을 체인에 전달
    # 1. prompt Runnable에서 메시지로 변환
    # 2. llm Runnable에서 모델 호출
    result = chain.invoke("프랑스의 수도는 어디인가요?")

    # 결과는 AIMessage 형태이므로 content 출력
    print(result.content)