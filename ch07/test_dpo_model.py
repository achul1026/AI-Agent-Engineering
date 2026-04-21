#!/usr/bin/env python3
"""
DPO로 파인튜닝한 Phi-3 헬프데스크 모델 테스트 스크립트

📚 이 스크립트의 목표:
- DPO 학습으로 미세 조정된 Phi-3 모델이 IT 헬프데스크 질문에 잘 대답하는지 테스트
- 기본 모델 + LoRA 어댑터를 함께 로드해서 추론 수행

🔧 사용 방법:
1. 기본 예시들로 테스트:
   python ch07/test_dpo_model.py

2. 사용자 정의 프롬프트로 테스트:
   python ch07/test_dpo_model.py --prompt "비밀번호를 잊어버렸어요"

3. 모델 경로 지정 (이전 학습 결과가 다른 위치에 있을 경우):
   python ch07/test_dpo_model.py --adapter ./my_models/phi3-dpo

4. 생성 길이 조정:
   python ch07/test_dpo_model.py --max-tokens 512

⚠️ Flash Attention 경고 해결:
경고 메시지: "You are not running the flash-attention implementation, expect numerical differences."

🔍 원인 분석:
- 출처: Phi-3의 modeling_phi3.py의 Phi3Attention.forward() 함수
- 상황: Phi-3는 Flash Attention 2를 지원하지만, 다음 이유로 사용 불가:
  1. Flash Attention은 CUDA 전용 (Mac의 Metal Performance Shaders 미지원)
  2. Phi-3의 Sliding Window Attention 구현이 Flash Attention의 window_size 파라미터 필요
  3. Mac에서 pip install flash-attn으로 설치하면 CUDA 없어서 빌드 실패

💡 의미:
"Flash Attention 대신 표준 PyTorch Attention(eager)을 사용하므로,
 계산 결과가 아주 약간 다를 수 있습니다"는 정보성 안내
- 오류가 아님 (실제 동작에는 문제 없음)
- 수치 정확도: 단 몇 % 미만의 차이 (LLM 추론에 영향 무시)

✅ 해결 방법:
- Mac: attn_implementation="eager" 사용 (현재 스크립트의 방식)
- Linux + NVIDIA GPU:
  1. Flash Attention 설치: pip install flash-attn
  2. attn_implementation="flash_attention_2"로 변경
"""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ============================================================================
# 🔧 설정 변수
# ============================================================================

# 기본 모델 (Phi-3 Mini)
# - 4K 컨텍스트 윈도우: 4,096개 토큰까지 이해 가능
# - 3.8B 파라미터: 가볍고 빠름
BASE_MODEL = "microsoft/Phi-3-mini-4k-instruct"

# LoRA 어댑터 경로
# - fine_tune_helpdesk_dpo.py의 OUTPUT_DIR과 동일해야 함
# - 이 경로에는 다음 파일들이 있어야 함:
#   * adapter_config.json: LoRA 설정 (r=8, target_modules 등)
#   * adapter_model.bin: 학습된 LoRA 가중치
#   * tokenizer.model: 토크나이저
#   * special_tokens_map.json: 특수 토큰 정의
ADAPTER_PATH = "ch07/fine_tuned_model/phi3-mini-helpdesk-dpo"

# 기본 테스트 프롬프트들
# - IT 헬프데스크 시나리오로 구성
# - 학습 데이터와 유사한 유형의 질문들
# - 모델이 실제 헬프데스크 상황에서 얼마나 잘 대응하는지 테스트
DEFAULT_PROMPTS = [
    "비밀번호를 잊어버려서 이메일이 잠겼습니다.",
    # → 예상: 비밀번호 재설정 프로세스 설명

    "VPN이 매시간 연결이 끊깁니다.",
    # → 예상: VPN 문제 해결 단계

    "3층 로비의 프린터가 용지 걸림입니다.",
    # → 예상: 프린터 문제 해결

    "Google Drive의 재무 공유 폴더에 접근 권한을 주세요.",
    # → 예상: 접근 권한 설정 방법
]


# ============================================================================
# 🛠️ 핵심 함수 #1: 모델 로드
# ============================================================================

def load_model(adapter_path: str = ADAPTER_PATH):
    """
    베이스 모델(Phi-3)과 LoRA 어댑터를 로드하는 함수

    🔄 동작 흐름:
    1. 기본 모델(microsoft/Phi-3-mini-4k-instruct) 로드
       - 원본 33B 파라미터 모델에서 필요한 부분만 준비
    2. LoRA 어댑터 로드
       - 학습된 LoRA 가중치 추가 (전체의 1%)
    3. 두 모델을 병합해서 추론 준비

    📊 모델 구조:
    ```
    Base Model (Phi-3)
         ↓
    + LoRA Adapter (학습된 가중치)
         ↓
    Full Model (추론 사용)
    ```

    Args:
        adapter_path (str): LoRA 어댑터가 저장된 폴더 경로
                           기본값: ch07/fine_tuned_model/phi3-mini-helpdesk-dpo

    Returns:
        tuple: (model, tokenizer)
            - model: 추론 준비된 모델 (eval 모드)
            - tokenizer: 텍스트 토크나이즈 도구
    """

    # ============ 호환성 처리 ============
    # 이전 학습 결과가 다른 위치에 있을 수 있으므로 폴백(fallback) 지원
    if not os.path.exists(adapter_path) and adapter_path == ADAPTER_PATH:
        # 프로젝트 루트의 이전 학습 결과 확인
        fallback = "phi3-mini-helpdesk-dpo"
        if os.path.exists(fallback):
            adapter_path = fallback
            print(
                f"💡 참고: {ADAPTER_PATH}을 찾을 수 없습니다.\n"
                f"   대신 이전 학습 결과 '{fallback}' 사용합니다.\n"
            )

    print("🔄 모델 로딩 중...\n")

    # ============ 토크나이저 로드 ============
    # 학습할 때 사용한 토크나이저를 그대로 사용해야 함
    # (다른 토크나이저 사용하면 토큰이 다르게 인코딩될 수 있음)
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_path,  # 어댑터 폴더에 tokenizer.model 파일이 있음
        trust_remote_code=True  # 커스텀 코드 실행 허용
    )

    # ============ 베이스 모델 로드 ============
    # Phi-3 기본 모델 로드 (LoRA 없는 상태)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,

        # 메모리 효율: bfloat16 (32비트 float 대신 16비트 사용)
        torch_dtype=torch.bfloat16,

        # 디바이스 자동 할당:
        # - GPU 있으면 GPU에 로드
        # - 없으면 CPU에 로드
        device_map="auto",

        # Phi-3의 커스텀 코드 실행 허용
        trust_remote_code=True,

        # ⚠️ 주의: Mac에서는 flash_attention_2를 사용할 수 없음
        # 이유:
        # 1. Flash Attention: CUDA 전용 (Metal Performance Shaders 미지원)
        # 2. Phi-3: Sliding Window Attention 구현이 flash_attn의
        #    window_size 파라미터 지원 필요
        # 해결: eager(표준 PyTorch attention) 사용
        # - 수치 정확도는 거의 같음 (1% 미만 차이)
        # - 계산 속도는 조금 느림 (1-2배 정도)
        attn_implementation="eager",  # Mac: eager, Linux+GPU: flash_attention_2 가능
    )

    # ============ LoRA 어댑터 로드 및 병합 ============
    # PeftModel을 사용해 LoRA 가중치를 베이스 모델에 로드
    model = PeftModel.from_pretrained(base_model, adapter_path)

    # ============ 추론 모드 설정 ============
    # eval() 모드:
    # - Dropout 비활성화 (추론에서는 필요 없음)
    # - Batch Normalization을 추론 모드로 설정
    # - 더 일관된 결과 생성
    model.eval()

    print("✅ 모델 로드 완료!\n")
    print(f"📊 모델 정보:")
    print(f"   - 베이스 모델: {BASE_MODEL}")
    print(f"   - LoRA 어댑터: {adapter_path}")
    print(f"   - 디바이스: {model.device}")
    print(f"   - 데이터 타입: bfloat16\n")

    return model, tokenizer


# ============================================================================
# 🛠️ 핵심 함수 #2: 텍스트 생성
# ============================================================================

def generate(
        model,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 256
) -> str:
    """
    프롬프트에 대해 응답을 생성하는 함수

    🔄 생성 과정:
    1. 프롬프트를 Chat Template으로 포매팅
    2. 텍스트를 토큰으로 변환 (tokenize)
    3. 모델에 입력해서 토큰 생성
    4. 생성된 토큰을 텍스트로 변환 (decode)

    💡 Chat Template이란?
    - Phi-3 모델이 대화할 때 사용하는 특별한 형식
    - 예시:
      ```
      <|user|>
      비밀번호를 잊어버렸어요<|end|>
      <|assistant|>
      ```
    - 이 형식으로 포매팅해야 모델이 대화 맥락을 제대로 이해

    Args:
        model: 로드된 Phi-3 모델 (LoRA 어댑터 포함)
        tokenizer: 토크나이저 (같은 모델 버전의 것)
        prompt (str): 사용자 프롬프트
                     예: "비밀번호를 잊어버렸어요"
        max_new_tokens (int): 생성할 최대 토큰 수
                             기본값: 256 (약 200-250 단어)

    Returns:
        str: 모델이 생성한 응답 텍스트

    생성 파라미터 설명:
    - do_sample=True: 결정론적이 아닌 샘플링 사용
      * True: 매번 다른 응답 생성 (자연스러움)
      * False: 항상 같은 응답 생성 (일관성)

    - temperature=0.7: 창의성 수준 조절
      * 0.0: 가장 확률 높은 단어만 선택 (보수적)
      * 0.7: 균형잡힌 창의성 (권장)
      * 1.0+: 매우 창의적이고 무작위적

    - use_cache=False: KV 캐시 비활성화
      * Phi-3 DynamicCache 호환성 이슈 회피
      * 캐시 없으면 계산이 반복되지만 안정적
    """

    # ============ 단계 1: Chat Template 포매팅 ============
    # 사용자 입력을 Phi-3의 대화 형식으로 변환
    messages = [{"role": "user", "content": prompt}]

    # Chat Template 적용
    # - 입력: messages 리스트
    # - 출력: 형식화된 텍스트 문자열
    # 예시 변환:
    # 입력: [{"role": "user", "content": "안녕하세요"}]
    # 출력: "<|user|>\n안녕하세요<|end|>\n<|assistant|>\n"
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # 아직 토큰화하지 않고 문자열로만 반환
        add_generation_prompt=True,  # <|assistant|> 태그 추가 (응답 준비 완료 신호)
    )

    print(f"📝 포매팅된 프롬프트 (내부):\n{text}\n")

    # ============ 단계 2: 토크나이제이션 ============
    # 텍스트 문자열을 토큰 ID로 변환
    # 예: "안녕" → [1234, 5678]
    inputs = tokenizer(
        text,
        return_tensors="pt"  # PyTorch 텐서 형식으로 반환
    ).to(model.device)  # 모델과 같은 디바이스(GPU/CPU)로 이동

    # ============ 단계 3: 텍스트 생성 ============
    # torch.no_grad(): 그래디언트 계산 비활성화
    # - 추론할 때는 역전파(backprop)가 필요 없음
    # - 메모리 절약, 속도 향상
    with torch.no_grad():
        outputs = model.generate(
            **inputs,  # input_ids, attention_mask 등 포함

            max_new_tokens=max_new_tokens,
            # 생성할 최대 "새로운" 토큰 수
            # (입력 토큰은 포함하지 않음)

            do_sample=True,
            # True: 확률에 따른 샘플링 사용
            #       (다양하고 자연스러운 응답)
            # False: 가장 확률 높은 토큰만 선택
            #       (반복적이고 보수적)

            temperature=0.7,
            # 0.7은 LLM 추론의 "황금 수치"
            # - 너무 낮으면(0.1): 반복적, 단조로움
            # - 적절함(0.7): 창의성과 일관성 균형
            # - 너무 높으면(1.5): 깨진 문장, 무의미

            pad_token_id=tokenizer.eos_token_id,
            # 패딩 토큰: <eos_token_id> 사용
            # (특별한 "끝" 토큰)

            use_cache=False,
            # KV 캐시 비활성화
            # 배경: Phi-3는 DynamicCache 사용하는데,
            #      eager attention과 호환성 문제 있을 수 있음
            # 해결: 캐시 없이 생성 (약간의 속도 트레이드오프)
        )

    # ============ 단계 4: 생성된 토큰 추출 ============
    # outputs: [배치, 입력_길이 + 새로운_토큰_수]
    # 입력 부분을 자르고 생성 부분만 추출
    generated = outputs[0][inputs["input_ids"].shape[1]:]

    # ============ 단계 5: 디코딩 ============
    # 토큰 ID를 다시 문자열로 변환
    response = tokenizer.decode(
        generated,
        skip_special_tokens=False  # 특수 토큰 포함 (<|end|> 등)
    )

    return response


# ============================================================================
# 🎯 메인 함수
# ============================================================================

def main():
    """
    메인 함수: 커맨드 라인 인터페이스 처리 및 테스트 실행

    🔄 동작 흐름:
    1. 커맨드 라인 인수 파싱
    2. 모델 로드
    3. 프롬프트 처리
    4. 각 프롬프트에 대해 응답 생성 및 출력

    💡 커맨드 라인 인수:
    --prompt: 사용자 정의 프롬프트 (지정하면 기본 예시 무시)
    --adapter: LoRA 어댑터 경로 (다른 경로에 저장된 경우)
    --max-tokens: 생성 길이 조정 (짧은 답변이 필요한 경우)
    """

    # ============ 커맨드 라인 파서 설정 ============
    parser = argparse.ArgumentParser(
        description="DPO 파인튜닝 모델로 IT 헬프데스크 질문에 답변하기"
    )

    # 선택적 인수 1: 사용자 정의 프롬프트
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,  # None이면 기본 예시들 사용
        help="테스트할 프롬프트 (미지정 시 DEFAULT_PROMPTS의 예시들 사용)",
    )

    # 선택적 인수 2: LoRA 어댑터 경로
    parser.add_argument(
        "--adapter",
        type=str,
        default=ADAPTER_PATH,
        help=f"LoRA 어댑터 경로 (기본값: {ADAPTER_PATH})",
    )

    # 선택적 인수 3: 생성 길이
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="생성할 최대 토큰 수 (기본값: 256)",
    )

    # 인수 파싱
    args = parser.parse_args()

    # ============ 모델 로드 ============
    model, tokenizer = load_model(args.adapter)

    # ============ 프롬프트 결정 ============
    # --prompt 지정됨: 사용자 입력만 사용
    # --prompt 미지정: DEFAULT_PROMPTS의 모든 예시 사용
    prompts = [args.prompt] if args.prompt else DEFAULT_PROMPTS

    # ============ 각 프롬프트에 대해 응답 생성 ============
    for i, prompt in enumerate(prompts):
        print(f"{'='*70}")
        print(f"📌 테스트 #{i+1}")
        print(f"{'='*70}")
        print(f"👤 사용자: {prompt}")
        print(f"{'-'*70}")

        # 응답 생성
        response = generate(model, tokenizer, prompt, args.max_tokens)

        print(f"🤖 모델 응답:\n{response.strip()}")
        print()

    print(f"{'='*70}")
    print(f"✅ 테스트 완료!")
    print(f"{'='*70}")


# ============================================================================
# 💻 프로그램 실행 진입점
# ============================================================================

if __name__ == "__main__":
    """
    if __name__ == "__main__": 의미?
    
    🎯 이 블록은 스크립트가 직접 실행될 때만 동작
    
    시나리오 1: 직접 실행
    > python ch07/test_dpo_model.py
    → if __name__ == "__main__": 블록 실행
    
    시나리오 2: 다른 파일에서 import
    > from ch07.test_dpo_model import generate
    → if __name__ == "__main__": 블록 실행 안 됨
    → generate 함수만 import됨
    
    💡 이렇게 하는 이유:
    - 라이브러리로도, 스크립트로도 사용 가능하게 함
    - import할 때 자동 실행 방지 (부작용 없음)
    """
    main()