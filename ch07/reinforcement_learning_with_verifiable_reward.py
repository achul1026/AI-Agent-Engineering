"""
RLVR(Reinforcement Learning with Verifiable Rewards) - GRPO로 도구 호출 품질 학습

🎯 이 코드의 목표:
- 모델이 도구(함수)를 올바르게 호출하도록 학습
- 예: IT 헬프데스크에서 "get_ticket_details" 함수를 올바른 파라미터로 호출
- GRPO라는 강화학습 알고리즘으로 여러 보상 신호를 통합해 학습

📚 주요 개념:
1. GRPO: Group Relative Policy Optimization (그룹 기반 상대적 정책 최적화)
2. Reward Function: 모델의 행동(도구 호출)이 얼마나 좋은지 점수 부여
3. Verifiable Rewards: 실제로 검증 가능한 보상 (정확한 도구 호출 여부)
"""

import json
import logging
import os
import re
from typing import Any, Dict, List

import torch
from datasets import load_dataset
from huggingface_hub import constants as hf_constants
from trl import GRPOConfig, GRPOTrainer

# ============================================================================
# 🔧 설정 변수
# ============================================================================

# RLVR 훈련 데이터 경로
# 필수 컬럼:
# - prompt: 사용자의 질문/명령
# - chosen: 올바른 도구 호출 (모범 응답)
# - rejected: 잘못된 도구 호출 (반면 응답)
# - label: 예상되는 도구명 (예: "get_ticket_details")
RLVR_DATA = "ch07/training_data/rlvr_it_help_desk_training_data.json"

# 사용할 기본 모델
# Qwen2-0.5B: 아주 가벼운 모델 (500M 파라미터)
# - Phi-3 (3.8B)보다 작음
# - 빠른 학습과 추론 가능
GRPO_MODEL = "Qwen/Qwen2-0.5B-Instruct"


# ============================================================================
# 🛠️ 유틸리티 함수
# ============================================================================

def _is_model_cached(repo_id: str) -> bool:
    """
    Hugging Face 모델이 로컬 캐시에 있는지 확인하는 함수

    동작:
    1. 로컬 경로로 존재하는지 확인 (예: "./models/qwen2")
    2. Hugging Face 캐시 폴더에 있는지 확인
       - 캐시 위치: ~/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct

    Returns:
        bool: 캐시되어 있으면 True
    """
    if os.path.exists(repo_id) and os.path.isdir(repo_id):
        return True

    cache_folder = "models--" + repo_id.replace("/", "--")
    cache_path = os.path.join(hf_constants.HF_HUB_CACHE, cache_folder)
    return os.path.exists(cache_path)


# ============================================================================
# 📝 로깅 설정
# ============================================================================

logger = logging.getLogger(__name__)

# 모델이 캐시되어 있지 않으면 경고 메시지
if not _is_model_cached(GRPO_MODEL):
    logger.warning(
        "로컬 캐시에서 '%s'를 찾을 수 없습니다. "
        "Hugging Face Hub에서 다운로드합니다...",
        GRPO_MODEL
    )


# ============================================================================
# 1️⃣ 데이터셋 로드
# ============================================================================

# RLVR 데이터 구조 예시:
# {
#   "prompt": "ticket #12345의 상세 정보를 가져와줘",
#   "chosen": "<tool_call>{\"name\": \"get_ticket_details\", \"parameters\": {\"ticket_id\": \"12345\"}}</tool_call>",
#   "rejected": "<tool_call>{\"name\": \"create_ticket\", \"parameters\": {\"title\": \"12345\"}}</tool_call>",
#   "label": "get_ticket_details"
# }

dataset = load_dataset("json", data_files=RLVR_DATA, split="train")

print(f"✅ 데이터셋 로드 완료")
print(f"   - 파일: {RLVR_DATA}")
print(f"   - 샘플 수: {len(dataset)}")


# ============================================================================
# 2️⃣ 보상 함수 #1: 도구 호출 품질 평가
# ============================================================================

def reward_tool_call_quality(completions: List[str], **kwargs) -> List[float]:
    """
    함수 호출 품질에 대한 세밀한 보상 함수

    📊 보상 체계:
    +1.0  : 올바른 도구명 + 유효한 JSON + 필수 매개변수 모두 포함 ✨ (최고)
    +0.5  : 올바른 도구명 + 유효한 JSON + 일부 매개변수 누락 🟡
    +0.2  : 올바른 도구명은 있지만 JSON 파싱 실패 🟠
    -0.3  : 잘못된 도구명 + 유효한 JSON ❌
    -0.5  : 잘못된 도구명 + JSON 파싱 실패 ❌❌
    -1.0  : 도구 호출 없거나 완전히 잘못됨 🔴 (최악)

    Args:
        completions (List[str]): 모델이 생성한 응답들
                                예: ["<tool_call>{...}</tool_call> text..."]

        **kwargs: 추가 정보
            - labels: 예상되는 도구명 목록
                     예: ["get_ticket", "create_ticket"]
            - required_params: 필수 매개변수 이름들
                              예: [["ticket_id"], ["title", "description"]]
            - num_generations: 각 샘플마다 생성한 응답 개수
                              (GRPO에서 여러 응답을 생성할 때 사용)

    Returns:
        List[float]: 각 응답에 대한 보상 점수
    """

    # kwargs에서 추가 정보 추출
    labels = kwargs.get("labels", [])
    required_params = kwargs.get('required_params', [])

    rewards = []
    num_generations = kwargs.get(
        "num_generations",
        getattr(trainer.args, 'num_generations', 1)
    )

    print(f"🔍 보상 함수 평가 중... (총 {len(completions)}개 응답)")

    for i, completion in enumerate(completions):
        """
        GRPO의 작동 방식:
        - 각 prompt에 대해 여러 개의 응답을 생성 (예: 4개)
        - completion[0:4]는 prompt[0]의 응답
        - completion[4:8]는 prompt[1]의 응답
        - 따라서 label_idx = i // num_generations
        """

        # 이 응답의 라벨 인덱스 계산
        label_idx = i // num_generations

        # 라벨이 없으면 -1.0 (최악)
        if label_idx >= len(labels):
            rewards.append(-1.0)
            continue

        # 예상되는 도구명 가져오기 및 정규화
        label = labels[label_idx]
        expected_tool = label.lower().strip()

        # ============ 단계 1: 도구 호출 추출 ============
        # XML 태그 안의 JSON을 정규표현식으로 찾기
        # 패턴: <tool_call> { ... } </tool_call>
        # re.DOTALL: . 가 줄바꿈도 매치하도록 설정
        tool_match = re.search(
            r'<tool_call>\s*(\{.*?\})\s*</tool_call>',
            completion,
            re.DOTALL
        )

        # 도구 호출을 찾지 못한 경우
        if not tool_match:
            # 모델이 도구를 호출하지 않음 → 가장 나쁜 상황
            rewards.append(-1.0)
            continue

        # 정규표현식 그룹 1에서 JSON 문자열 추출
        tool_json_str = tool_match.group(1)

        # ============ 단계 2: JSON 파싱 시도 ============
        try:
            tool_call = json.loads(tool_json_str)
        except json.JSONDecodeError:
            # JSON 파싱 실패 - 하지만 도구명이 보이는지는 확인
            # 예: {"name" get_ticket} (따옴표 빠짐)
            if expected_tool in tool_json_str.lower():
                # 도구명은 맞지만 형식이 잘못됨
                rewards.append(0.2)
            else:
                # 잘못된 형식 + 잘못된 도구명
                rewards.append(-0.5)
            continue

        # ============ 단계 3: 도구명 검증 ============
        # JSON에서 'name' 필드 추출
        tool_name = tool_call.get('name', '').lower().strip()

        # 도구명 비교 (정확한 매칭 또는 부분 매칭 허용)
        # 예: "get_ticket_details"와 "get_ticket" 중 어느 것을 원했는지
        tool_name_correct = (
                expected_tool in tool_name or      # "get_ticket" in "get_ticket_details"
                tool_name in expected_tool or      # "get_ticket" in "get_ticket_details"
                tool_name == expected_tool         # 정확히 일치
        )

        # 도구명이 틀린 경우
        if not tool_name_correct:
            # 유효한 JSON이지만 잘못된 도구
            # 예: create_ticket 대신 delete_ticket 호출
            rewards.append(-0.3)
            continue

        # ============ 단계 4: 매개변수 검증 (선택 사항) ============
        # 올바른 도구명 + 유효한 JSON 상태
        # 이제 필요한 매개변수가 모두 들어있는지 확인

        if required_params and label_idx < len(required_params):
            # 필수 매개변수 목록 가져오기
            required = required_params[label_idx]

            # JSON에서 매개변수 추출
            # 주의: 필드명이 'parameters' 또는 'arguments'일 수 있음
            provided_params = tool_call.get('parameters', tool_call.get('arguments', {}))

            # 매개변수가 딕셔너리인지 확인
            if isinstance(provided_params, dict):
                # 모든 필수 매개변수가 있고, None/빈 문자열이 아닌지 확인
                has_all_required = all(
                    param in provided_params and
                    provided_params[param] not in [None, '', []]
                    for param in required
                )

                if has_all_required:
                    # 완벽: 올바른 도구 + 유효한 JSON + 모든 매개변수
                    rewards.append(1.0)
                else:
                    # 부분 점수: 올바른 도구이지만 일부 매개변수 누락
                    # 예: ticket_id는 있지만 status는 없음
                    rewards.append(0.5)
            else:
                # 매개변수 형식이 잘못됨 (문자열이거나 리스트 등)
                rewards.append(0.5)
        else:
            # 매개변수 검증을 하지 않음 → 도구명만 맞으면 1.0
            rewards.append(1.0)

    print(f"✅ 보상 평가 완료: {rewards}")
    return rewards


# ============================================================================
# 3️⃣ 보상 함수 #2: 형식 준수 평가
# ============================================================================

def reward_format_compliance(completions: List[str], **kwargs) -> List[float]:
    """
    형식 준수를 위한 보상 함수

    📋 검사 항목:
    - 올바른 XML 태그 (<tool_call>...</tool_call>)
    - 균형잡힌 중괄호 개수
    - 유효한 JSON 구조
    - "name" 키 존재 확인

    💡 이유:
    모델이 도구를 호출할 때 일관된 형식을 사용하도록 학습
    예: 항상 <tool_call>{...}</tool_call> 형식 사용
    """
    rewards = []

    for completion in completions:
        reward = 0.0

        # ✅ 검사 1: 올바른 XML 태그 존재
        # <tool_call>...</tool_call> 모두 있어야 함
        if '<tool_call>' in completion and '</tool_call>' in completion:
            reward += 0.3

        # ✅ 검사 2: 균형잡힌 중괄호
        # { 개수 == } 개수 (유효한 JSON의 기초)
        if completion.count('{') == completion.count('}'):
            reward += 0.2

        # ✅ 검사 3: 기본 JSON 구조
        # tool_call 태그 안의 내용 추출
        tool_match = re.search(
            r'<tool_call>(.*?)</tool_call>',
            completion,
            re.DOTALL
        )

        if tool_match:
            tool_content = tool_match.group(1).strip()

            # "name" 또는 'name' 키가 따옴표와 함께 있는지 확인
            if '"name"' in tool_content or "'name'" in tool_content:
                reward += 0.3

            # ✅ 검사 4: 파싱 가능한 JSON인지 확인
            try:
                json.loads(tool_content)
                # 파싱 성공! 추가 보상은 없지만 음수 페널티도 없음
            except json.JSONDecodeError:
                # JSON 파싱 실패 - 패널티는 주지 않음
                # (quality 함수에서 이미 처리됨)
                pass

        return rewards  # ⚠️ 주의: 각 completion마다 하나의 점수를 반환해야 함


# ============================================================================
# 4️⃣ 보상 함수 #3: 여러 보상 신호 통합
# ============================================================================

def combined_reward(completions: List[str], **kwargs) -> List[float]:
    """
    여러 보상 신호의 가중치 결합

    🎯 전략:
    - quality_rewards: 도구 호출의 정확성 (90% 가중치)
    - format_rewards: 형식 준수 (10% 가중치)

    💡 왜 이렇게 가중치를 설정했나?
    - 가장 중요한 것: 올바른 도구를 올바르게 호출
    - 그 다음: 일관된 형식 유지

    예시 계산:
    - quality = 1.0 (완벽한 도구 호출)
    - format = 0.8 (약간의 형식 문제)
    - combined = 0.9 * 1.0 + 0.1 * 0.8 = 0.98
    """

    # 품질 점수 계산 (도구 호출의 정확성)
    quality_rewards = reward_tool_call_quality(completions, **kwargs)

    # 형식 점수 계산 (형식 준수)
    format_rewards = reward_format_compliance(completions, **kwargs)

    # 가중치 결합
    # 품질에 90%, 형식에 10% 가중치
    combined = [
        0.9 * q + 0.1 * f  # 형식 가중치 0.1로 수정 (원본의 오류 수정)
        for q, f in zip(quality_rewards, format_rewards)
    ]

    print(f"📊 통합 보상: {combined}")
    return combined


# ============================================================================
# 5️⃣ GRPO(Group Relative Policy Optimization) 설정
# ============================================================================

# GRPO란?
# - Policy Gradient 방법 (강화학습의 한 종류)
# - 여러 응답을 생성한 후 상대적 성능으로 업데이트
# - DPO/RLHF보다 간단하고 참조 모델 불필요
# - "Group Relative": 같은 prompt의 여러 응답들을 비교해서 학습

grpo_config = GRPOConfig(
    # ============ 기본 설정 ============
    output_dir="ch07/fine_tuned_model/qwen-helpdesk-rlvr",

    # ============ 생성 설정 ============
    # GRPO의 핵심: 각 prompt마다 여러 응답 생성
    # - 4개의 응답을 생성해서 상대적으로 비교
    # - 좋은 응답과 나쁜 응답의 차이를 학습
    num_generations=4,

    # ============ 학습 설정 ============
    learning_rate=5e-6,  # 매우 낮은 학습률 (미세 조정)
    per_device_train_batch_size=4,  # 배치 크기
    # 효과적 배치 크기 = 4 (num_generations과는 별개)

    num_train_epochs=3,  # 3번 반복

    # ============ 로깅 설정 ============
    logging_steps=1,  # 매 스텝마다 로그 (학습 과정 확인용)
    save_strategy="epoch",  # 매 에포크마다 모델 저장
    report_to=None,  # Weights&Biases 등 외부 서비스 사용 안 함
)

print(f"✅ GRPO 설정 완료")
print(f"   - 학습률: {grpo_config.learning_rate}")
print(f"   - 배치 크기: {grpo_config.per_device_train_batch_size}")
print(f"   - 생성 개수: {grpo_config.num_generations}")
print(f"   - 에포크: {grpo_config.num_train_epochs}")


# ============================================================================
# 6️⃣ GRPO 트레이너 초기화
# ============================================================================

# GRPO 트레이닝 워크플로우:
# 1. prompt를 모델에 입력
# 2. 4개의 응답 생성 (num_generations=4)
# 3. 각 응답에 대해 combined_reward 함수로 보상 계산
# 4. 보상이 높은 응답으로 모델 업데이트
# 5. 반복

trainer = GRPOTrainer(
    model=GRPO_MODEL,  # 모델명 또는 모델 객체
    # - 자동으로 로드됨

    reward_funcs=combined_reward,  # 보상 함수
    # - 앞에서 정의한 combined_reward 함수
    # - 모델의 응답이 얼마나 좋은지 점수 부여

    train_dataset=dataset,  # 훈련 데이터
    # - prompt, chosen, rejected, label 포함

    args=grpo_config,  # 훈련 설정
)

print(f"✅ GRPO 트레이너 초기화 완료")
print(f"   - 모델: {GRPO_MODEL}")
print(f"   - 훈련 장치: {trainer.model.device}")
print(f"   - 훈련 샘플 수: {len(trainer.train_dataset)}")


# ============================================================================
# 7️⃣ 훈련 실행
# ============================================================================

print(f"\n{'='*60}")
print(f"🚀 GRPO 훈련 시작!")
print(f"{'='*60}\n")

print("훈련을 시작합니다...")
print(f"예상 시간: {len(dataset) * 3 * 4 / 4} 스텝")
print(f"(샘플수 × 에포크 × 생성수 / 배치크기)\n")

try:
    # 훈련 실행
    # - 손실값, 보상 등이 자동으로 로깅됨
    # - 매 에포크마다 모델 저장
    trainer.train()

    print(f"\n{'='*60}")
    print(f"✅ 훈련 완료!")
    print(f"{'='*60}\n")

except KeyboardInterrupt:
    print(f"\n⚠️  훈련이 사용자에 의해 중단되었습니다.")

except Exception as e:
    print(f"\n❌ 훈련 중 오류 발생: {e}")
    raise


# ============================================================================
# 8️⃣ 모델 저장 및 완료
# ============================================================================

print(f"✅ 훈련 완료! 모델이 저장되었습니다:")
print(f"   - 저장 위치: {grpo_config.output_dir}")
print(f"   - 파일 형식: 완전한 모델 (LoRA 어댑터 아님)")

print(f"\n💡 다음 단계: 저장된 모델 사용하기")
print(f"""
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "{grpo_config.output_dir}"
)
tokenizer = AutoTokenizer.from_pretrained(
    "{grpo_config.output_dir}"
)

# 도구 호출 추론
prompt = "ticket #12345의 상세 정보를 가져와"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0])
print(response)
""")

print(f"\n📊 훈련 통계:")
print(f"   - 총 샘플: {len(dataset)}")
print(f"   - 에포크: {grpo_config.num_train_epochs}")
print(f"   - 생성 개수: {grpo_config.num_generations}")
print(f"   - 총 업데이트 횟수: {len(dataset) * grpo_config.num_train_epochs}")