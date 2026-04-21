# fine_tune_helpdesk_dpo.py
# 📚 DPO(Direct Preference Optimization)를 사용한 헬프데스크 모델 파인튜닝
#
# DPO란?
# - SFT(Supervised Fine-Tuning) 이후 모델의 응답을 선호도 기반으로 최적화
# - 더 나은 응답(chosen)과 나쁜 응답(rejected)의 쌍을 사용해 학습
# - PPO(Proximal Policy Optimization)보다 효율적이고 안정적

import logging
import os
import platform
import torch
from datasets import load_dataset
from huggingface_hub import constants as hf_constants
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import DPOConfig, DPOTrainer

# ============================================================================
# 🔧 설정 변수
# ============================================================================

# 기본 모델: Microsoft의 Phi-3 Mini (4K 컨텍스트 윈도우)
# - 가볍고 빠르며 임베딩 성능이 우수한 모델
BASE_SFT_CKPT = "microsoft/Phi-3-mini-4k-instruct"

# DPO 학습 데이터: chosen/rejected 응답 쌍이 포함된 JSON 파일
DPO_DATA = "ch07/training_data/dpo_it_help_desk_training_data.json"

# 파인튜닝된 모델 저장 경로
OUTPUT_DIR = "ch07/fine_tuned_model/phi3-mini-helpdesk-dpo"


# ============================================================================
# 🛠️ 유틸리티 함수
# ============================================================================

def is_model_cached(repo_id: str) -> bool:
    """
    Hugging Face 모델이 로컬 캐시에 있는지 확인하는 함수

    Args:
        repo_id (str): 모델 저장소 ID (예: "microsoft/Phi-3-mini-4k-instruct")

    Returns:
        bool: 캐시되어 있으면 True, 없으면 False

    동작 원리:
        1. 로컬 경로로 존재하는지 확인
        2. Hugging Face 캐시 폴더에 있는지 확인
           - 캐시 폴더: ~/.cache/huggingface/hub/models--microsoft--Phi-3-mini-4k-instruct
    """
    # 로컬 경로에 직접 있는 경우 (예: "./models/phi3")
    if os.path.exists(repo_id) and os.path.isdir(repo_id):
        return True

    # Hugging Face 캐시 폴더에 있는지 확인
    # repo_id "a/b"를 "models--a--b" 형식으로 변환
    cache_folder = "models--" + repo_id.replace("/", "--")
    cache_path = os.path.join(hf_constants.HF_HUB_CACHE, cache_folder)
    return os.path.exists(cache_path)


# ============================================================================
# 📝 로깅 설정
# ============================================================================

logger = logging.getLogger(__name__)

# 모델이 캐시되어 있지 않으면 경고 로깅
if not is_model_cached(BASE_SFT_CKPT):
    logger.warning(
        "로컬 캐시에서 '%s'를 찾을 수 없습니다. "
        "Hugging Face Hub에서 다운로드합니다...",
        BASE_SFT_CKPT
    )


# ============================================================================
# 1️⃣ 토크나이저 로드
# ============================================================================

tok = AutoTokenizer.from_pretrained(
    BASE_SFT_CKPT,
    padding_side="right",  # 문장의 오른쪽에 패딩 추가 (LLM 학습 관례)
    trust_remote_code=True  # 커스텀 코드 실행 허용 (Phi-3에 필요)
)

# 토크나이저가 특별한 토큰을 설정하도록 강제
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


# ============================================================================
# 2️⃣ 량자화(Quantization) 설정
# ============================================================================

# Mac (Darwin)에서는 bitsandbytes가 불안정하므로 4bit 양자화 비활성화
# - bitsandbytes는 주로 Linux/Windows의 NVIDIA GPU용으로 최적화됨
# - Mac은 Metal Performance Shaders(MPS) 사용
USE_4BIT = platform.system() != "Darwin"

if USE_4BIT:
    # ✅ Linux/Windows: 4bit 양자화로 메모리 절약 (원래 33B → ~8GB)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # 모델을 4비트로 로드
        bnb_4bit_use_double_quant=True,  # 더블 양자화 (메모리 추가 절약)
        bnb_4bit_compute_dtype=torch.bfloat16  # 계산은 bfloat16으로 수행 (정확도 유지)
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_SFT_CKPT,
        quantization_config=bnb_config,  # 양자화 설정 적용
        device_map="auto",  # GPU 자동 할당
        trust_remote_code=True
    )
else:
    # ❌ Mac: 양자화 비활성화, bfloat16으로 로드
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_SFT_CKPT,
        device_map="auto",  # CPU 또는 MPS 자동 할당
        torch_dtype=torch.bfloat16,  # 메모리 효율성과 성능 균형
        attn_implementation="eager",  # Mac에서 Flash Attention 미지원이므로 표준 attention 사용
        trust_remote_code=True
    )

print(f"✅ 기본 모델 로드 완료")
print(f"   - 모델: {BASE_SFT_CKPT}")
print(f"   - 숨겨진 차원(hidden_size): {base_model.config.hidden_size}")
print(f"   - 4비트 양자화: {USE_4BIT}")


# ============================================================================
# 3️⃣ LoRA(Low-Rank Adaptation) 설정
# ============================================================================

# LoRA란?
# - 전체 파라미터 대신 일부 행렬에만 저순위 행렬을 추가해 학습
# - 원래 파라미터: W_0 (frozen)
# - 학습되는 부분: W_0 + A × B (A, B는 저순위 행렬)
# - 메모리: 33B 모델 → ~8GB로 감소, 전체 파라미터의 1%만 학습

lora_config = LoraConfig(
    # LoRA 행렬의 순위 (rank)
    # - 값이 작을수록 파라미터 적음, 값이 클수록 표현력 증가
    # - 8: 가볍고 빠름, 충분한 성능 (이 프로젝트에 적합)
    r=8,

    # LoRA 스케일 인수
    # - LoRA 출력을 (lora_alpha / r)로 스케일링
    # - 16: 안정적인 학습률 유지
    lora_alpha=16,

    # LoRA 드롭아웃 (정규화 기법)
    # - LoRA 가중치의 5%를 학습 중 임의로 0으로 설정
    # - 과적합 방지
    lora_dropout=0.05,

    # LoRA를 적용할 대상 모듈들
    # - Transformer의 주요 선형 계층들:
    #   * q_proj, k_proj, v_proj: 멀티헤드 어텐션의 Q, K, V 투영
    #   * o_proj: 어텐션 출력 투영
    #   * gate_proj, up_proj, down_proj: Feed-Forward Network (FFN)
    target_modules=[
        "q_proj",  # Query 투영
        "k_proj",  # Key 투영
        "v_proj",  # Value 투영
        "o_proj",  # Output 투영
        "gate_proj",  # FFN 게이트
        "up_proj",  # FFN Up-Projection
        "down_proj"  # FFN Down-Projection
    ],

    # 바이어스 항 학습 여부
    # - "none": 바이어스 학습 안 함 (메모리 절약, 대부분의 경우 충분)
    bias="none",

    # 작업 타입
    # - "CAUSAL_LM": 인과 언어 모델링 (다음 토큰 예측)
    task_type="CAUSAL_LM",
)

# LoRA 어댑터를 기본 모델에 적용
model = get_peft_model(base_model, lora_config)

# 학습 가능한 파라미터 수 확인
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"\n✅ LoRA 적용 완료")
print(f"   - 학습 가능한 파라미터: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
print(f"   - 전체 파라미터: {total_params:,}")


# ============================================================================
# 4️⃣ 데이터셋 로드
# ============================================================================

# DPO 데이터 형식 예시:
# {
#   "prompt": "IT 헬프데스크 문제: ...",
#   "chosen": "정답(좋은 응답): ...",      <- 선호하는 응답
#   "rejected": "오답(나쁜 응답): ..."     <- 선호하지 않는 응답
# }

ds = load_dataset("json", data_files=DPO_DATA, split="train")

print(f"\n✅ 데이터셋 로드 완료")
print(f"   - 파일: {DPO_DATA}")
print(f"   - 샘플 수: {len(ds)}")
print(f"   - 샘플 형식: {ds.column_names}")


# ============================================================================
# 5️⃣ DPO(Direct Preference Optimization) 학습 설정
# ============================================================================

# DPO 훈련 파라미터 설명:
# - 기존 방식(SFT → RLHF): 먼저 지도학습, 다음 보상 모델 학습, 마지막 PPO 최적화
# - DPO 방식: 선호도 데이터로 직접 최적화 (더 간단하고 효율적)

dpo_config = DPOConfig(
    # ============ 기본 설정 ============
    output_dir=OUTPUT_DIR,  # 모델 저장 위치

    # ============ 배치 크기 및 누적 설정 ============
    # 효과적인 배치 크기 = per_device_train_batch_size × gradient_accumulation_steps
    # 예: 4 × 4 = 16 (메모리 제약이 있을 때 이렇게 분할하여 계산)
    per_device_train_batch_size=4,  # 각 GPU에서 한 번에 처리할 샘플 수
    gradient_accumulation_steps=4,  # 그래디언트를 4번 누적한 후 가중치 업데이트

    # ============ 학습률 및 에포크 ============
    learning_rate=5e-6,  # 매우 낮은 학습률 (미세 조정이므로)
    num_train_epochs=3.0,  # 3번 반복 학습

    # ============ 데이터 타입 설정 ============
    bf16=True,  # bfloat16(Brain Float 16) 사용
    # - 16비트 부동소수점으로 메모리 절약
    # - float32의 약 50% 메모리 사용
    # - 정확도는 거의 손실 없음

    # ============ 로깅 및 저장 설정 ============
    logging_steps=10,  # 10 스텝마다 손실값 출력
    save_strategy="epoch",  # 매 에포크마다 모델 저장
    report_to=None,  # Weights&Biases 등 외부 서비스에 보고하지 않음

    # ============ DPO 특화 파라미터 ============
    beta=0.1,  # DPO 정규화 계수
    # - 선호도 손실과 KL 발산의 가중치 조절
    # - 작을수록: 선호도 학습에 더 중점
    # - 클수록: KL 발산 제약을 더 강하게 (원래 모델과의 거리 제한)

    loss_type="sigmoid",  # 손실 함수 타입
    # - "sigmoid": log sigmoid 손실 (DPO 기본값)
    # - "hinge": 힌지 손실
    # - "ipo": Implicit Preference Optimization

    label_smoothing=0.0,  # 라벨 스무딩 (미사용)
    # - 과적합 방지 기법 (0 = 사용 안 함)

    # ============ 길이 제한 ============
    max_prompt_length=4096,  # 프롬프트 최대 길이 (토큰)
    max_completion_length=4096,  # 응답 최대 길이 (토큰)
    max_length=8192,  # 전체 최대 길이 (프롬프트 + 응답)

    # ============ 패딩 및 절단 ============
    label_pad_token_id=-100,  # 패딩된 부분의 손실 계산에서 무시
    # - -100: PyTorch의 표준 무시 인덱스 (손실 계산에서 제외)

    truncation_mode="keep_end",  # 길이 초과 시 끝 부분 유지
    # - "keep_end": 최신 토큰 유지 (프롬프트 → 응답 순서에서 응답 우선)
    # - "keep_start": 초기 토큰 유지

    # ============ 평가 및 고급 설정 ============
    generate_during_eval=False,  # 평가 중 생성 안 함 (속도 향상)
    disable_dropout=False,  # 드롭아웃 활성화 (정규화)

    reference_free=True,  # 참조 모델 없이 학습
    # - DPO의 장점: 별도의 참조 모델 필요 없음
    # - RLHF는 보상 모델 + 참조 모델 필요 (메모리 3배)

    model_init_kwargs=None,  # 모델 초기화 파라미터 (사용 안 함)
    ref_model_init_kwargs=None,  # 참조 모델 파라미터 (미사용)
)

print(f"\n✅ DPO 학습 설정 완료")
print(f"   - 효과적 배치 크기: {dpo_config.per_device_train_batch_size * dpo_config.gradient_accumulation_steps}")
print(f"   - 학습률: {dpo_config.learning_rate}")
print(f"   - 에포크: {dpo_config.num_train_epochs}")


# ============================================================================
# 6️⃣ DPO 트레이너 초기화
# ============================================================================

trainer = DPOTrainer(
    model=model,  # LoRA가 적용된 Phi-3 모델

    ref_model=None,  # 참조 모델 불필요 (reference_free=True이므로)
    # - RLHF와 달리 DPO는 같은 모델 내에서 선호도 최적화

    args=dpo_config,  # 위에서 설정한 모든 학습 파라미터

    train_dataset=ds,  # 선호도 데이터셋 (chosen/rejected 쌍)

    processing_class=tok,  # 토크나이저 (padding_side="right" 설정 반영)
    # - 데이터 전처리 시 이 토크나이저 사용
)

print(f"\n✅ DPO 트레이너 초기화 완료")
print(f"   - 학습 샘플 수: {len(trainer.train_dataset)}")


# ============================================================================
# 7️⃣ 모델 학습 실행
# ============================================================================

print(f"\n{'='*60}")
print(f"🚀 DPO 학습 시작!")
print(f"{'='*60}\n")

try:
    # 학습 실행 (손실값, 정확도 등이 자동으로 출력됨)
    trainer.train()
    print(f"\n{'='*60}")
    print(f"✅ 학습 완료!")
    print(f"{'='*60}\n")

except KeyboardInterrupt:
    # 사용자가 Ctrl+C로 중단하면
    print(f"\n⚠️  학습이 사용자에 의해 중단되었습니다.")

except Exception as e:
    # 기타 에러 발생 시
    print(f"\n❌ 학습 중 오류 발생: {e}")
    raise


# ============================================================================
# 8️⃣ 모델 저장
# ============================================================================

# 학습된 모델 저장
trainer.save_model()

# 토크나이저도 함께 저장 (추론 시 필요)
tok.save_pretrained(OUTPUT_DIR)

print(f"\n✅ 모델 및 토크나이저 저장 완료!")
print(f"   - 저장 위치: {OUTPUT_DIR}")
print(f"\n📋 저장된 파일들:")
print(f"   - adapter_config.json: LoRA 설정")
print(f"   - adapter_model.bin: 학습된 LoRA 가중치")
print(f"   - tokenizer.model: 토크나이저")
print(f"   - special_tokens_map.json: 특수 토큰 정의")

print(f"\n💡 다음 단계: 저장된 모델로 추론하기")
print(f"""
from peft import AutoPeftModelForCausalLM
model = AutoPeftModelForCausalLM.from_pretrained("{OUTPUT_DIR}")
tokenizer = AutoTokenizer.from_pretrained("{OUTPUT_DIR}")
# 이제 모델을 사용할 준비 완료!
""")