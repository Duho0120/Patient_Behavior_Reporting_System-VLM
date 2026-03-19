# 📘 Phase 2: QLoRA 파인튜닝 — Colab 진행 기록 & 공부 포인트
> 작성일: 2026-03-07

---

## ✅ Colab에서 진행한 순서

### 1. 환경 설치
필요한 라이브러리를 설치했다.

```
transformers  → 모델 로드/학습
peft          → LoRA 적용
bitsandbytes  → 4-bit 양자화
trl           → Trainer 도구
accelerate    → GPU 분산 학습 지원
```

**트러블**: `bitsandbytes`와 `peft` 버전 충돌 발생  
→ `bitsandbytes>=0.44.0`, `peft==0.9.0`으로 고정 해결

---

### 2. Google Drive 마운트 & 데이터 경로 설정
Colab에서 Drive를 `/content/drive`에 마운트하고 경로 변수를 설정했다.

```
DRIVE_BASE   = '/content/drive/MyDrive/fall_dataset'
IMAGE_ZIP    = 이미지 zip 파일 경로
LABEL_DIR    = JSON 라벨 폴더 경로
IMAGE_DIR    = '/content/images'  (압축 해제 위치)
OUTPUT_DIR   = '/content/output'  (학습 결과 저장 위치)
```

---

### 3. 이미지 ZIP 압축 해제
`zipfile` 라이브러리로 Drive에 있는 zip을 `/content/images`에 압축 해제했다.

---

### 4. 데이터 전처리 (JSON → 학습 포맷 변환)

**실제 JSON 구조:**
```json
{
  "metadata": { "file_name": "00004_H_A_SY_C3_I005.JPG" },
  "vlm_labels": {
    "status": "moving",
    "captions": ["캡션1", "캡션2", "캡션3", "캡션4", "캡션5"]
  }
}
```

**변환 결과 (학습 포맷):**
```json
{
  "image": "00004_H_A_SY_C3_I005.JPG",
  "conversations": [
    { "from": "human", "value": "현재 환자의 상태를 보고하세요." },
    { "from": "gpt",   "value": "캡션1" }
  ]
}
```

- 이미지 1장 × 5캡션 × 5질문 → **샘플 5개씩 생성**
- 총 학습 샘플: 약 **11,600개** (Train 90% / Val 10%)
- 이미지 파일이 없으면 skip → 실제 학습 샘플: **1,845개** (일부 미매칭)

---

### 5. MobileVLM V2 모델 로드 (4-bit 양자화)

**트러블**: `AutoModelForCausalLM`이 `mobilevlm` 모델 타입을 인식 못 함  
→ MobileVLM GitHub 레포를 클론하고 `sys.path`에 추가 후 모델 클래스를 수동 등록하여 해결

```python
# 핵심 설정
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)
```

---

### 6. LoRA 설정 적용

```python
LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", ...],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
```

`prepare_model_for_kbit_training(model)` → 4-bit 모델을 학습 가능 상태로 준비

---

### 7. Dataset 클래스 & Trainer 설정

- `PatientBehaviorDataset`: JSON 로딩 → 토크나이징 → `input_ids`, `attention_mask`, `labels` 반환
- `labels`에서 질문 부분은 `-100`으로 마스킹 → 답변 부분만 손실 계산

**트러블**: `SFTTrainer` 사용 시 `formatting_func` 미설정 에러  
→ 커스텀 Dataset에는 표준 `Trainer`가 더 적합 → 교체 해결

**학습 설정:**
```
batch_size × gradient_accumulation = 4 × 4 = 16
learning_rate = 2e-4
epochs = 3
optimizer = paged_adamw_8bit
fp16 = True
```

---

### 8. 학습 실행 & 결과
```
총 steps: 345 (1,845샘플 ÷ 16배치 × 3epoch)
소요 시간: 약 16분 (A100)
speed: 0.35 it/s
경고: use_reentrant warning → 에러 아님, 무시 가능
```

---

### 9. 모델 저장 (Drive 백업)

```
/content/output/best_model/ → Drive 복사
저장 파일:
  adapter_config.json        ← LoRA 설정 (r, alpha 등)
  adapter_model.safetensors  ← 학습된 가중치
  tokenizer.json             ← 토크나이저
  tokenizer_config.json
  tokenizer.model
  special_tokens_map.json
```

---

## 📌 내가 공부해서 설명할 수 있어야 하는 것들

| 주제 | 설명할 수 있어야 하는 내용 |
|---|---|
| **LoRA 원리** | `W' = W + A×B` 에서 왜 A, B만 학습하는지 / rank(r)의 의미 |
| **4-bit 양자화** | 왜 float32 → 4bit로 줄이는지 / nf4가 뭔지 |
| **prepare_model_for_kbit_training** | 양자화된 모델을 왜 이 함수로 준비해야 하는지 |
| **labels 마스킹 (-100)** | 왜 질문 부분은 손실 계산에서 빼는지 |
| **gradient accumulation** | 배치 크기를 늘리지 않고 왜 이걸 쓰는지 |
| **LoRA adapter 저장** | base 모델 없이 adapter만 저장하면 어떻게 나중에 합치는지 |
| **Teacher-Student 구조** | Gemini가 Teacher이고 MobileVLM이 Student인 이유 |

---

## 1. 이 단계에서 한 일 (한눈에 보기)

```
[Google Drive]                    [Google Colab A100]
  images.zip    ──→  압축 해제  ──→
  labels_vlm/   ──→  데이터 변환 ──→  학습 데이터  ──→  QLoRA 파인튜닝  ──→  LoRA Adapter 저장
  (JSON 2,320개)      (11,600 샘플)                    (MobileVLM V2)       (Drive 백업)
```

**핵심 한 줄 요약:**  
_"Gemini가 만든 라벨로 MobileVLM을 파인튜닝해서, 병원 CCTV 이미지를 보고 환자 상태를 설명하는 모델을 만든다."_

---

## 2. 핵심 개념 (PPT에 담을 내용)

### 2-1. 왜 파인튜닝인가?

| 방법 | 설명 | 이 프로젝트에서의 한계 |
|---|---|---|
| **처음부터 학습** | 수백만 장 데이터 + 수개월 + 비용 천만원↑ | ❌ 불가능 |
| **파인튜닝** | 이미 학습된 모델을 내 데이터로 조금 더 학습 | ✅ 채택 |

> 💡 **비유**: 영어를 이미 잘하는 사람에게 "의학 용어"만 추가로 가르치는 것

---

### 2-2. Teacher-Student 구조 (아이디어 포인트!)

```
[Teacher]  Gemini 2.5 Flash (대형 유료 API)
    ↓   "이 이미지에서 환자가 뭐하는지 5가지 표현으로 설명해줘"
    ↓
[라벨] 2,320장 × 5캡션 = 11,600개 학습 데이터 자동 생성

[Student]  MobileVLM V2 1.7B (경량 엣지 모델)
    ↓   Teacher의 지식을 흡수해 파인튜닝
    ↓
[결과]  병원 CCTV 이미지 → "환자가 침대 이탈 중" (자동 설명)
```

**포트폴리오 어필 포인트:**  
대형 모델(Gemini)의 지식을 소형 모델(MobileVLM)로 전달하는  
**Knowledge Distillation** 응용 전략 → 실제 배포 가능한 경량 모델 구현

---

### 2-3. QLoRA란? (면접 단골 질문)

#### ① 일반 파인튜닝의 문제
```
일반 파인튜닝: 모델 전체 파라미터(1.7B개) 학습
→ VRAM 수십 GB 필요, 비용 매우 큰
```

#### ② LoRA (Low-Rank Adaptation)
```
핵심 아이디어: 기존 가중치(W)는 건드리지 말고,
              작은 행렬 2개(A, B)만 추가해서 학습!

W' = W + ΔW = W + A × B
      ↑고정         ↑이것만 학습 (파라미터 0.5% 수준)
```

| | 일반 파인튜닝 | LoRA |
|---|---|---|
| 학습 파라미터 | 1,700,000,000개 | ~8,000,000개 (0.5%) |
| VRAM | 70GB + | 8GB 수준 |
| 속도 | 느림 | 빠름 |
| 성능 손실 | 없음 | 거의 없음 |

#### ③ QLoRA = LoRA + 4-bit 양자화
```
양자화(Quantization): 모델 가중치를 32bit → 4bit로 압축
→ 메모리 추가 절약 (40GB A100에서 충분히 학습 가능)

QLoRA = 4-bit 양자화 + LoRA 학습
      = 최소 VRAM으로 최대 효율
```

---

### 2-4. 학습 데이터 구조

Gemini가 생성한 JSON 라벨을 아래 포맷으로 변환:

```json
{
  "image": "00004_H_A_SY_C3_I005.JPG",
  "conversations": [
    {
      "from": "human",
      "value": "현재 환자의 상태를 보고하세요."
    },
    {
      "from": "gpt",
      "value": "환자가 침대 밖 바닥에서 서서 움직이며 활동 중입니다."
    }
  ]
}
```

**왜 5가지 다른 질문?**  
같은 이미지라도 질문 방식을 다양하게 → 모델이 한 가지 패턴에만 과적합되지 않음  
→ **데이터 증강(Data Augmentation)** 효과

---

## 3. 전체 파이프라인 코드 흐름 요약

### Step 1: 환경 설정
```
핵심 라이브러리:
- transformers: HuggingFace의 모델 로딩/학습 프레임워크
- peft        : LoRA 적용 도구
- bitsandbytes: 4-bit 양자화 도구
- trl         : 파인튜닝 훈련 도구
```

### Step 2: 데이터 준비
```
JSON 2,320개
    → 이미지 파일명 추출
    → 5개 캡션 × 5개 질문 형태로 변환
    → 총 11,600개 학습 샘플 (Train 90% / Val 10%)
```

### Step 3: 모델 로드 (4-bit 양자화)
```python
# 핵심: nf4 (NormalFloat4) 양자화 방식
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",   ← QLoRA 논문에서 제안한 최적 방식
    bnb_4bit_compute_dtype=float16,
    bnb_4bit_use_double_quant=True  ← 2중 압축으로 추가 절약
)
```

### Step 4: LoRA 설정
```python
LoraConfig(
    r=16,          ← 행렬 랭크 (클수록 표현력↑, VRAM↑)
    lora_alpha=32, ← 스케일링 계수 (보통 r의 2배)
    target_modules=["q_proj", "k_proj", "v_proj", ...]  ← Attention 레이어에 적용
)
```

### Step 5: 학습
```
배치 크기: 4(per device) × 4(gradient accumulation) = 실질 16
Epoch: 3회
학습률: 2e-4 (LoRA 파인튜닝 표준값)
소요 시간: ~16분 (A100 기준, 텍스트만)
```

### Step 6: 저장
```
/content/output/best_model/ → Drive 백업
  ├── adapter_model.safetensors  ← 학습 결과물 (~50MB)
  ├── adapter_config.json         ← LoRA 설정
  └── tokenizer.*                 ← 토크나이저
```

---

## 4. 이 프로젝트에서 내가 기여한 것 (포트폴리오 서술 포인트)

| 역할 | 내용 |
|---|---|
| **문제 정의** | 병원 CCTV 환자 상태를 자동으로 기록하는 시스템 필요성 제안 |
| **전체 파이프라인 설계** | Teacher-Student (Gemini→MobileVLM) 방식 아이디어 |
| **데이터 전략** | AI허브 낙상 데이터 필터링 (SY+C3, 2,320장) |
| **라벨링 설계** | 4대 원칙(Fall>Bed_Exit>Moving>Resting), 5캡션 다양성 전략 |
| **모델 선택** | 엣지 배포를 위한 MobileVLM V2 선정 (추론속도 vs 성능 균형) |
| **코드 구현** | AI 어시스턴트와 협업하여 전체 파이프라인 구현 |

---

## 5. 면접/발표에서 나올 수 있는 질문 & 답변

**Q: 왜 MobileVLM을 선택했나요?**
> A: 실제 병원 배포 환경(ASUS 노트북, NPU)에서 1~2초 내 추론이 필요합니다. BLIP-2(8~15초), PaliGemma(3~5초)를 검토했지만 MobileVLM이 ~1초로 유일하게 실시간 요구사항을 충족합니다.

**Q: QLoRA를 선택한 이유는?**
> A: 1.7B 모델을 풀 파인튜닝하면 VRAM 70GB 이상이 필요합니다. QLoRA로 4-bit 양자화 + LoRA를 적용하면 A100(40GB) 한 장으로 충분하고, 학습 파라미터가 0.5%이지만 성능 손실이 거의 없습니다.

**Q: Teacher-Student 방식이란?**
> A: 대형 모델(Gemini 2.5 Flash)이 이미지 설명 라벨을 생성하고(Teacher), 경량 모델(MobileVLM)이 그 라벨로 학습합니다(Student). 대형 모델의 지식을 소형 모델로 전이하는 Knowledge Distillation 전략입니다.

**Q: 데이터가 2,320장으로 적지 않나요?**
> A: 적습니다. 이를 보완하기 위해 이미지 1장당 5가지 서로 다른 질문-답변 쌍을 생성(×5)했고, Gemini가 같은 이미지를 5가지 표현으로 설명하게 해 총 11,600개의 다양한 학습 샘플을 확보했습니다.

---

## 6. 더 공부하면 좋을 것들

| 주제 | 왜 중요한가 | 난이도 |
|---|---|---|
| **LoRA 논문** (Hu et al., 2021) | LoRA의 수학적 원리 이해 | ⭐⭐⭐ |
| **QLoRA 논문** (Dettmers et al., 2023) | nf4 양자화, double quantization 이해 | ⭐⭐⭐ |
| **Instruction Tuning** | 왜 Q&A 형식으로 학습하는지 | ⭐⭐ |
| **PEFT 라이브러리** | LoRA 코드 직접 읽어보기 | ⭐⭐ |
| **VLM 아키텍처** | 이미지 → 언어 모델 연결 원리 | ⭐⭐⭐⭐ |
| **Gradient Checkpointing** | VRAM 절약 기법 | ⭐⭐ |

---

## 7. 다음 단계 (Phase 3~5 예고)

```
Phase 3: 모델 변환
  MobileVLM (PyTorch) → ONNX → OpenVINO IR
  (Intel NPU/ARC GPU에서 돌릴 수 있는 형태로 변환)

Phase 4: C++ 추론 파이프라인
  OpenVINO C++ API로 실시간 이미지 분석 프로그램 구축
  이미지 → "환자 침대 이탈" → 간호사 알림

Phase 5: ASUS 노트북 배포
  NPU + Intel ARC GPU에서 1~2초 추론 실현
  PDF 환자 행동 보고서 자동 생성
```

---

## 8. 트러블슈팅 기록 (실제 경험)

| 에러 | 원인 | 해결 |
|---|---|---|
| `KeyError: 'mobilevlm'` | AutoModel이 커스텀 모델 타입 미인식 | MobileVLM 레포 클론 후 수동 등록 |
| `ImportError: sync_gpu` | bitsandbytes 버전 충돌 | `bitsandbytes>=0.44.0`으로 업그레이드 |
| `SFTTrainer ValueError` | formatting_func 미설정 | SFTTrainer → 표준 Trainer로 교체 |
| `use_reentrant Warning` | PyTorch 2.9 API 변경 예고 | 무시 가능 (에러 아님) |

> 💡 **포트폴리오 팁**: 위 트러블슈팅 경험은 "실제 문제를 직접 해결한 경험"으로 어필 가능!

---

_이 문서는 Phase 2 진행 중 학습한 내용을 기록한 것입니다._  
_Phase 3 (모델 변환) 완료 후 PHASE3_CONVERSION_NOTES.md를 추가할 예정입니다._
