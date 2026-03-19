# 📋 PROJECT PLAN: 스마트병동 환자안전 모니터링 시스템

> 이 문서는 프로젝트의 핵심 결정사항과 방향성을 기록합니다.
> 새 대화 시작 시 이 파일을 읽고 맥락을 파악해 주세요.

---

## 1. 프로젝트 목표

**VLM 기반 실시간 환자 행동 기록 시스템** — 병원 CCTV 영상에서 환자의 행동 변화를 감지하고, 자연어 보고서를 자동 생성한다.

### 핵심 컨셉

- VLM이 2~3초 간격으로 이미지를 분석하여 **짧은 상태 문장** 생성
- **행동 변화가 감지되면** (예: 휴식→이탈) 해당 시점을 보고서에 기록
- 낙상 감지 시 즉시 간호사 호출

### 예시 보고서

```
10:00 환자 침대에서 휴식중
10:37 환자 침대 이탈, 화장실 방향으로 이동
10:45 환자 침대에서 휴식중
12:46 환자 낙상 발생 → 간호사 호출
```

---

## 2. 모델 결정

### 최종 모델: MobileVLM v2 1.4B

| 항목 | 내용 |
|---|---|
| 모델 | MobileVLM v2 1.4B |
| 학습 방식 | QLoRA (4-bit) |
| 학습 환경 | Google Colab (A100/H100) |
| 배포 환경 | ASUS 17s 노트북 (NPU + Intel ARC) |
| 추론 속도 | ~1~2초 (OpenVINO + NPU) |
| VRAM | ~1GB (INT4) |

### 배포 환경

- **장비**: ASUS 17s 노트북 (Jetson Orin Nano는 사용 불가)
- **NPU 보유**: Intel AI Boost NPU → OpenVINO 최적화 가능
- **GPU**: Intel ARC 내장 그래픽
- **최적화**: OpenVINO + NPU/GPU 가속

### 모델 선택 이유 (의사결정 이력)

1. **BLIP-2 6.7B** → 엣지에서 8~15초 추론, 메모리 부족 → ❌ 기각
2. **PaliGemma 3B** → 가능하지만 3~5초로 빠듯 → 후보
3. **MobileVLM 1.4B** → 1~2초 추론, 1GB VRAM, 엣지 최적화 설계 → ✅ 채택

### 왜 분류 모델이 아닌 VLM인가?

- 분류 모델: "class 1" (고정 라벨) → 방향, 자세 등 세부 정보 불가
- VLM: "환자 침대 이탈, 화장실 방향 이동" → 자연어로 상황 설명 가능

---

## 3. 전체 파이프라인

```
Phase 1: 데이터 구축 (Gemini API 라벨링)        ← ✅ 1차 완료
Phase 2: QLoRA 파인튜닝 (MobileVLM 1.4B)        ← ✅ 1차 완료
Phase 2.5: 데이터 재라벨링 (Gemini 2.5 Pro)      ← ✅ 라벨링 완료, 검수 중
Phase 3: 모델 변환 (PyTorch → ONNX → OpenVINO)  ← 대기 (2차 파인튜닝 완료 후)
Phase 4: C++ 실시간 추론 파이프라인 구축
Phase 5: ASUS 17s (NPU + Intel ARC) 배포 + 보고서 자동화
```

### Phase 1: 데이터 구축 ✅ **완료**

- [x] AI허브 라벨링 JSON 다운로드 (217,536개)
- [x] 필터링: SY(측면낙상) + 병원(H_A,H_D,H_B) + C3 카메라
- [x] 이미지/라벨 Google Drive 업로드
- [x] Gemini 2.5 Flash REST API 연동 완료
- [x] SY+C3 2,320장 × 5캡션 자동 라벨링 (`labels_vlm/` 2,320개 JSON)

### Phase 2: QLoRA 파인튜닝 ✅ **완료**

- [x] MobileVLM 1.4B + QLoRA (Colab A100, 약 16분)
- [x] 대화형 학습 데이터 포맷 (instruction tuning, 학습 샘플 1,845개)
- [x] 학습/검증 분리 (Train 90% / Val 10%), 3 epoch 학습
- [x] LoRA 어댑터 저장 (`mobilevlm_lora_adapter.zip`, ~57MB)

### Phase 2.5: 데이터 재라벨링 & 2차 파인튜닝 ← **현재 단계**

- **배경 및 필요성 (1차 시도의 교훈)**: 
  - 1차 파인튜닝 시 **COCO 데이터셋 스타일(서술형)**을 채택하여 이미지당 5개의 다양하고 긴 정답(Label) 문장을 구성함.
  - 이로 인해 언어 구사력이 부족한 1.4B 소형 모델이 **지나치게 말을 길게 지어내려는 환각(Hallucination)** 증세를 보임.
  - 여러 개의 긴 서술형 추론 결과는 목표로 했던 **"1줄 형태의 단답형 보고서(예: `환자 낙상 발생`)" 형식과 구조적으로 상극**이었음.
  - 쓸데없이 긴 문장을 생성하느라 온디바이스 환경(CPU/NPU)에서 심각한 추론 속도 저하(FPS 병목 현상)가 유발됨.
  - 명령어(Prompt Engineering) 강제만으로는 소형 모델의 행동 제어에 한계가 있음을 확인.
- **해결 전략 (온디바이스 최적화)**:
  - 출력 결과를 극도로 짧고 명확하게 제한(예: `10:00 환자 휴식 중`, `12:46 침대 이탈`)하여 생성 속도를 극대화(FPS 향상)하고 환각을 차단.
  - COCO 스타일의 다중 묘사를 버리고, 목표로 하는 "예시 보고서" 형식과 100% 동일한 **4가지 단답형 정답(객관식)** 텍스트로 모델을 새롭게 세뇌(2차 파인튜닝).
  - V2 라벨링의 4가지 고정 상태: `환자 휴식 중`, `환자 이동 중`, `환자 침대 이탈`, `환자 낙상 발생`
- [x] Gemini 2.5 Pro API 연동 및 단답형(고정 4지선다) 프롬프트 설계 완료
- [x] v2 파인튜닝 데이터셋(JSON) 재구축 완료 (2,320개 이미지 라벨링 생성)
- [/] 전용 GUI 도구(`scripts/label_reviewer.py`)를 이용한 라벨 전수 검사 및 교정 (진행 중)
- [ ] MobileVLM 2차 QLoRA 파인튜닝 진행

### Phase 3: 모델 경량화 & 변환 (보류 및 설계 변경)

- **왜 1차 시도에서 NPU를 쓰지 못하고 CPU로 폴백(Fallback) 되었는가? (한계점 파악)**
  - **1) 동적 입력 크기(Dynamic Shape) 미지원:** NPU(Intel AI Boost) 하드웨어 특성상 입력과 출력의 크기가 완벽히 고정된(Static Shape) 행렬 연산만 고속으로 처리 가능함. 그러나 VLM 텍스트 생성 특성상 토큰 길이가 계속 변하므로 오픈비노(OpenVINO) 컴파일 단계에서 호환성 에러(`RuntimeError`)가 발생함.
  - **2) 커스텀 모델 구조의 충돌:** LLaMA 구조에 `CLIPVisionTower`와 `LDPNetV2`(프로젝터)가 결합된 MobileVLM 특성상, 모델을 통째로 OpenVINO IR로 한 번에 변환하려 하면 내부 그래프 로직이 꼬임.
- **V2 추론 파이프라인 최적화 목표: 하이브리드(Heterogeneous) 분배 전략**
  - **이론적 한계의 인정:** NPU는 연산 속도는 준수하나, 메모리 대역폭이 좁고 언어 모델의 핵심인 **KV Cache 지원이 빈약**하여 텍스트 생성부(LLoMA)를 올렸을 때 오히려 성능(FPS)과 지능이 저하될 위험성이 존재함. 양자화(INT8)로 인해 언어 생성 문맥이 무너질 우려도 있음.
  - **하이브리드 분업 전략 (가장 현실적인 대안):**
    - [ ] **눈(Vision Encoder)은 NPU로:** 가장 고정된 행렬 연산을 요구하는 `CLIP Vision Encoder` 파트만 분할 추출 및 정적 패딩(Static Padding)을 적용하여 NPU(INT8)에 할당.
    - [ ] **입(Language Decoder)은 GPU/CPU로:** 캐시 관리가 복잡하고 동적으로 길이가 변하는 텍스트 생성 파트는 메모리 대역폭이 넓은 **Intel ARC GPU** 또는 **FP16/BF16 CPU**에 할당.
    - [ ] **파이프라이닝 구축:** 파이썬(추후 C++) 환경에서 NPU가 추출한 이미지 임베딩 결과값을 GPU/CPU의 언어 모델 입력으로 직렬 전송하여 최종 추론 속도 및 지능 손실 최소화.

### Phase 4: C++ 추론 파이프라인

- [ ] OpenVINO C++ API 또는 ONNX Runtime
- [ ] 실시간 추론 프로그램 구축
- [ ] 상태 변화 감지 로직

### Phase 5: 배포 & 보고서

- [ ] Jetson Orin Nano 적용 목표지만 현재 보유하지 않음 -> 보유중인 노트북 활용
- [ ] ASUS 17s 노트북 배포 (NPU + Intel ARC)
- [ ] PDF 환자 행동 보고서 자동 생성

---

## 4. 데이터 정보

### 데이터 출처

- **AI허브**: [낙상사고 위험동작 영상-센서 쌍 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71641) (datasetkey: 71641)
- **이전 프로젝트**: 딥러닝 프로젝트 팀원 데이터 (이미지 원본)

### 데이터 규모

| 구분 | 시나리오 | 이미지 | 비고 |
|---|---|---|---|
| SY(측면낙상)+C3 | 232개 | 2,320장 | ← 라벨링 대상 |
| N(정상)+C3 | 284개 | 2,840장 | 필요시 추가 |

### 데이터 경로

| 위치 | 경로 |
|---|---|
| 이미지 원본 (로컬) | `C:\Users\ASUS\Desktop\제로베이스\딥러닝 프로젝트\낙상사고 위험동작 영상-센서 쌍 데이터_병원,후면낙상\3.개방데이터\1.데이터\Training\01.원천데이터\TS\이미지\Y\SY` |
| 라벨 원본 (로컬) | `C:\Users\ASUS\Desktop\제로베이스\Patient_Behavior_Reporting_System-VLM\data\v2\labels_vlm` |
| Google Drive 이미지 | `G:\내 드라이브\fall_dataset\images\` |
| Google Drive 라벨 | `G:\내 드라이브\fall_dataset\labels\` |
| Colab 마운트 | `/content/drive/MyDrive/fall_dataset/` |

### 시나리오 구조 (10장/시나리오)

```
I001~I003: 침대 이탈 (bed_exit)
I004~I007: 이동 중 (moving)
I008~I010: 낙상 (fall)
```

---

## 5. 라벨링 설계

### Teacher-Student 방식

```
[Teacher] Gemini 2.5 Flash API → 5개 캡션 생성
[Student] MobileVLM 1.4B → QLoRA 파인튜닝
```

### 라벨링 4대 원칙 (Golden Rules)

| 우선순위 | 규칙 | 조건 | 라벨링 |
|---|---|---|---|
| 1 | Fall | 바닥 접촉 | "낙상 발생" |
| 2 | Bed_Exit | 침대 + 이탈 | "침대 이탈" |
| 3 | Moving | 서서 걷기 | "이동 중" |
| 4 | Resting | 침대에 누움 | "휴식 중" |

### COCO 스타일 5개 캡션

1장의 이미지에 **5가지 다른 표현**으로 라벨링 (표현 다양성 → 과적합 방지)

### 학습 데이터 포맷 (MobileVLM instruction tuning)

```json
{
  "image": "00004_H_A_SY_C3_I005.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\n현재 환자의 상태를 보고하세요."},
    {"from": "gpt", "value": "환자가 침대에서 이탈하여 바닥 매트 위에서 몸을 숙이고 있습니다."}
  ]
}
```

### Gemini API 호출 정보 (Phase 2.5 기준)

- **API**: Gemini 2.5 Pro (고해상도 이미지 분석 및 정교한 라벨링 목적)
- **API Key**: `API_KEY_to_Labelling.env` 파일에 저장
- **대상 규모**: SY 2,320장 (이미지당 1회 호출)
- **특이사항**: 429(Quota), 503(Server Demand) 에러를 자동 재시도로 처리하며 안정적 라벨링 수행

---

## 5-2. Phase 2 파인튜닝 결과물

### 학습 결과 요약

| 항목 | 내용 |
|---|---|
| 학습 샘플 | 1,845개 (이미지 미매칭 skip, 이론상 11,600개 중) |
| Train / Val | 90% / 10% |
| Epochs | 3 |
| Steps | 345 (1,845 ÷ 16 × 3) |
| 소요 시간 | 약 16분 (Colab A100) |
| Batch size | 4 + gradient_accumulation 4 → 실질 16 |
| Learning rate | 2e-4 (paged_adamw_8bit) |
| LoRA r / alpha | 16 / 32 |

### 로컬 저장 파일

| 파일 | 설명 |
|---|---|
| `mobilevlm_lora_adapter.zip` | LoRA 어댑터 전체 (~57MB) |
| `training_logs.csv` | step별 학습 손실 기록 |
| `loss_curve.png` | 학습 손실 시각화 |
| `labels_vlm/` | Gemini 라벨링 결과 JSON 2,320개 |
| `labels_vlm.zip` | 위 폴더 압축본 |

### LoRA 어댑터 구성 (`mobilevlm_lora_adapter/`)

```
adapter_config.json         ← LoRA 설정 (r=16, alpha=32)
adapter_model.safetensors   ← 학습된 가중치 (핵심)
tokenizer.json
tokenizer_config.json
tokenizer.model
special_tokens_map.json
```

> 자세한 파인튜닝 과정: [PHASE2_FINETUNING_NOTES.md](PHASE2_FINETUNING_NOTES.md)

---

## 6. 기술 포트폴리오 어필 포인트

| 역량 | 상세 |
|---|---|
| VLM 파인튜닝 | Teacher-Student (Gemini→MobileVLM) + QLoRA |
| 모델 경량화 | MobileVLM 1.4B, INT4 양자화 |
| 모델 변환 | PyTorch → ONNX → OpenVINO |
| C++ 엔지니어링 | ONNX Runtime / OpenVINO C++ API 추론 |
| 엣지 배포 | ASUS 17s (NPU + Intel ARC) + OpenVINO 실시간 추론 |
| 데이터 파이프라인 | AI허브 API 연동, 자동 라벨링 |

---

## 7. 참고사항

- Python 환경: `ds_study` conda env (Python 3.8) → REST API 사용
- Git: `https://github.com/Duho0120/Patient_Behavior_Reporting_System-VLM.git`
- 이전 프로젝트: YOLO 기반 낙상 감지 (팀 프로젝트, 딥러닝 프로젝트 폴더)
