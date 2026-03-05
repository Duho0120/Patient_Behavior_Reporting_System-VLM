# 🏥 스마트병동 환자안전 모니터링 시스템 (Patient Behavior Reporting System)

거대 시각-언어 모델(VLM)을 활용하여 병원 내 환자의 이상 행동(낙상, 침대 이탈 등)을 실시간으로 감지하고, 환자 행동 기록 보고서를 자동 생성하는 **End-to-End 솔루션**입니다.

## 기술 스택

| 구분 | 기술 | 도입 이유 |
|---|---|---|
| 모델 | MobileVLM v2 1.4B | 경량 VLM, ASUS 17s 노트북 배포 가능 (1~2초 추론) |
| 학습 | QLoRA (4-bit) | 대형 모델을 효율적으로 학습, 메모리 절감 |
| 인프라 | Colab Pro (A100) | 대형 모델 고속 학습 |
| 라벨링 | Gemini 2.5 Flash API | Teacher-Student 방식, 이미지당 5개 캡션 자동 생성 |
| 배포 | ASUS 17s + OpenVINO | NPU + Intel ARC GPU 가속 실시간 추론 |

## 시스템 아키텍처

```
[카메라] → 2~3초 간격 캡처 → [MobileVLM] → 상태 분류 + 설명
                                              ↓
                              이전 상태와 비교 → 변화 시 기록
                                              ↓
                              [환자 행동 기록 보고서] → 간호사 전달
```

### 예시 보고서 출력

```
10:00 환자 침대에서 휴식중
10:37 환자 침대 이탈, 화장실 방향으로 이동
10:45 환자 침대에서 휴식중
12:46 환자 낙상 발생 → 간호사 호출
```

## 프로젝트 진행 단계

### ✅ Phase 1: 고품질 학습 데이터 구축 (완료)

- [x] AI허브 API 연동 및 데이터셋 구조 파악
- [x] 라벨링 데이터(JSON) 다운로드 및 분석 (217,536개 파일)
- [x] 데이터 필터링: **측면낙상(SY) + 병원(H_A, H_D, H_B) + C3 카메라 → 2,320장**
- [x] 이미지 원천 데이터 확보 → Google Drive 업로드 완료
- [x] Gemini 2.5 Flash API 연동 (REST API, Python 3.8 호환)
- [x] **COCO 스타일 5개 캡션 자동 라벨링 완료 (2,320장 × 5캡션 = 11,600개 샘플)**
- [x] 라벨링 결과 로컬(`labels_vlm/`) + Google Drive 저장 완료

### 🔄 Phase 2: QLoRA 기반 파인튜닝 ← **현재 단계**

- [ ] 학습 포맷 변환: `labels_vlm` JSON → LLaVA instruction tuning 포맷
- [ ] MobileVLM 1.4B + QLoRA 어댑터 적용 (Colab A100)
- [ ] 한국어 환자 상태 보고 문체 학습
- [ ] 학습/검증 분리 및 성능 평가

### Phase 3: 모델 경량화 & 변환

- [ ] PyTorch → ONNX 변환
- [ ] ONNX → OpenVINO IR 변환 (Intel NPU/ARC 최적화)
- [ ] 추론 속도 벤치마크

### Phase 4: C++ 실시간 추론 파이프라인

- [ ] OpenVINO C++ API 또는 ONNX Runtime
- [ ] 실시간 상태 변화 감지 로직

### Phase 5: 배포 & 보고서 자동화

- [ ] ASUS 17s 노트북 배포 (NPU + Intel ARC)
- [ ] PDF 환자 행동 보고서 자동 생성

## 데이터셋 정보

- **출처:** [AI허브 - 낙상사고 위험동작 영상-센서 쌍 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71641)
- **Google Drive:** `fall_dataset/images/` + `fall_dataset/labels_vlm/`

### 라벨링 기준 (Golden Rules)

| 우선순위 | 규칙 | 조건 | 라벨 |
|---|---|---|---|
| 1 | **Fall** | 신체 일부가 바닥에 접촉 | `fall` |
| 2 | **Bed_Exit** | 침대에서 이탈 중인 동작 | `bed_exit` |
| 3 | **Moving** | 서서 이동 중 | `moving` |
| 4 | **Resting** | 침대에 누워 있거나 앉아 있음 | `resting` |

### 라벨링 샘플

```json
{
  "vlm_labels": {
    "status": "fall",
    "captions": [
      "환자가 침대 밖 바닥에 쓰러진 상태로 발견되었습니다. 즉각적인 환자 상태 확인이 필요합니다.",
      "현재 환자는 병실 바닥에 누워 있으며, 낙상으로 인한 부상 여부를 신속히 평가해야 합니다.",
      "환자는 침대에서 떨어진 바닥 매트 위에 엎드린 자세로 완전히 몸이 접촉되어 있습니다.",
      "낙상이 발생한 것으로 판단되며, 즉각적인 개입이 요구됩니다.",
      "환자, 침대 이탈 후 바닥에 낙상 발생."
    ]
  }
}
```

## 프로젝트 구조

```
Patient_Behavior_Reporting_System-VLM/
├── Load_Data.ipynb                           # 데이터 다운로드 및 탐색
├── Labelling_with_Gemini_2.5Flash_API.ipynb  # Gemini API 자동 라벨링 (Phase 1 완료)
├── make_images_zip.py                        # 이미지 zip 패키징 스크립트
├── labels/                                   # 원본 라벨링 JSON (git 제외)
├── labels_vlm/                               # VLM 라벨링 결과 JSON (git 제외)
├── outputs/                                  # 모델 출력 결과 (git 제외)
├── .gitignore
├── PROJECT_PLAN.md                           # 상세 프로젝트 계획 및 의사결정 이력
└── README.md
```

## 포트폴리오 어필 포인트

| 역량 | 상세 |
|---|---|
| **VLM 파인튜닝** | Teacher-Student (Gemini → MobileVLM) + QLoRA |
| **데이터 파이프라인** | AI허브 API 연동, Gemini API 자동 라벨링 (11,600 샘플) |
| **모델 경량화** | MobileVLM 1.4B, INT4 양자화 |
| **모델 변환** | PyTorch → ONNX → OpenVINO |
| **엣지 배포** | ASUS 17s (NPU + Intel ARC) + OpenVINO 실시간 추론 |