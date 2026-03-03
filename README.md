# 🏥 스마트병동 환자안전 모니터링 시스템 (Patient Behavior Reporting System)

거대 시각-언어 모델(VLM)을 활용하여 병원 내 환자의 이상 행동(낙상, 침대 이탈 등)을 실시간으로 감지하고, 환자 행동 기록 보고서를 자동 생성하는 **End-to-End 솔루션**입니다.

## 기술 스택

| 구분 | 기술 | 도입 이유 |
|---|---|---|
| 모델 | MobileVLM v2 1.4B | 경량 VLM, Jetson Orin Nano 배포 가능 (1~2초 추론) |
| 학습 | QLoRA (4-bit) | 대형 모델을 효율적으로 학습, 메모리 절감 |
| 인프라 | Colab (A100/H100) | 대형 모델 고속 학습 |
| 라벨링 | Gemini 2.5 Flash API | Teacher-Student 방식, 이미지당 5개 캡션 자동 생성 |
| 배포 | Jetson Orin Nano | VLM 단독 실시간 추론 (2~3초 간격) |

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

### Phase 1: 고품질 학습 데이터 구축 ← **현재 단계**

- [x] AI허브 API 연동 및 데이터셋 구조 파악
- [x] 라벨링 데이터(JSON) 다운로드 및 분석 (217,536개 파일)
- [x] 데이터 필터링 조건 확정: **측면낙상(SY) + 병원(H_A, H_D, H_B) + C3 카메라**
- [x] C3 이미지 **5,900장** 확보 (정상 3,230 + 측면낙상 2,670) → Google Drive 업로드 완료
- [x] SY+C3 라벨링 JSON **3,430개** → Google Drive 업로드 완료
- [x] Gemini 2.5 Flash API 연동 (REST API 방식)
- [ ] COCO 스타일 5개 캡션 자동 라벨링 (진행 중)

### Phase 2: QLoRA 기반 파인튜닝

- [ ] MobileVLM 1.4B + QLoRA 어댑터 적용
- [ ] 한국어 환자 상태 보고 문체 학습
- [ ] 하이퍼파라미터 튜닝

### Phase 3: 실시간 감지 로직

- [ ] 2~3초 간격 실시간 캠 영상 분석
- [ ] 상태 변화 감지 (이전 상태와 비교)
- [ ] Jetson Orin Nano 배포 (INT4 양자화)

### Phase 4: 시연 및 보고서 자동화

- [ ] 실시간 환자 행동 기록
- [ ] PDF 환자 안전 보고서 자동 생성

## 데이터셋 정보

- **출처:** [AI허브 - 낙상사고 위험동작 영상-센서 쌍 데이터](https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71641)
- **Google Drive:** `fall_dataset/images/` (5,900장) + `fall_dataset/labels/` (3,430개)

### 라벨링 4대 원칙 (Golden Rules)

| 규칙 | 조건 | 라벨링 |
|---|---|---|
| Fall (최우선) | 바닥 접촉 | 무조건 '낙상 발생' |
| Bed_Exit | 침대 + 이탈 동작 | "환자가 병상을 이탈중입니다" |
| No_Bed | 침대 미식별 | '병상 이탈' 사용 금지 |
| Moving | 서서 걷는 동작 | '이동 중' |

## 프로젝트 구조

```
Patient_Behavior_Reporting_System-VLM/
├── Load_Data.ipynb                           # 데이터 다운로드 및 탐색
├── Labelling_with_Gemini_2.5Flash_API.ipynb  # Gemini API 자동 라벨링
├── images/                                   # 원천 이미지 데이터 (git 제외)
├── labels/                                   # 라벨링 JSON 데이터 (git 제외)
├── outputs/                                  # 모델 출력 결과 (git 제외)
├── .gitignore
└── README.md
```