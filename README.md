# 🏥 Patient Behavior Reporting System - VLM Phase 2

본 프로젝트는 CCTV 영상을 분석하여 환자의 위험 행동(낙상, 침대 이탈 등)을 자동으로 감지하고 보고하는 시스템을 구축하는 것을 목표로 합니다. Phase 2에서는 **Gemini 2.5 Pro (1.5 Pro API)**를 활용하여 고정밀 라벨링 데이터를 생성하고 이를 검수하는 파이프라인을 구축했습니다.

---

## 🚀 주요 진행 사항 (Recent Progress)

### 1. V2 자동 라벨링 파이프라인 구축
- **모델**: `gemini-1.5-pro` (Gemini 2.5 Pro 기반 최신 성능)
- **대상**: 총 **2,320개**의 CCTV 프레임 (시나리오 C3: 낙상 및 이탈 중심)
- **성과**: 
  - API 쿼터(429 Error) 및 서버 부하(503 Error)를 자동으로 감지하고 재시도하는 안정적인 데이터 생성 로직 구현
  - 복잡한 병원 환경에서의 환자 상태를 상세 텍스트로 라벨링 완료

### 2. 전용 라벨 검수 GUI 도구 개발 (`label_reviewer.py`)
Gemini가 생성한 라벨의 정확도를 인간이 최종 검증할 수 있도록 돕는 경량 GUI 프로그램을 제작했습니다.
- **주요 기능**:
  - 이미지와 JSON 라벨의 1:1 대조 확인
  - 방향키를 이용한 초고속 내비게이션 및 `PageUp/Down`을 통한 100장 단위 스킵
  - 특정 카테고리(예: '낙상 발생')만 모아보는 필터링 기능
  - GUI 내부에서 즉시 라벨 수정 및 JSON 파일 자동 저장

---

## 📂 프로젝트 구조 (Structure)

```text
├── data/
│   ├── images/         # [Ignore] 원본 CCTV 이미지 데이터
│   └── v2/
│       └── labels_vlm/ # [Ignore] Gemini가 생성한 JSON 라벨 결과물
├── notebooks/
│   └── 07_v2_Labelling_with_Gemini.ipynb  # 자동 라벨링 실행 노트북
├── scripts/
│   └── label_reviewer.py                  # 전용 검수 GUI 도구
├── PROJECT_PLAN.md      # 향후 추진 계획
└── README.md            # 프로젝트 개요 (현재 파일)
```

---

## 🛠️ 사용 방법 (Usage)

### 라벨 검수 도구 실행
수천 장의 라벨링 결과를 빠르게 확인하고 수정하려면 다음 명령어를 실행하세요:
```powershell
python scripts/label_reviewer.py
```

### 환경 설정
본 프로젝트는 `.env` 파일을 통해 Google API KEY를 관리합니다. (깃허브 업로드 시 보안을 위해 자동 제외됨)

---

## 📈 향후 계획 (Next Steps)
- [ ] 생성된 2,320개 라벨링 데이터 전수 검사 완료
- [ ] MobileVLM V2 모델 파인튜닝 (Google Colab A100/L4 활용)
- [ ] 실시간 추론 엔진과 검증 결과 비교 분석

---
> **Note**: 현재 데이터 보호를 위해 이미지 및 라벨 원본 파일은 `.gitignore`에 의해 제외되어 있습니다.