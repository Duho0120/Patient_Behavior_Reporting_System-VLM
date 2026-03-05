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
Phase 1: 데이터 구축 (Gemini API 라벨링)        ← 현재 진행 중
Phase 2: QLoRA 파인튜닝 (MobileVLM 1.4B)
Phase 3: 모델 변환 (PyTorch → ONNX → OpenVINO)
Phase 4: C++ 실시간 추론 파이프라인 구축
Phase 5: ASUS 17s (NPU + Intel ARC) 배포 + 보고서 자동화
```

### Phase 1: 데이터 구축 ← **현재**

- [x] AI허브 라벨링 JSON 다운로드 (217,536개)
- [x] 필터링: SY(측면낙상) + 병원(H_A,H_D,H_B) + C3 카메라
- [x] 이미지/라벨 Google Drive 업로드
- [x] Gemini 2.5 Flash REST API 연동 완료
- [ ] SY+C3 2,320장 × 5캡션 자동 라벨링

### Phase 2: QLoRA 파인튜닝

- [ ] MobileVLM 1.4B + QLoRA (Colab)
- [ ] 대화형 학습 데이터 포맷 (instruction tuning)
- [ ] 학습/검증 분리, 성능 평가

### Phase 3: 모델 경량화 & 변환

- [ ] PyTorch → ONNX 변환
- [ ] ONNX → OpenVINO IR 변환 (Intel NPU/ARC 최적화)
- [ ] 추론 속도 벤치마크

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
| 이미지 원본 (로컬) | `C:\...\딥러닝 프로젝트\낙상사고...\이미지\Y\SY\` |
| 라벨 원본 (로컬) | `프로젝트\labels\이미지\Y\SY\` |
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

### API 호출 정보

- **API**: Gemini 2.5 Flash (REST API, Python 3.8 호환)
- **URL**: `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent`
- **API Key**: `API_KEY_to_Labelling.env` 파일에 저장
- **예상 비용**: SY 2,320장 × ~$0.001 = ~$2.3 (약 3,200원)
- **예상 시간**: ~8시간 20분 (이미지당 ~13초)

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
