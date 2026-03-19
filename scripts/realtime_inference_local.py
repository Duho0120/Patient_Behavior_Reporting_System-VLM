"""
realtime_inference_local.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
스마트병동 환자안전 모니터링 — 로컬 실행 버전
Fine-tuned MobileVLM V2로 영상을 2초 간격 분석하여
환자 행동 변화 타임라인 보고서를 생성합니다.

실행 방법:
  conda activate ds_study
  python realtime_inference_local.py --video demo_videos/video1.mp4

필요 패키지 (ds_study 환경에 없을 경우 설치):
  pip install transformers peft pillow opencv-python
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import sys
import argparse
import zipfile
import time
import cv2
from datetime import datetime
from pathlib import Path
from PIL import Image

# ── 경로 설정 ─────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
ADAPTER_ZIP   = BASE_DIR / 'mobilevlm_lora_adapter.zip'
ADAPTER_DIR   = BASE_DIR / 'mobilevlm_lora_adapter'
VIDEO_DIR     = BASE_DIR / 'demo_videos'
RESULT_DIR    = BASE_DIR / 'demo_results'
MOBILEVLM_DIR = BASE_DIR / 'MobileVLM'    # GitHub 레포 클론 위치

# ── 설정 ──────────────────────────────────────────────────────────────────
INTERVAL_SEC  = 2          # 추론 간격 (초)
PROMPT        = '현재 환자의 상태를 보고하세요.'
ALERT_KEYWORD = '낙상'     # 이 단어 감지 시 즉시 경보
MODEL_NAME    = 'mtgv/MobileVLM_V2-1.7B'


def setup():
    """초기 환경 준비 — 레포 클론, 어댑터 압축 해제"""
    RESULT_DIR.mkdir(exist_ok=True)

    # MobileVLM 레포 클론 (없을 경우)
    if not MOBILEVLM_DIR.exists():
        print('[setup] MobileVLM 레포 클론 중...')
        os.system(f'git clone -q https://github.com/Meituan-AutoML/MobileVLM {MOBILEVLM_DIR}')
    sys.path.insert(0, str(MOBILEVLM_DIR))

    # LoRA 어댑터 압축 해제 (없을 경우)
    if not ADAPTER_DIR.exists():
        if ADAPTER_ZIP.exists():
            print('[setup] LoRA 어댑터 압축 해제 중...')
            with zipfile.ZipFile(ADAPTER_ZIP, 'r') as z:
                z.extractall(BASE_DIR)
            print(f'[setup] 완료: {ADAPTER_DIR}')
        else:
            print(f'[ERROR] 어댑터 파일이 없습니다: {ADAPTER_ZIP}')
            sys.exit(1)


def load_model():
    """Fine-tuned MobileVLM 로드 (CPU, fp32 — CUDA 없음)

    확인된 실제 시그니처:
      load_pretrained_model(model_path, load_8bit=False, load_4bit=False,
                            device_map='auto', device='cuda')
    - model_base, model_name 파라미터 없음
    - load_4bit=True 시 최신 transformers와 충돌 → False 사용
    - 로컬 배포 시 OpenVINO INT4 변환으로 경량화 예정 (Phase 3)
    """
    import torch
    from peft import PeftModel
    from mobilevlm.model.mobilevlm import load_pretrained_model
    from mobilevlm.utils import disable_torch_init

    print('[모델] 로드 중... (CPU/fp32 — 최초 다운로드 시 시간 소요)')
    print('[모델] 주의: CUDA가 없어 추론이 이미지당 1~2분 예상됩니다.')
    print('[모델] 향후 OpenVINO INT4 변환 후 1~2초/장으로 개선 예정')
    disable_torch_init()

    # ── 실제 확인된 호출 방식 ──────────────────────────────────────
    # load_4bit=False: 최신 transformers와 bitsandbytes 충돌 회피
    # device='cpu': 로컬 CUDA 없음
    tokenizer, model_base, image_processor, _ = load_pretrained_model(
        MODEL_NAME,          # model_path (위치 인자 1개만)
        load_8bit=False,
        load_4bit=False,
        device='cpu',
    )

    print('[모델] LoRA 어댑터 적용 중...')
    model = PeftModel.from_pretrained(model_base, str(ADAPTER_DIR))
    model.eval()
    print('[모델] ✅ Fine-tuned MobileVLM 로드 완료')
    return tokenizer, model, image_processor


def infer_frame(pil_image, tokenizer, model, image_processor):
    """PIL 이미지 1장 → 추론 결과 문자열"""
    import torch
    from mobilevlm.utils import process_images, tokenizer_image_token
    from mobilevlm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from mobilevlm.conversation import conv_templates

    image_tensor = process_images([pil_image], image_processor, model.config)[0]
    image_tensor = image_tensor.to('cpu', dtype=torch.float32)

    full_prompt = DEFAULT_IMAGE_TOKEN + '\n' + PROMPT
    conv = conv_templates['v1'].copy()   # MobileVLM V2 = Vicuna 기반 → 'v1' 사용
    conv.append_message(conv.roles[0], full_prompt)
    conv.append_message(conv.roles[1], None)

    input_ids = tokenizer_image_token(
        conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0),
            do_sample=False,
            max_new_tokens=64,
            use_cache=True,
        )
    return tokenizer.decode(
        output_ids[0, input_ids.shape[1]:], skip_special_tokens=True
    ).strip()


def format_time(seconds):
    return f'{int(seconds)//60:02d}:{int(seconds)%60:02d}'


def process_video(video_path, tokenizer, model, image_processor):
    """영상 1개 처리 → 타임라인 보고서 반환"""
    cap = cv2.VideoCapture(str(video_path))
    fps        = cap.get(cv2.CAP_PROP_FPS) or 30
    total_f    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration   = total_f / fps
    frame_step = max(1, int(fps * INTERVAL_SEC))

    print(f'\n{"="*60}')
    print(f'📹 영상: {video_path.name}')
    print(f'   길이: {format_time(duration)} | FPS: {fps:.1f} | 분석 간격: {INTERVAL_SEC}초')
    print(f'{"="*60}')

    report      = []
    prev_status = None
    frame_idx   = 0
    alert_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step != 0:
            frame_idx += 1
            continue

        timestamp  = frame_idx / fps
        time_label = format_time(timestamp)

        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        t0      = time.time()
        status  = infer_frame(pil_img, tokenizer, model, image_processor)
        elapsed = time.time() - t0

        changed = (status != prev_status)

        if changed or prev_status is None:
            has_alert = ALERT_KEYWORD in status
            if has_alert:
                alert_count += 1
            alert_str = '  ⚠️  → 간호사 호출!' if has_alert else ''
            print(f'{time_label}  {status}{alert_str}  [{elapsed:.1f}s]')
            report.append({'time': time_label, 'status': status, 'alert': has_alert})
            prev_status = status
        else:
            print(f'{time_label}  (유지) [{elapsed:.1f}s]')

        frame_idx += 1

    cap.release()
    print(f'\n✅ 완료 | 상태 변화: {len(report)}회 | 낙상 경보: {alert_count}회')
    return report, alert_count


def save_report(video_name, report, alert_count):
    """보고서 txt 저장"""
    out_path = RESULT_DIR / f'{video_name}_report.txt'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('환자 행동 모니터링 보고서\n')
        f.write(f'생성일시: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n')
        f.write(f'모델: Fine-tuned MobileVLM V2 1.7B (QLoRA)\n')
        f.write('=' * 40 + '\n')
        for entry in report:
            alert_str = '  ⚠️ 낙상 경보!' if entry['alert'] else ''
            f.write(f"{entry['time']}  {entry['status']}{alert_str}\n")
        f.write('=' * 40 + '\n')
        f.write(f'총 상태 변화: {len(report)}회\n')
        f.write(f'낙상 경보 발생: {alert_count}회\n')
    print(f'📄 보고서 저장: {out_path}')
    return out_path


def main():
    parser = argparse.ArgumentParser(description='스마트병동 환자 행동 모니터링')
    parser.add_argument('--video', type=str, default=None,
                        help='영상 파일 경로 (미지정 시 demo_videos/ 전체 처리)')
    parser.add_argument('--interval', type=int, default=2,
                        help='추론 간격 (초, 기본값: 2)')
    args = parser.parse_args()

    global INTERVAL_SEC
    INTERVAL_SEC = args.interval

    # 환경 설정
    setup()

    # 모델 로드
    tokenizer, model, image_processor = load_model()

    # 처리할 영상 목록 결정
    if args.video:
        video_list = [Path(args.video)]
    else:
        video_list = sorted(VIDEO_DIR.glob('*'))
        video_list = [
            v for v in video_list
            if v.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv')
        ]

    if not video_list:
        print(f'[ERROR] 처리할 영상이 없습니다. demo_videos/ 폴더에 영상을 넣어주세요.')
        sys.exit(1)

    print(f'\n총 {len(video_list)}개 영상 처리 시작')

    # 영상 순차 처리
    for video_path in video_list:
        report, alert_count = process_video(video_path, tokenizer, model, image_processor)
        save_report(video_path.stem, report, alert_count)

    print('\n🎉 전체 처리 완료!')


if __name__ == '__main__':
    main()