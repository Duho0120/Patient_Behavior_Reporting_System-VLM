"""
SY+C3 이미지 → 한 폴더에 모은 뒤 zip으로 저장
실행: python make_images_zip.py
"""

import os
import shutil
import zipfile
import time

# ===== 경로 설정 =====
SRC_DIR  = r"C:\Users\ASUS\Desktop\제로베이스\딥러닝 프로젝트\낙상사고 위험동작 영상-센서 쌍 데이터_병원,후면낙상\3.개방데이터\1.데이터\Training\01.원천데이터\TS\이미지\Y\SY"
FLAT_DIR = r"C:\Users\ASUS\Desktop\sy_c3_images"       # 이미지 모을 임시 폴더
ZIP_PATH = r"C:\Users\ASUS\Desktop\sy_c3_images.zip"   # 최종 zip 저장 위치

# ===== Step 1: 이미지 한 폴더로 모으기 =====
os.makedirs(FLAT_DIR, exist_ok=True)

scenarios = sorted([
    d for d in os.listdir(SRC_DIR)
    if "_C3" in d and os.path.isdir(os.path.join(SRC_DIR, d))
])

print(f"📁 SY+C3 시나리오: {len(scenarios)}개")

count = 0
for scenario in scenarios:
    scenario_path = os.path.join(SRC_DIR, scenario)
    for fname in os.listdir(scenario_path):
        if fname.lower().endswith(".jpg"):
            src  = os.path.join(scenario_path, fname)
            dst  = os.path.join(FLAT_DIR, fname)
            shutil.copy2(src, dst)
            count += 1

print(f"✅ 이미지 복사 완료: {count}장 → {FLAT_DIR}")

# ===== Step 2: 폴더 → zip 생성 =====
print(f"\n📦 zip 생성 중...")
start = time.time()

with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_STORED) as zf:
    for fname in os.listdir(FLAT_DIR):
        zf.write(os.path.join(FLAT_DIR, fname), fname)

elapsed = time.time() - start
size_gb = os.path.getsize(ZIP_PATH) / 1e9

print(f"✅ zip 완료!")
print(f"  - 파일 수: {count}장")
print(f"  - 크기:    {size_gb:.2f} GB")
print(f"  - 소요:    {elapsed:.0f}초")
print(f"  - 저장 위치: {ZIP_PATH}")
print(f"\n이 파일을 Google Drive에 업로드하세요!")
