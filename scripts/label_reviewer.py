import os
import json
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from pathlib import Path

class LabelReviewer:
    def __init__(self, root):
        self.root = root
        self.root.title("V2 Label Review Tool - Patient Behavior")
        self.root.geometry("1100x800")

        # --- 경로 설정 (사용자 제공 절대 경로 반영) ---
        self.img_root = Path(r"C:\Users\ASUS\Desktop\제로베이스\딥러닝 프로젝트\낙상사고 위험동작 영상-센서 쌍 데이터_병원,후면낙상\3.개방데이터\1.데이터\Training\01.원천데이터\TS\이미지\Y\SY")
        self.label_dir = Path(r"C:\Users\ASUS\Desktop\제로베이스\Patient_Behavior_Reporting_System-VLM\data\v2\labels_vlm")
        
        # 경로 확인용 로그
        if not self.img_root.exists():
             print(f"이미지 경로를 찾을 수 없습니다: {self.img_root}")
        if not self.label_dir.exists():
             print(f"라벨 경로를 찾을 수 없습니다: {self.label_dir}")

        # --- 데이터 로드 ---
        self.image_list = []
        self.load_data()
        
        self.current_idx = 0
        self.filtered_list = self.image_list.copy()
        
        # --- UI 구성 ---
        self.setup_ui()
        
        # --- 데이터 표시 ---
        if self.image_list:
            self.show_current()
        else:
            msg = f"라벨링된 데이터를 찾을 수 없습니다.\n\n탐색 경로:\n1. 이미지: {self.img_root}\n2. 라벨: {self.label_dir}"
            messagebox.showwarning("경고", msg)

    def load_data(self):
        """SY 폴더 내의 _C3 시나리오 이미지와 대응하는 JSON 라벨 매핑"""
        if not self.img_root.exists():
            return

        # 중복 방지를 위해 딕셔너리 사용 (파일명 기준)
        seen_images = {}
        extensions = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG"]
        
        for ext in extensions:
            for path in self.img_root.rglob(ext):
                # 파일명에 _C3이 포함된 것만 처리
                if "_C3" in str(path):
                    # 파일명만 추출 (중복 체크용 키)
                    fname = path.name
                    if fname in seen_images:
                        continue
                        
                    # 대응하는 JSON 라벨 파일명 생성
                    json_name = Path(fname).stem + ".json"
                    json_path = self.label_dir / json_name
                    
                    if json_path.exists():
                        seen_images[fname] = {
                            "img_path": path,
                            "json_path": json_path,
                            "name": fname
                        }
        
        # 딕셔너리의 값을 리스트로 변환 후 정렬
        self.image_list = sorted(seen_images.values(), key=lambda x: x['name'])

    def setup_ui(self):
        # 상단 제어 바
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(side="top", fill="x")

        self.filter_var = tk.StringVar(value="전체 보기")
        categories = ["전체 보기", "환자 휴식 중", "환자 이동 중", "환자 침대 이탈", "환자 낙상 발생"]
        filter_combo = ttk.Combobox(top_frame, textvariable=self.filter_var, values=categories, state="readonly")
        filter_combo.pack(side="left", padx=5)
        filter_combo.bind("<<ComboboxSelected>>", self.apply_filter)

        self.stats_label = ttk.Label(top_frame, text=f"총 {len(self.image_list)}개 로드됨")
        self.stats_label.pack(side="left", padx=20)

        # 메인 영역 (이미지 + 정보)
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill="both", expand=True)

        # 이미지 캔버스
        self.img_label = ttk.Label(main_frame)
        self.img_label.pack(side="left", fill="both", expand=True)

        # 우측 정보창
        info_frame = ttk.Frame(main_frame, width=300, padding="10")
        info_frame.pack(side="right", fill="y")

        ttk.Label(info_frame, text="파일명:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.name_label = ttk.Label(info_frame, text="", wraplength=250)
        self.name_label.pack(anchor="w", pady=(0, 20))

        ttk.Label(info_frame, text="현재 라벨:", font=("Arial", 10, "bold")).pack(anchor="w")
        self.category_var = tk.StringVar()
        self.cat_menu = ttk.Combobox(info_frame, textvariable=self.category_var, values=categories[1:], state="readonly")
        self.cat_menu.pack(fill="x", pady=5)

        ttk.Label(info_frame, text="상세 프롬프트/내용:", font=("Arial", 10, "bold")).pack(anchor="w", pady=(15, 0))
        self.prompt_text = tk.Text(info_frame, height=5, width=30, wrap="word")
        self.prompt_text.pack(pady=5)

        save_btn = ttk.Button(info_frame, text="저장 (Enter)", command=self.save_current)
        save_btn.pack(fill="x", pady=20)

        # 하단 내비게이션
        nav_frame = ttk.Frame(self.root, padding="10")
        nav_frame.pack(side="bottom", fill="x")

        ttk.Button(nav_frame, text="⏪ -100", command=lambda: self.skip_n(-100)).pack(side="left", padx=5)
        ttk.Button(nav_frame, text="◀ 이전 (←)", command=self.prev_item).pack(side="left", padx=5)
        
        self.progress_label = ttk.Label(nav_frame, text="0 / 0", font=("Arial", 10, "bold"))
        self.progress_label.pack(side="left", expand=True)
        
        ttk.Button(nav_frame, text="다음 (→) ▶", command=self.next_item).pack(side="left", padx=5)
        ttk.Button(nav_frame, text="+100 ⏩", command=lambda: self.skip_n(100)).pack(side="left", padx=5)

        # 단축키 바인딩
        self.root.bind("<Left>", lambda e: self.prev_item())
        self.root.bind("<Right>", lambda e: self.next_item())
        self.root.bind("<Prior>", lambda e: self.skip_n(-100)) # PageUp
        self.root.bind("<Next>", lambda e: self.skip_n(100))  # PageDown
        self.root.bind("<Return>", lambda e: self.save_current())

    def show_current(self):
        if not self.filtered_list:
            return
            
        item = self.filtered_list[self.current_idx]
        
        # 이미지 표시
        img = Image.open(item['img_path'])
        # 캔버스 크기에 맞게 리사이즈 (비율 유지)
        img.thumbnail((700, 600))
        self.tk_img = ImageTk.PhotoImage(img)
        self.img_label.config(image=self.tk_img)

        # 정보 표시
        self.name_label.config(text=item['name'])
        
        with open(item['json_path'], 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.category_var.set(data.get('label', ''))
            self.prompt_text.delete("1.0", tk.END)
            self.prompt_text.insert("1.0", data.get('prompt', ''))
            
        self.progress_label.config(text=f"{self.current_idx + 1} / {len(self.filtered_list)}")

    def next_item(self):
        if self.current_idx < len(self.filtered_list) - 1:
            self.current_idx += 1
            self.show_current()

    def prev_item(self):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.show_current()

    def skip_n(self, n):
        """n만큼 건너뛰기 (100장 등)"""
        new_idx = self.current_idx + n
        if new_idx < 0:
            new_idx = 0
        if new_idx >= len(self.filtered_list):
            new_idx = len(self.filtered_list) - 1
            
        self.current_idx = new_idx
        self.show_current()

    def save_current(self):
        item = self.filtered_list[self.current_idx]
        new_label = self.category_var.get()
        new_prompt = self.prompt_text.get("1.0", tk.END).strip()

        with open(item['json_path'], 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        data['label'] = new_label
        data['prompt'] = new_prompt

        with open(item['json_path'], 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        print(f"저장됨: {item['name']} -> {new_label}")
        # 다음 아이템으로 자동 이동 (편의성)
        self.next_item()

    def apply_filter(self, event=None):
        target = self.filter_var.get()
        if target == "전체 보기":
            self.filtered_list = self.image_list.copy()
        else:
            new_list = []
            for item in self.image_list:
                with open(item['json_path'], 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get('label') == target:
                        new_list.append(item)
            self.filtered_list = new_list
        
        self.current_idx = 0
        if self.filtered_list:
            self.show_current()
        else:
            self.img_label.config(image='')
            self.name_label.config(text="해당 조건의 파일이 없습니다.")
            self.progress_label.config(text="0 / 0")

if __name__ == "__main__":
    root = tk.Tk()
    app = LabelReviewer(root)
    root.mainloop()
