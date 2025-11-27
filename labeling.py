import os
import cv2
import csv

SEQ_ROOT = "Data/test/sequences"
OUT_CSV = "Data/test/labels.csv"

os.makedirs(SEQ_ROOT, exist_ok=True)

# ================================
# 기존 CSV 라벨 불러오기
# ================================
labeled = {}
if os.path.exists(OUT_CSV):
    with open(OUT_CSV, newline='') as f:
        reader = csv.DictReader(f)
        for r in reader:
            labeled[r['sequence']] = int(r['label'])

# ================================
# 라벨링 시작
# ================================
seq_list = sorted([d for d in os.listdir(SEQ_ROOT) if d.startswith("seq_")])

# CSV 파일 열어두기
with open(OUT_CSV, "a", newline="") as f:
    writer = csv.writer(f)

    # CSV가 비어있는 경우 헤더 생성
    if os.stat(OUT_CSV).st_size == 0:
        writer.writerow(["sequence", "label"])

    for seq in seq_list:
        if seq in labeled:
            print(f"[SKIP] {seq} (already labeled)")
            continue

        seq_path = os.path.join(SEQ_ROOT, seq)
        frames = sorted([
            os.path.join(seq_path, fn)
            for fn in os.listdir(seq_path)
            if fn.lower().endswith((".jpg", ".png"))
        ])

        if not frames:
            print(f"[WARN] {seq} is empty.")
            continue

        print(f"\n=== Labeling {seq} ===")
        print("Press 0=non-rally, 1=rally, s=skip, q=quit")

        key_pressed = None

        # ================================
        # 시퀀스 재생
        # ================================
        for frame_path in frames:
            img = cv2.imread(frame_path)
            if img is None:
                continue

            cv2.imshow("Sequence Preview", img)
            key = cv2.waitKey(33)  # 30fps (1000ms/30 ≈ 33ms)

            if key != -1:
                key_pressed = chr(key & 0xFF)
                break

        # 입력 없으면 기다림
        if key_pressed is None:
            key = cv2.waitKey(0)
            key_pressed = chr(key & 0xFF)

        # ================================
        # 라벨 처리
        # ================================
        if key_pressed == '0':
            writer.writerow([seq, 0])
            print(f"> Saved {seq} → 0 (non-rally)")
        elif key_pressed == '1':
            writer.writerow([seq, 1])
            print(f"> Saved {seq} → 1 (rally)")
        elif key_pressed == 's':
            print(f"> Skip {seq}")
        elif key_pressed == 'q':
            print("Exiting...")
            break
        else:
            print(f"> Unknown key '{key_pressed}', skipping...")

cv2.destroyAllWindows()