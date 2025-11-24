import cv2
import os
import csv

# ===== 설정 =====
image_folder = "Data/frames"   # 이미지 폴더
csv_file = "Data/labels.csv"   # CSV 파일

# CSV 파일이 없으면 생성하고, 있으면 이어쓰기
if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])  # 헤더

# 이미지 리스트 정렬
images = sorted(os.listdir(image_folder))

# 이미 CSV에 라벨링된 이미지 제외
labeled_images = set()
with open(csv_file, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        labeled_images.add(row["filename"])

# CSV 열기 (append 모드)
with open(csv_file, "a", newline="") as f:
    writer = csv.writer(f)

    for img_name in images:
        if img_name in labeled_images:
            continue  # 이미 라벨링된 이미지 건너뛰기

        img_path = os.path.join(image_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_name}, skipping…")
            continue

        # 이미지 보여주기
        cv2.imshow("Image", img)
        print(f"Label the image: {img_name} (0=non rally, 1=rally)")

        key = cv2.waitKey(0)  # 키 입력 기다리기
        if key == ord("0"):
            label = 0
        elif key == ord("1"):
            label = 1
        else:
            print("Invalid input, skipping...")
            cv2.destroyAllWindows()
            continue

        # CSV에 저장
        writer.writerow([img_name, label])
        cv2.destroyAllWindows()