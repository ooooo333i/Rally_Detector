import pandas as pd

# 라벨 파일 경로
label_path = "Data/labels.csv"

df = pd.read_csv(label_path)

print("===== Label Count =====")
print(df['label'].value_counts())

print("\n===== Label Ratio =====")
print(df['label'].value_counts(normalize=True) * 100)

print("\n===== Total Sequences =====")
print(len(df))