import os
import csv
import glob
import random
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# -----------------------
# 설정
# -----------------------
DATA_ROOT = "Data/sequences"      # 시퀀스 폴더 루트
TRAIN_CSV = "Data/train.csv"
VAL_CSV   = "Data/val.csv"
TEST_CSV  = "Data/test.csv"

NUM_FRAMES = 30       # 시퀀스 길이 (프레임 수)
BATCH_SIZE = 8
NUM_WORKERS = 4
EPOCHS = 30
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델 선택: "resnet" (single-frame) or "cnn_lstm" (recommended)
MODEL_TYPE = "cnn_lstm"  # 또는 "resnet"

# -----------------------
# Dataset (sequence)
# -----------------------
class SequenceDataset(Dataset):
    """
    sequence csv 에 sequence,label 이 있고
    각 sequence 폴더 안에 00001.jpg ... 존재한다고 가정
    mode: "single" -> pick center frame for single-frame baseline
          "sequence" -> return np array of frames
    """
    def __init__(self, csv_path, seq_root, num_frames=30, transform=None, mode="sequence"):
        self.df = pd.read_csv(csv_path)
        self.seq_root = Path(seq_root)
        self.num_frames = num_frames
        self.transform = transform
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def _load_frames(self, seq_folder):
        files = sorted([p for p in os.listdir(seq_folder) if p.lower().endswith((".jpg",".png"))])
        # If frames less than num_frames, pad last frame
        frames = []
        for i in range(self.num_frames):
            idx = min(i, len(files)-1)
            img_path = os.path.join(seq_folder, files[idx])
            img = T.functional.to_pil_image(cv2_imread_rgb(img_path))
            frames.append(img)
        return frames

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq_name = row['sequence']
        label = int(row['label'])
        seq_folder = os.path.join(self.seq_root, seq_name)

        # get frame list sorted
        files = sorted([p for p in os.listdir(seq_folder) if p.lower().endswith((".jpg",".png"))])
        # ensure at least one frame
        if len(files) == 0:
            raise RuntimeError(f"No frames in {seq_folder}")

        if self.mode == "single":
            # pick center frame
            center_idx = len(files)//2
            frame_path = os.path.join(seq_folder, files[center_idx])
            img = cv2_imread_rgb(frame_path)
            img = T.functional.to_pil_image(img)
            if self.transform:
                img = self.transform(img)
            return img, label

        # sequence mode: load up to num_frames frames (pad by last)
        imgs = []
        for i in range(self.num_frames):
            idx_f = min(i, len(files)-1)
            img = cv2_imread_rgb(os.path.join(seq_folder, files[idx_f]))
            img = T.functional.to_pil_image(img)
            if self.transform:
                img = self.transform(img)  # transform returns tensor
            imgs.append(img)

        # imgs: list of tensors (C,H,W) -> stack into (T,C,H,W)
        seq_tensor = torch.stack(imgs)  # (T, C, H, W)
        return seq_tensor, label

# -----------------------
# helper: cv2 read -> RGB ndarray
# -----------------------
import cv2
def cv2_imread_rgb(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise RuntimeError(f"Failed to read {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

# -----------------------
# Models
# -----------------------
class ResNetBaseline(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        num_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.classifier = nn.Linear(num_f, 1)

    def forward(self, x):
        # x: (B, C, H, W)
        f = self.backbone(x)
        out = self.classifier(f)
        return out.squeeze(1)

class CNN_LSTM(nn.Module):
    def __init__(self, pretrained=True, cnn_out_dim=512, lstm_hidden=256, lstm_layers=1, bidirectional=False):
        super().__init__()
        # use ResNet18 as frame-feature extractor (remove final fc)
        backbone = models.resnet18(pretrained=pretrained)
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])  # until avgpool
        self.cnn_out_dim = backbone.fc.in_features  # 512 for resnet18
        self.lstm = nn.LSTM(input_size=self.cnn_out_dim, hidden_size=lstm_hidden,
                            num_layers=lstm_layers, batch_first=True, bidirectional=bidirectional)
        lstm_out_dim = lstm_hidden * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        feat = self.cnn(x)            # shape (B*T, feat, 1,1)
        feat = feat.view(B, T, -1)    # (B, T, feat_dim)
        out, (h,c) = self.lstm(feat)  # out: (B, T, hidden)
        # use last time-step
        last = out[:, -1, :]
        out = self.fc(last).squeeze(1)
        return out

# -----------------------
# transforms
# -----------------------
train_transform_single = T.Compose([
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
train_transform_seq = train_transform_single  # applied per-frame

val_transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# -----------------------
# DataLoader helpers
# -----------------------
def collate_fn_sequence(batch):
    # batch: list of (seq_tensor(T,C,H,W), label)
    seqs = torch.stack([b[0] for b in batch], dim=0)  # (B, T, C, H, W)
    labels = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    return seqs, labels

# -----------------------
# Metrics utils
# -----------------------
def compute_metrics_all(y_true, y_pred_logits, thresh=0.5):
    probs = torch.sigmoid(torch.tensor(y_pred_logits)).numpy()
    preds = (probs >= thresh).astype(int)
    y_true = np.array(y_true).astype(int)
    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, zero_division=0)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    return {"acc": acc, "f1": f1, "prec": prec, "rec": rec}

# -----------------------
# Training / Eval loops
# -----------------------
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    ys, y_logits = [], []
    for batch in tqdm(loader, desc="train", leave=False):
        if MODEL_TYPE == "resnet":
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)
            logits = model(imgs)
        else:
            seqs, labels = batch
            seqs = seqs.to(device)   # (B,T,C,H,W)
            labels = labels.to(device)
            logits = model(seqs)

        loss = criterion(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        ys.extend(labels.detach().cpu().numpy().tolist())
        y_logits.extend(logits.detach().cpu().numpy().tolist())

    avg_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics_all(ys, y_logits)
    metrics["loss"] = avg_loss
    return metrics

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    ys, y_logits = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", leave=False):
            if MODEL_TYPE == "resnet":
                imgs, labels = batch
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits = model(imgs)
            else:
                seqs, labels = batch
                seqs = seqs.to(device)
                labels = labels.to(device)
                logits = model(seqs)

            loss = criterion(logits, labels)
            running_loss += loss.item() * labels.size(0)
            ys.extend(labels.detach().cpu().numpy().tolist())
            y_logits.extend(logits.detach().cpu().numpy().tolist())

    avg_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics_all(ys, y_logits)
    metrics["loss"] = avg_loss
    return metrics

# -----------------------
# Main train function
# -----------------------
def main():
    # dataset + dataloader
    if MODEL_TYPE == "resnet":
        train_ds = SequenceDataset(TRAIN_CSV, DATA_ROOT, num_frames=NUM_FRAMES, transform=train_transform_single, mode="single")
        val_ds   = SequenceDataset(VAL_CSV, DATA_ROOT, num_frames=NUM_FRAMES, transform=val_transform, mode="single")
        test_ds  = SequenceDataset(TEST_CSV, DATA_ROOT, num_frames=NUM_FRAMES, transform=val_transform, mode="single")
    else:
        train_ds = SequenceDataset(TRAIN_CSV, DATA_ROOT, num_frames=NUM_FRAMES, transform=train_transform_seq, mode="sequence")
        val_ds   = SequenceDataset(VAL_CSV, DATA_ROOT, num_frames=NUM_FRAMES, transform=val_transform, mode="sequence")
        test_ds  = SequenceDataset(TEST_CSV, DATA_ROOT, num_frames=NUM_FRAMES, transform=val_transform, mode="sequence")

    if MODEL_TYPE == "resnet":
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, collate_fn=collate_fn_sequence)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn_sequence)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, collate_fn=collate_fn_sequence)

    # model
    if MODEL_TYPE == "resnet":
        model = ResNetBaseline(pretrained=True)
    else:
        model = CNN_LSTM(pretrained=True, lstm_hidden=256, lstm_layers=1, bidirectional=False)

    model = model.to(DEVICE)

    # class imbalance -> pos_weight for BCEWithLogitsLoss
    df_train = pd.read_csv(TRAIN_CSV)
    pos = (df_train['label']==1).sum()
    neg = (df_train['label']==0).sum()
    print("Train pos/neg:", pos, neg)
    # pos_weight = neg / pos  (torch expects float tensor)
    pos_weight = torch.tensor([neg / (pos + 1e-6)], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    best_val_f1 = -1.0
    patience = 6
    cur_pat = 0

    for epoch in range(1, EPOCHS+1):
        print(f"\n=== Epoch {epoch}/{EPOCHS} ===")
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)

        print("Train:", train_metrics)
        print("Val:  ", val_metrics)

        # scheduler step on val f1
        scheduler.step(val_metrics['f1'])

        # checkpoint based on val f1
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "epoch": epoch,
                "val_f1": best_val_f1
            }, "best_model.pth")
            print("[SAVE] best_model.pth (val f1 improved)")
            cur_pat = 0
        else:
            cur_pat += 1
            print(f"[INFO] no improvement (patience {cur_pat}/{patience})")
            if cur_pat >= patience:
                print("[EARLY STOP] no improvement for many epochs.")
                break

    # load best and test
    print("\n=== LOAD BEST and TEST ===")
    ck = torch.load("best_model.pth", map_location=DEVICE)
    model.load_state_dict(ck["model_state"])
    test_metrics = evaluate(model, test_loader, criterion, DEVICE)
    print("Test:", test_metrics)

if __name__ == "__main__":
    main()