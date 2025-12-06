# Artificial-Intelligence-Fall-2025-AnSeungGi


## Setup

1. Clone this repository.
2. Create a new folder 'Data' and download datasets. [Dataset](https://drive.google.com/drive/folders/19a5XDU64GR2ml62koi11TIn2tR8SgR4_?usp=drive_link)
3. Run 'main.ipynb'.


## DataSet

Download here.[Dataset-(GoogleDrive)](https://drive.google.com/drive/folders/19a5XDU64GR2ml62koi11TIn2tR8SgR4_?usp=drive_link)

(it might take much time.)

---

The dataset consists of badminton rally sequences extracted from YouTube videos.
Each sequence contains **30 consecutive frames**, resized to **224×224**.

The dataset is split as follows:
- **Train**: 618 sequences
- **Validation**: 256 sequences
- **Test**: 116 sequences

---

### 🔧 Data Preprocessing - role of 'collecting_data.py'

1. Download YouTube videos using **pytubefix**  
2. Extract frames every *interval seconds*  
3. Resize to **224×224**  
4. Group into sequences of **30 frames**
5. Manually label using 'labeling.py'

### structure

```
Data
│
├── train
│     │
│     ├── sequences
│     │       ├── seq_00001
│     │       │       ├── 00001.jpg
│     │       │       ├── 00002.jpg
│     │       │       └── ...
│     │       ├── seq_00002
│     │       │       └── ...
│     │       └── ...
│     │
│     └── train.csv
│
├── validation
│     │
│     ├── sequences
│     │       ├── seq_00001
│     │       │       ├── 00001.jpg
│     │       │       └── ...
│     │       └── ...
│     │
│     └── validation.csv
│
└── test
      │
      ├── sequences
      │       ├── seq_00001
      │       │       ├── 00001.jpg
      │       │       └── ...
      │       └── ...
      │
      └── test.csv

```

## 🧠 Model: RESNET + LSTM Hybrid

This model consists of two major parts:

---

### **1. CNN Feature Extractor (ResNet18)**

- Pretrained on ImageNet  
- Outputs a **512-dimensional feature** for each frame  
- Frozen backbone to prevent overfitting and reduce compute

---

### **2. LSTM Temporal Model**

- 2-layer LSTM  
- Hidden size = 128  
- Mean pooling across the 30-frame sequence  
- Fully connected head for binary classification

---

### RES_LSTM Model Summary - (tsummary)

input (1, 30, 3, 244, 244)

| Layer Type         | Name / (Depth-Idx)          | Output Shape       | Param #    |
|-------------------|-----------------------------|--------------------|------------|
| **Model**         | RES_LSTM                    | [1, 1]             | --         |
| **CNN Backbone (ResNet18, frozen)** ||||
| Conv2d            | 2-1                         | [30, 64, 112,112]  | 9,408      |
| BatchNorm2d       | 2-2                         | [30, 64,112,112]   | 128        |
| ReLU              | 2-3                         | [30, 64,112,112]   | 0          |
| MaxPool2d         | 2-4                         | [30, 64,56,56]     | 0          |
| BasicBlock        | 3-1                         | [30, 64,56,56]     | 73,984     |
| BasicBlock        | 3-2                         | [30, 64,56,56]     | 73,984     |
| BasicBlock        | 3-3                         | [30,128,28,28]     | 230,144    |
| BasicBlock        | 3-4                         | [30,128,28,28]     | 295,424    |
| BasicBlock        | 3-5                         | [30,256,14,14]     | 919,040    |
| BasicBlock        | 3-6                         | [30,256,14,14]     | 1,180,672  |
| BasicBlock        | 3-7                         | [30,512, 7, 7]     | 3,673,088  |
| BasicBlock        | 3-8                         | [30,512, 7, 7]     | 4,720,640  |
| AdaptiveAvgPool2d | 2-9                         | [30,512,1,1]       | 0          |
| **LSTM**          | 1-2                         | [1, 30,128]        | 460,800    |
| **FC Head**       | 1-3                         | [1,1]              | --         |
| Linear            | 2-10                        | [1,64]             | 8,256      |
| ReLU              | 2-11                        | [1,64]             | 0          |
| Dropout           | 2-12                        | [1,64]             | 0          |
| Linear            | 2-13                        | [1,1]              | 65         |

---


## 🏋️ Training Process

### Training Settings

- **Loss**: BCEWithLogitsLoss
- **Optimizer**: Adam (lr = 1e-4)
- **Batch size**: 4
- **Epochs**: 20
- **Eearly Stopping**

### Data Augmentation - not used

(After applying data augmentation, it shows that the performance get lower)

- Random horizontal flip
- Random rotation
- Color jitter

### Optimized Hyper Parameter

After trying various hyperparameters, this one performed the best.

- Batchsize : 4
- Learning rate : 1e-4
- LSTM Hidden size : 128
- LSTM Number of layers : 2
- Dropout probability : 0.3

**Only this case can trigger the early stopping.**

---

### Train Result 

The average time of per iteration is around 2.5 sec.


```
Training: 100%|██████████| 155/155 [06:39<00:00,  2.58s/it]
Evaluating: 100%|██████████| 64/64 [02:36<00:00,  2.45s/it]
Epoch 01 | Train Loss: 0.625, Acc: 68.28% | Val Loss: 0.597, Acc: 58.98%

~~~

Training: 100%|██████████| 155/155 [06:40<00:00,  2.58s/it]
Evaluating: 100%|██████████| 64/64 [02:33<00:00,  2.40s/it]
Epoch 17 | Train Loss: 0.403, Acc: 80.42% | Val Loss: 0.445, Acc: 81.25%
```

![image](/output.png)

### Test Result

```
===== Test Metrics =====
Avg Loss: 0.3646
Accuracy:  0.8621
F1 Score:  0.8095
Precision: 0.8293
Recall:    0.7907
```

**The reason the validation loss is high while the test loss remains low is that the validation set does not have a similar class distribution to the test set.**

## Limitaions

### Limited Dataset Size / Imbalanced Data

- The total number of labeled sequences (618 training samples) is relatively small for a deep-learning model combining CNN and LSTM.
This can lead to unstable convergence, higher variance, and sensitivity to hyperparameter choices.

- The validation set has a different class distribution compared to the test set.
This difference causes higher validation loss while the test loss remains low, making validation metrics less reliable for early stopping or hyperparameter tuning.


- Train 
```
===== TRAIN label distribution =====
label
0    425
1    193
Name: count, dtype: int64

===== Percentage (%) =====
label
0    68.77
1    31.23
Name: proportion, dtype: float64

Total samples: 618
```
- Validation
```
===== VALIDATION label distribution =====
label
0    151
1    105
Name: count, dtype: int64

===== Percentage (%) =====
label
0    58.98
1    41.02
Name: proportion, dtype: float64

Total samples: 256
```

- Test
```
===== TEST label distribution =====
label
0    73
1    43
Name: count, dtype: int64

===== Percentage (%) =====
label
0    62.93
1    37.07
Name: proportion, dtype: float64

Total samples: 116
```
