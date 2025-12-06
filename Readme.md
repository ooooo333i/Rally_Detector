# Artificial-Intelligence-Fall-2025-AnSeungGi



## Setup

1. Clone this repository.
2. generate a new folder 'Data' and download datasets. [Dataset](https://drive.google.com/drive/folders/19a5XDU64GR2ml62koi11TIn2tR8SgR4_?usp=drive_link)
3. Run 'main.ipynb'.


## DataSet

Download here.[Dataset-googledrive](https://drive.google.com/drive/folders/19a5XDU64GR2ml62koi11TIn2tR8SgR4_?usp=drive_link)


(it might take much time.)

---

### 🔧 Data Preprocessing - role of 'collecting_data.py'

1. Download YouTube videos using **pytubefix**  
2. Extract frames every *interval seconds*  
3. Resize to **224×224**  
4. Group into sequences of **30 frames**
5. Manually label using 'labeling.py'

### structure

```
/Data  # make a new 'Data' directory
    /test
        /sequences
            /seq_#####
                /#####.jpg
                ···
        test.csv
    /train
        /sequences
            /seq_#####
                /#####.jpg
                ···
        train.csv
    /validatioon
        /sequences
            /seq_#####
                /#####.jpg
                ···
        validation.csv
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

### Test Result

```
===== Test Metrics =====
Avg Loss: 0.3646
Accuracy:  0.8621
F1 Score:  0.8095
Precision: 0.8293
Recall:    0.7907
```
