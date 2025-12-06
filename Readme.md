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
- **Metrics**: Accuracy, F1-score, Precision, Recall
- **Eearly Stopping**

### Data Augmentation - not used

(After applying data augmentation, it shows that the performance get lower)

- Random horizontal flip
- Random rotation
- Color jitter

output
'''
Training: 100%|██████████| 155/155 [06:39<00:00,  2.58s/it]
Evaluating: 100%|██████████| 64/64 [02:36<00:00,  2.45s/it]
Epoch 01 | Train Loss: 0.625, Acc: 68.28% | Val Loss: 0.597, Acc: 58.98%
improved
Training: 100%|██████████| 155/155 [06:35<00:00,  2.55s/it]
Evaluating: 100%|██████████| 64/64 [02:34<00:00,  2.41s/it]
Epoch 02 | Train Loss: 0.530, Acc: 74.60% | Val Loss: 0.533, Acc: 73.44%
improved
Training: 100%|██████████| 155/155 [3:06:06<00:00, 72.04s/it]   
Evaluating: 100%|██████████| 64/64 [02:35<00:00,  2.43s/it]
Epoch 03 | Train Loss: 0.481, Acc: 77.35% | Val Loss: 0.500, Acc: 73.44%
improved
Training: 100%|██████████| 155/155 [06:58<00:00,  2.70s/it]
Evaluating: 100%|██████████| 64/64 [02:37<00:00,  2.46s/it]
Epoch 04 | Train Loss: 0.456, Acc: 77.51% | Val Loss: 0.485, Acc: 73.83%
improved
Training: 100%|██████████| 155/155 [06:27<00:00,  2.50s/it]
Evaluating: 100%|██████████| 64/64 [02:28<00:00,  2.33s/it]
Epoch 05 | Train Loss: 0.461, Acc: 77.99% | Val Loss: 0.466, Acc: 76.56%
improved
Training: 100%|██████████| 155/155 [06:26<00:00,  2.49s/it]
Evaluating: 100%|██████████| 64/64 [02:31<00:00,  2.37s/it]
Epoch 06 | Train Loss: 0.431, Acc: 78.32% | Val Loss: 0.450, Acc: 76.56%
improved
Training: 100%|██████████| 155/155 [1:03:23<00:00, 24.54s/it] 
Evaluating: 100%|██████████| 64/64 [1:17:26<00:00, 72.61s/it]   
Epoch 07 | Train Loss: 0.450, Acc: 77.99% | Val Loss: 0.455, Acc: 75.39%
not improved
Training: 100%|██████████| 155/155 [06:25<00:00,  2.49s/it]
Evaluating: 100%|██████████| 64/64 [02:32<00:00,  2.38s/it]
Epoch 08 | Train Loss: 0.422, Acc: 79.45% | Val Loss: 0.490, Acc: 76.17%
not improved
Training: 100%|██████████| 155/155 [06:28<00:00,  2.51s/it]
Evaluating: 100%|██████████| 64/64 [02:30<00:00,  2.35s/it]
Epoch 09 | Train Loss: 0.430, Acc: 78.32% | Val Loss: 0.437, Acc: 80.86%
improved
Training: 100%|██████████| 155/155 [06:25<00:00,  2.49s/it]
Evaluating: 100%|██████████| 64/64 [02:32<00:00,  2.39s/it]
Epoch 10 | Train Loss: 0.403, Acc: 81.23% | Val Loss: 0.422, Acc: 79.69%
improved
Training: 100%|██████████| 155/155 [06:29<00:00,  2.51s/it]
Evaluating: 100%|██████████| 64/64 [02:44<00:00,  2.57s/it]
Epoch 11 | Train Loss: 0.391, Acc: 82.52% | Val Loss: 0.481, Acc: 74.22%
not improved
Training: 100%|██████████| 155/155 [06:32<00:00,  2.53s/it]
Evaluating: 100%|██████████| 64/64 [02:33<00:00,  2.39s/it]
Epoch 12 | Train Loss: 0.402, Acc: 80.58% | Val Loss: 0.414, Acc: 84.38%
improved
Training: 100%|██████████| 155/155 [06:59<00:00,  2.71s/it]
Evaluating: 100%|██████████| 64/64 [02:33<00:00,  2.39s/it]
Epoch 13 | Train Loss: 0.393, Acc: 80.91% | Val Loss: 0.415, Acc: 83.98%
not improved
Training: 100%|██████████| 155/155 [06:28<00:00,  2.50s/it]
Evaluating: 100%|██████████| 64/64 [02:35<00:00,  2.43s/it]
Epoch 14 | Train Loss: 0.417, Acc: 80.58% | Val Loss: 0.464, Acc: 76.95%
not improved
Training: 100%|██████████| 155/155 [06:56<00:00,  2.69s/it]
Evaluating: 100%|██████████| 64/64 [02:33<00:00,  2.40s/it]
Epoch 15 | Train Loss: 0.389, Acc: 82.04% | Val Loss: 0.524, Acc: 76.95%
not improved
Training: 100%|██████████| 155/155 [06:38<00:00,  2.57s/it]
Evaluating: 100%|██████████| 64/64 [02:33<00:00,  2.39s/it]
Epoch 16 | Train Loss: 0.361, Acc: 84.14% | Val Loss: 0.543, Acc: 74.22%
not improved
Training: 100%|██████████| 155/155 [06:40<00:00,  2.58s/it]
Evaluating: 100%|██████████| 64/64 [02:33<00:00,  2.40s/it]
Epoch 17 | Train Loss: 0.403, Acc: 80.42% | Val Loss: 0.445, Acc: 81.25%
not improved
Early stopping triggered!
'''
