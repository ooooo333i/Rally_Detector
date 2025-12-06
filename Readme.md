# Artificial-Intelligence-Fall-2025-AnSeungGi



## Setup - using dataset

1. Clone this repository.
2. generate a new folder 'Data' and download datasets. [Dataset](https://drive.google.com/drive/folders/19a5XDU64GR2ml62koi11TIn2tR8SgR4_?usp=drive_link)
3. Run 'main.ipynb'.


## DataSet

[Dataset-googledrive](https://drive.google.com/drive/folders/19a5XDU64GR2ml62koi11TIn2tR8SgR4_?usp=drive_link)


(it might take much time.)

---

## 🔧 Data Preprocessing - role of 'collecting_data.py'

1. Download YouTube videos using **pytubefix**  
2. Extract frames every *interval seconds*  
3. Resize to **224×224**  
4. Group into sequences of **30 frames**



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


## RES_LSTM Model Summary

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

### **Total Parameters**
| Type                | Count        |
|---------------------|--------------|
| Total params        | **11,645,633** |
| Trainable params    | **469,121**   |
| Frozen params       | **11,176,512** |
| Mult-Adds           | **54.42 GB** |

### **Memory Usage**
| Category                   | Size (MB) |
|---------------------------|-----------|
| Input size                | 18.06     |
| Forward/backward pass     | 1192.21   |
| Params size               | 46.58     |
| **Estimated Total Size**  | **1256.86 MB** |