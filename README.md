# Project_DL-for-Biomedical-Images

Dissertation Source Code

Deep learning models for Biomedical Images- Fighting class imbalance and small dataset problem

Task: Bacteria detection in Blood sample

Dataset: 366 darkfield microscope images and masks that were manually annoted

Data source: https://www.kaggle.com/longnguyen2306/bacteria-detection-with-darkfield-microscopy

This project use:
- Unet++ state-of-art model: https://github.com/MrGiovanni/UNetPlusPlus 
- Python library with Neural Networks for Image Segmentation based on PyTorch: https://github.com/qubvel/segmentation_models.pytorch

Projec objectives and contributions:

1. Apply CNN Unet++ to detect spirochaete bacteria in microscope images of blood samples. Count the number of bacteria in the positive samples

2. Attempt to solve problem of small dataset by 2 methods:
- Experimenting different pre-trained backbones: VGG16, Timm mobilenet, Resnet, EfficientNet, DenseNet, Dpn. 
- Applying data augmentation techniques: Random Rotate, Grid Distortion, Horizontal Flip, Vertical Flip 

3. Attempt to fight class imbalance by introducing new loss fuction (created by combining Dice Loss and Focal Loss). Formula as in Project report. 

4. Evaluate results to find out the best combination that can have valuable application, especially with real-time data. 

5. Develop web app with the best model that allow users to upload images and download segmented results


# For training and some functions

## 0. Requirements

```
pip install -r requirements.txt
pip install -U git+https://github.com/albu/albumentations --no-cache-dir
```

## 1. Training

```
python train.py \
--data_dir [data path] \
--data_mode [normal or preprocessing] \
--loss_function [focal_loss or dice_loss or focal_dice_loss] \
```

Example:
```
python train.py \
--data_dir '/content/drive/MyDrive/segmentation_models.pytorch/data' \
--data_mode 'normal' \
--loss_function 'focal_dice_loss' \
```

## 2. Testing

```
python test.py \
--data_dir [test data path]
--path_model [path model]
--encoder [encoder]
```

Example:
```
python test.py \
--data_dir '/content/drive/MyDrive/segmentation_models.pytorch/data/normal/test' \
--path_model '/content/drive/MyDrive/segmentation_models.pytorch/models/vgg16_bn_normal.pth' \
--encoder 'vgg16_bn'
```

## 3. Demo
```
python demo.py \
--dir_img [img path] \
--path_model [model path]
```

Example:
```
python demo.py \
--dir_img '/content/drive/MyDrive/segmentation_models.pytorch/data/normal/test/imgs/033.png' \
--path_model '/content/drive/MyDrive/segmentation_models.pytorch/models/vgg16_bn_normal.pth'
```

Results:

![](https://i.imgur.com/mt7oefm.png)
```
Nums of spirochaete: 3
```

## 4. Augmentation

```
python augment.py \
--data_dir [data path] \
--dir_save [save path]
```

```
python augment.py \
--data_dir '/content/drive/MyDrive/segmentation_models.pytorch/data/normal/test' \
--dir_save './data_augment'
```

## 5. Preprocessing data
```
python preprocessing.py \
--data_dir [data path] \
--dir_save [save path]
```

```
python preprocessing.py \
--data_dir '/content/drive/MyDrive/segmentation_models.pytorch/data/normal/test' \
--dir_save './processed_data'
```

# Demo webapp
```
%cd Project_appdemo
```

Step-by-step:
1. Put model into `models` folder
2. `pip install requirements.txt`
3. `python server.py`

Output:

![](https://i.imgur.com/0CVgrU8.png)

