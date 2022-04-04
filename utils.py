import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import albumentations as albu
import segmentation_models_pytorch as smp
import numpy as np

ENCODER = 'vgg16_bn' # backbone sử dụng trong encoder
ENCODER_WEIGHTS = 'imagenet'
CLASSES = ['erythrocytes', 'spirochaete']
ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multiclass segmentation
DEVICE = 'cuda'
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

# Preprocessor kernels 
kernels_features = {
    'edge_detection_horizontal': [np.array([
        [-1,0,1],
        [-1,0,1],
        [-1,0,1]
    ])],
    'edge_detection_vertical': [np.array([
        [-1,-1,-1],
        [0,0,0],
        [1,1,1]
    ])],
    'gradient_magnitude': [np.array([
        [-1,-1,-1],
        [0,0,0],
        [1,1,1]
    ]),
        np.array([
        [-1,0,1],
        [-1,0,1],
        [-1,0,1]
    ])],
    'gradient_direction': [np.array([
        [-1,-1,-1],
        [0,0,0],
        [1,1,1]
        ]),
        np.array([
        [-1,0,1],
        [-1,0,1],
        [-1,0,1]
    ])],
    'sobel_gradient_magnitude': [np.array([
        [1,2,1],
        [0,0,0],
        [-1,-2,-1]
    ]),
    np.array([
        [1,0,-1],
        [2,0,-2],
        [1,0,-1]
    ])],
    'sobel_gradient_direction': [np.array([
        [1,2,1],
        [0,0,0],
        [-1,-2,-1]
    ]),
    np.array([
        [1,0,-1],
        [2,0,-2],
        [1,0,-1]
    ])],
    'gaussian_blur': [],
    'sharpening': [np.array([
        [1,1,1],
        [0,0,0],
        [-1,-1,-1]
    ]),
    np.array([
        [1,0,-1],
        [1,0,-1],
        [1,0,-1]
    ])],
    'emboss': [np.array([
        [-1,-1,0],
        [-1,0,1],
        [0,1,1]
    ])],
    'super_pixel': []
}

class Dataset(BaseDataset):
    """
    Args:
        images_dir (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline 
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing 
            (e.g. noralization, shape manipulation, etc.)
    
    """
    
    CLASSES = ['erythrocytes', 'spirochaete']
    
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            classes=None, 
            augmentation=None, 
            preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]
        
        # convert str names to class values on masks
        self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        
        # read data
        image = cv2.imread(self.images_fps[i])
        try:
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
          print(self.images_fps[i],' error')
        #image = cv2.resize(image, (512, 512))
        mask = cv2.imread(self.masks_fps[i], 0)
        #mask = cv2.resize(mask, (512, 512))
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')
        
        # add background if mask is not binary
        if mask.shape[-1] != 1:
            background = 1 - mask.sum(axis=-1, keepdims=True)
            mask = np.concatenate((mask, background), axis=-1)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        
    def __len__(self):
        return len(self.ids)

# Function used to enhance data 
def get_training_augmentation():
    train_transform = [

        albu.HorizontalFlip(p=0.5),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),

        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
        albu.RandomCrop(height=320, width=320, always_apply=True),

        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),

        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
    ]
    return albu.Compose(train_transform)

# Function used to augment data on the set val 
def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.PadIfNeeded(384, 480)
    ]
    return albu.Compose(test_transform)

# Convert from numpy to tensor 
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def draw_img(dict_counters, img):
  colors = {
      'erythrocytes': (255, 0, 0), 'spirochaete': (0, 255, 0)
  }
  for cls in dict_counters:
    for i, cnt in enumerate(dict_counters[cls]):
      cv2.drawContours(img, [cnt], -1, colors[cls], 2)
  return img

def convert_unit8(I):
    I = I*255
    return I.astype(np.uint8)

# Đổi từ numpy sang dạng tensor
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def read_1_img(image_dir):
    global preprocessing_fn
    global model
    # read data
    preprocessing = get_preprocessing(preprocessing_fn)
    
    CLASSES = ['erythrocytes', 'spirochaete']

    class_values = [CLASSES.index(cls.lower()) for cls in CLASSES]

    image = cv2.imread(image_dir)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width, _ = image.shape
    _ = np.zeros((height, width))
    masks = [(_ == v) for v in class_values]
    mask = np.stack(masks, axis=-1).astype('float')

    # apply preprocessing
    sample = preprocessing(image=image, mask=mask)
    image, mask = sample['image'], sample['mask']
    return image
    return image, mask