import os
import cv2
from tqdm import tqdm
from glob import glob
from albumentations import CenterCrop, RandomRotate90, GridDistortion, HorizontalFlip, VerticalFlip
import argparse

def get_args():
    parser = argparse.ArgumentParser('Train')
    parser.add_argument('--data_dir', type=str, default='./',
                        help='Data dir', dest='data_dir')
    parser.add_argument('--dir_save', type=str, default='./data_augment',
                        help='', dest='dir_save')
    args = parser.parse_args()

    return args

def load_data(path):
    images = sorted(glob(os.path.join(path, "imgs/*")))
    masks = sorted(glob(os.path.join(path, "masks/*")))
    return images, masks

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def augment_data(images, masks, save_path, augment=True):
    H = 512
    W = 512

    for x, y in tqdm(zip(images, masks), total=len(images)):
        name = x.split("/")[-1].split(".")
        """ Extracting the name and extension of the image and the mask. """
        image_name = name[0]
        image_extn = name[1]

        name = y.split("/")[-1].split(".")
        mask_name = name[0]
        mask_extn = name[1]

        """ Reading image and mask. """
        x = cv2.imread(x, cv2.IMREAD_COLOR)
        y = cv2.imread(y, cv2.IMREAD_COLOR)

        """ Augmentation """
        if augment == True:
            # aug = CenterCrop(H, W, p=1.0)
            # augmented = aug(image=x, mask=y)
            # x1 = augmented["image"]
            # y1 = augmented["mask"]

            aug = RandomRotate90(p=1.0)
            augmented = aug(image=x, mask=y)
            x2 = augmented['image']
            y2 = augmented['mask']

            aug = GridDistortion(p=1.0)
            augmented = aug(image=x, mask=y)
            x3 = augmented['image']
            y3 = augmented['mask']

            aug = HorizontalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x4 = augmented['image']
            y4 = augmented['mask']

            aug = VerticalFlip(p=1.0)
            augmented = aug(image=x, mask=y)
            x5 = augmented['image']
            y5 = augmented['mask']

            save_images = [x, x2, x3, x4, x5]
            save_masks =  [y, y2, y3, y4, y5]

        else:
            save_images = [x]
            save_masks = [y]

        """ Saving the image and mask. """
        idx = 0
        for i, m in zip(save_images, save_masks):
            try:
              i = cv2.resize(i, (W, H))
              m = cv2.resize(m, (W, H))

              if len(images) == 1:
                  tmp_img_name = f"{image_name}.{image_extn}"
                  tmp_mask_name = f"{mask_name}.{mask_extn}"
              else:
                  tmp_img_name = f"{image_name}_{idx}.{image_extn}"
                  tmp_mask_name = f"{mask_name}_{idx}.{mask_extn}"

              image_path = os.path.join(save_path, "images", tmp_img_name)
              mask_path = os.path.join(save_path, "masks", tmp_mask_name)

              cv2.imwrite(image_path, i)
              cv2.imwrite(mask_path, m)

              idx += 1
            except:
              print('miss')

def augment(data_dir, save_dir):
    """ Loading original images and masks. """
    path = data_dir
    images, masks = load_data(path)
    print(f"Original Images: {len(images)} - Original Masks: {len(masks)}")

    """ Creating folders. """
    create_dir(save_dir)
    create_dir(save_dir + '/imgs/')
    create_dir(save_dir + '/masks/')

    """ Applying data augmentation. """
    augment_data(images, masks, save_dir, augment=True)

    """ Loading augmented images and masks. """
    images, masks = load_data(save_dir)
    print(f"Augmented Images: {len(images)} - Augmented Masks: {len(masks)}")


if __name__ == '__main__':
    args = get_args()
    augment(args.data_dir, args.dir_save)