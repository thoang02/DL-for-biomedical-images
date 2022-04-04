# Pre-processing the images and saving the image 
import glob
import cv2
import shutil
import os
from tqdm import tqdm
import argparse
from utils import *

def get_args():
    parser = argparse.ArgumentParser('Train')
    parser.add_argument('--data_dir', type=str, default='./',
                        help='Data dir', dest='data_dir')
    parser.add_argument('--dir_save', type=str, default='./processing_data',
                        help='processing_dir', dest='dir_save')
    args = parser.parse_args()

    return args

def processing(data_dir, processing_dir):
  all_images = glob.glob(os.path.join(data_dir, '/**/imgs/**.png'))
  for pre_precessing in kernels_features:
    
    dir_save = os.path.join(processing_dir, pre_precessing)
    
    if not os.path.exists(dir_save):
      os.mkdir(dir_save)

    print("We are applying ", pre_precessing)
    for img in tqdm(all_images):
      image = cv2.imread(img, 0)

      if pre_precessing in ['edge_detection_horizontal', 'edge_detection_vertical', 'Emboss']:
          image = cv2.filter2D(src=image, kernel=kernels_features[pre_precessing][0], ddepth=-1)

      if pre_precessing in ['gradient_magnitude', 'sobel_gradient_magnitude', 'sharpening']:
          img_x = cv2.filter2D(src=image, kernel=kernels_features[pre_precessing][0], ddepth=-1)
          img_y = cv2.filter2D(src=image, kernel=kernels_features[pre_precessing][1], ddepth=-1)
          G_image = np.sqrt(np.square(img_x) + np.square(img_y))
          image = G_image * 255.0 / G_image.max()

      if pre_precessing in ['gradient_direction', 'sobel_gradient_direction']:
          img_x = cv2.filter2D(src=image, kernel=kernels_features[pre_precessing][0], ddepth=-1)
          img_y = cv2.filter2D(src=image, kernel=kernels_features[pre_precessing][1], ddepth=-1)
          G_image = np.arctan(img_x) / np.arctan(img_y)
          image = G_image * 255.0 / G_image.max()

      if pre_precessing == 'guassian_blur':
          image = cv2.GaussianBlur(image, (7, 7), 0)
      # save file

      dir_split_data = os.path.join(dir_save, img.split('/')[-3])
      dir_split_data_imgs = os.path.join(dir_split_data, 'imgs')
      dir_split_data_masks = os.path.join(dir_split_data, 'masks')
      
      if not os.path.exists(dir_split_data):
        os.mkdir(dir_split_data)
      if not os.path.exists(dir_split_data_imgs):
        os.mkdir(dir_split_data_imgs)
      if not os.path.exists(dir_split_data_masks):
        os.mkdir(dir_split_data_masks)
      
      image = np.stack((image,)*3, axis=-1)
      # save image
      cv2.imwrite(os.path.join(dir_split_data_imgs, img.split('/')[-1]), np.float32(image))

if __name__ == '__main__':
    args = get_args()
    processing(args.data_dir, args.dir_save)