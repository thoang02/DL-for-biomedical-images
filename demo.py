import cv2
from utils import *
import torch
import os
import argparse
from config import Config

def get_args():
    parser = argparse.ArgumentParser('Train')
    parser.add_argument('--dir_img', type=str, default='./',
                        help='Data dir', dest='dir_img')
    parser.add_argument('--path_model', type=str, default='./models',
                        help='path_model', dest='path_model')
    args = parser.parse_args()

    return args

def demo(dir_img, path_model):

    cv2_img = cv2.imread(dir_img) # read img
    height, width, _ = cv2_img.shape # get resolution

    best_model = torch.load(path_model) # read model
    # solving
    image = read_1_img(dir_img)
    x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
    pr_mask = best_model.predict(x_tensor)
    pr_mask = (pr_mask.squeeze().cpu().numpy().round())

    dict_contours = {}

    erythrocytes_predicted_mask=pr_mask[1].squeeze(),
    spirochaete_predicted_mask=pr_mask[2].squeeze(),

    erythrocytes_predicted_mask = convert_unit8(erythrocytes_predicted_mask[0])
    spirochaete_predicted_mask = convert_unit8(spirochaete_predicted_mask[0])

    dict_contours['erythrocytes'], _ = cv2.findContours(erythrocytes_predicted_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    dict_contours['spirochaete'], _ = cv2.findContours(spirochaete_predicted_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    out = draw_img(dict_contours, cv2_img)
    #cv2_imshow(out)
    print('Nums of spirochaete:',len(dict_contours['spirochaete']))
    
    cv2.imwrite('prediction.png', out)
    return

if __name__ == '__main__':
    args = get_args()
    demo(args.dir_img, args.path_model)