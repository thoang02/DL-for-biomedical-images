import torch
import numpy as np
import segmentation_models_pytorch as smp
from utils import *
import os
import argparse
from config import Config
import json

def get_args():
    parser = argparse.ArgumentParser('Train')
    parser.add_argument('--data_dir', type=str, default='./',
                        help='Data dir', dest='data_dir')
    parser.add_argument('--path_model', type=str, default='./models',
                        help='path_model', dest='path_model')
    parser.add_argument('--encoder', type=str, default='vgg16_bn',
                        help='encoder', dest='encoder')
    args = parser.parse_args()

    return args

encoders = ['vgg16_bn',
'timm-mobilenetv3_large_100',
'resnet34',
'efficientnet-b4',
'dpn68',
'densenet169']

def test(data_dir, path_model, encoder):
  results = {}
  config = Config()
  best_model_path = path_model
  ENCODER = encoder # backbone used in encoder 
  ENCODER_WEIGHTS = 'imagenet'
  CLASSES = ['erythrocytes', 'spirochaete']
  ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multiclass segmentation

  loss = smp.utils.losses.DiceLoss()
  metrics = [
      smp.utils.metrics.IoU(threshold=0.5),
      smp.utils.metrics.Fscore(threshold=0.5),
      smp.utils.metrics.Recall(threshold=0.5),
      smp.utils.metrics.Precision(threshold=0.5),
  ]

  # create segmentation model with pretrained encoder
  best_model = torch.load(best_model_path)

  preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

  x_test_dir = os.path.join(data_dir, 'imgs')
  y_test_dir = os.path.join(data_dir, 'masks')

  # create test dataset
  test_dataset = Dataset(
      x_test_dir, 
      y_test_dir,
      preprocessing=get_preprocessing(preprocessing_fn),
      classes=CLASSES,
  )

  test_dataloader = DataLoader(test_dataset)
  
  # evaluate model on test set
  test_epoch = smp.utils.train.ValidEpoch(
      model=best_model,
      loss=loss,
      metrics=metrics,
      device=config.DEVICE,
  )

  logs = test_epoch.run(test_dataloader)

  results[encoder] = logs
  
  with open('results.json', 'w') as f:
      json.dump(results, f, indent=4)
  
  return

if __name__ == '__main__':
    args = get_args()
    test(args.data_dir, args.path_model, args.encoder)