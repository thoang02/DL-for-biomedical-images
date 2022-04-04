import torch
import numpy as np
import segmentation_models_pytorch as smp
from utils import *
import os
import argparse
from config import Config

encoders = ['vgg16_bn',
    'timm-mobilenetv3_large_100',
    'resnet34',
    'efficientnet-b4',
    'dpn68',
    'densenet169']

def get_args():
    parser = argparse.ArgumentParser('Train')
    parser.add_argument('--data_dir', type=str, default='./',
                        help='Data dir', dest='data_dir')
    parser.add_argument('--data_mode', type=str, default='normal',
                        help='Data mode', dest='data_mode')
    parser.add_argument('--loss_function', type=str, default='focal_dice_loss',
                        help='Loss function', dest='loss_function')
    parser.add_argument('--dir_save', type=str, default='./models',
                        help='Loss function', dest='dir_save')
    args = parser.parse_args()

    return args

def train_normal(data_dir, loss_function, dir_save):
    assert (loss_function in ['dice_loss', 'focal_loss', 'focal_dice_loss'])
    config = Config()
    global encoders
    
    path_save_model = dir_save
    if not os.path.exists(path_save_model):
      os.mkdir(path_save_model)

    for encoder in encoders:
      ENCODER = encoder # backbone used in encoder 
      ENCODER_WEIGHTS = 'imagenet'
      CLASSES = ['erythrocytes', 'spirochaete']
      ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multiclass segmentation
      DEVICE = config.DEVICE
      # create segmentation model with pretrained encoder
      model = smp.UnetPlusPlus(
          encoder_name=ENCODER, 
          encoder_weights=ENCODER_WEIGHTS, 
          classes=len(CLASSES)+1, 
          activation=ACTIVATION,
          # in_channels=3
      )
      preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
          
      x_train_dir = os.path.join(data_dir, 'normal/train/imgs')
      y_train_dir = os.path.join(data_dir, 'normal/train/masks')

      x_valid_dir = os.path.join(data_dir, 'normal/val/imgs')
      y_valid_dir = os.path.join(data_dir, 'normal/val/masks')

      x_test_dir = os.path.join(data_dir, 'normal/test/imgs')
      y_test_dir = os.path.join(data_dir, 'normal/test/masks')
        

      train_dataset = Dataset(
        x_train_dir, 
        y_train_dir,
        preprocessing=get_preprocessing(preprocessing_fn),
        classes=CLASSES,
      )

      valid_dataset = Dataset(
          x_valid_dir, 
          y_valid_dir,
          preprocessing=get_preprocessing(preprocessing_fn),
          classes=CLASSES,
      )

      train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=12)
      valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

      # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
      # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

      if loss_function == 'dice_loss':
          loss = smp.utils.losses.DiceLoss()
      elif loss_function == 'focal_dice_loss':
          loss = smp.losses.FocalDiceLoss(mode='multiclass', tradeoff=0.5)
          loss.__name__ = 'focal_dice_loss'
      elif loss_function == 'focal_loss':
          loss = smp.losses.FocalLoss(mode='multiclass', tradeoff=1)
          loss.__name__ = 'focal_loss'
      
      metrics = [
          smp.utils.metrics.IoU(threshold=0.5),
          smp.utils.metrics.Fscore(threshold=0.5),
          smp.utils.metrics.Recall(threshold=0.5),
          smp.utils.metrics.Precision(threshold=0.5),
      ]

      optimizer = torch.optim.Adam([ 
          dict(params=model.parameters(), lr=0.0001),
      ])

      train_epoch = smp.utils.train.TrainEpoch(
        model, 
        loss=loss, 
        metrics=metrics, 
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
      )

      valid_epoch = smp.utils.train.ValidEpoch(
          model, 
          loss=loss, 
          metrics=metrics, 
          device=DEVICE,
          verbose=True,
      )

      max_score = 0
      EPOCHS = config.EPOCHS

      for i in range(0, EPOCHS):
          print('\nEpoch: {}'.format(i))
          train_logs = train_epoch.run(train_loader)
          valid_logs = valid_epoch.run(valid_loader)

          if max_score < valid_logs['iou_score']:
              max_score = valid_logs['iou_score']
              torch.save(model, os.path.join(path_save_model, encoder + '.pth'))
              print('Model saved!')
              
          if i == 25:
              optimizer.param_groups[0]['lr'] = 1e-5
              print('Decrease decoder learning rate to 1e-5!')

    return

def train_preprocessing(data_dir, loss_function, dir_save):
    assert (loss_function in ['dice_loss', 'focal_loss', 'focal_dice_loss'])
    global encoders
    config = Config()
    path_save_model = dir_save

    if not os.path.exists(path_save_model):
      os.mkdir(path_save_model)

    for encoder in encoders:
      ENCODER = encoder # backbone used in encoder 
      ENCODER_WEIGHTS = 'imagenet'
      CLASSES = ['erythrocytes', 'spirochaete']
      ACTIVATION = 'softmax2d' # could be None for logits or 'softmax2d' for multiclass segmentation
      DEVICE = config.DEVICE

      # create segmentation model with pretrained encoder
      model = smp.UnetPlusPlus(
          encoder_name=ENCODER, 
          encoder_weights=ENCODER_WEIGHTS, 
          classes=len(CLASSES)+1, 
          activation=ACTIVATION,
          # in_channels=3
      )
      preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)

      for pre_processing in kernels_features:
        print('We are training', pre_processing)

        x_train_dir = os.path.join(data_dir, pre_processing, '/train/imgs')
        y_train_dir = os.path.join(data_dir, 'normal/train/masks')

        x_valid_dir = os.path.join(data_dir, pre_processing, '/val/imgs')
        y_valid_dir = os.path.join(data_dir, 'normal/val/masks')

        x_test_dir = os.path.join(data_dir, pre_processing, '/test/imgs')
        y_test_dir = os.path.join(data_dir, 'normal/test/masks')
          

        train_dataset = Dataset(
          x_train_dir, 
          y_train_dir,
          preprocessing=get_preprocessing(preprocessing_fn),
          classes=CLASSES,
        )

        valid_dataset = Dataset(
            x_valid_dir, 
            y_valid_dir,
            preprocessing=get_preprocessing(preprocessing_fn),
            classes=CLASSES,
        )

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=12)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

        # Dice/F1 score - https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
        # IoU/Jaccard score - https://en.wikipedia.org/wiki/Jaccard_index

        if loss_function == 'dice_loss':
            loss = smp.utils.losses.DiceLoss()
        elif loss_function == 'focal_dice_loss':
            loss = smp.losses.FocalDiceLoss(mode='multiclass', tradeoff=0.5)
            loss.__name__ = 'focal_dice_loss'
        elif loss_function == 'focal_loss':
            loss = smp.losses.FocalLoss(mode='multiclass', tradeoff=1)
            loss.__name__ = 'focal_loss'
        
        metrics = [
            smp.utils.metrics.IoU(threshold=0.5),
            smp.utils.metrics.Fscore(threshold=0.5),
            smp.utils.metrics.Recall(threshold=0.5),
            smp.utils.metrics.Precision(threshold=0.5),
        ]

        optimizer = torch.optim.Adam([ 
            dict(params=model.parameters(), lr=0.0001),
        ])

        train_epoch = smp.utils.train.TrainEpoch(
          model, 
          loss=loss, 
          metrics=metrics, 
          optimizer=optimizer,
          device=DEVICE,
          verbose=True,
        )

        valid_epoch = smp.utils.train.ValidEpoch(
            model, 
            loss=loss, 
            metrics=metrics, 
            device=DEVICE,
            verbose=True,
        )

        max_score = 0
        EPOCHS = config.EPOCHS

        for i in range(0, EPOCHS):
            print('\nEpoch: {}'.format(i))
            train_logs = train_epoch.run(train_loader)
            valid_logs = valid_epoch.run(valid_loader)

            if max_score < valid_logs['iou_score']:
                max_score = valid_logs['iou_score']
                torch.save(model, os.path.join(path_save_model, encoder + '_' + pre_processing + '.pth'))
                print('Model saved!')
                
            if i == 25:
                optimizer.param_groups[0]['lr'] = 1e-5
                print('Decrease decoder learning rate to 1e-5!')
    return

if __name__ == '__main__':
    args = get_args()
    assert (args.data_mode in ['normal', 'preprocessing'])
    if args.data_mode == 'normal':
      train_normal(args.data_dir, args.loss_function, args.dir_save)
    elif args.data_mode == 'preprocessing':
      train_preprocessing(args.data_dir, args.loss_function, args.dir_save)