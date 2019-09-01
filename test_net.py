# --------------------------------------------------------
# Pytorch Multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import cv2

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import pickle
from roi_data_layer.roibatchLoader import roibatchLoader
from roi_data_layer.nuscenes_dataloader import nuscenes_dataloader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

from utils import sampler,  get_bounding_boxes

from detection_metric.utils import *
from detection_metric.BoundingBox import BoundingBox
from detection_metric.BoundingBoxes import BoundingBoxes
from detection_metric.Evaluator import *

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
  parser.add_argument('--dataset', dest='dataset',
                      help='training dataset',
                      default='pascal_voc', type=str)
  parser.add_argument('--date', dest='date',
                      help='date (folder name)',
                      default='08_28_01', type=str)                    
  parser.add_argument('--cfg', dest='cfg_file',
                      help='optional config file',
                      default='cfgs/vgg16.yml', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res101', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                      help='set config keys', default=None,
                      nargs=argparse.REMAINDER)
  parser.add_argument('--load_dir', dest='load_dir',
                      help='directory to load models', default="models",
                      type=str)
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
                      action='store_true')
  parser.add_argument('--ls', dest='large_scale',
                      help='whether use large imag scale',
                      action='store_true')
  parser.add_argument('--mGPUs', dest='mGPUs',
                      help='whether use multiple GPUs',
                      action='store_true')
  parser.add_argument('--cag', dest='class_agnostic',
                      help='whether perform class_agnostic bbox regression',
                      action='store_true')
  parser.add_argument('--parallel_type', dest='parallel_type',
                      help='which part of model to parallel, 0: all, 1: model before roi pooling',
                      default=0, type=int)
  parser.add_argument('--checksession', dest='checksession',
                      help='checksession to load model',
                      default=1, type=int)
  parser.add_argument('--checkepoch', dest='checkepoch',
                      help='checkepoch to load network',
                      default=8, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=20799, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

nusc_classes = ('__background__', 
                           'pedestrian', 'barrier', 'trafficcone', 'bicycle', 'bus', 'car', 'construction', 'motorcycle', 'trailer', 'truck')
                           
if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

  np.random.seed(cfg.RNG_SEED)
  if args.dataset == "pascal_voc":
      args.imdb_name = "voc_2007_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "pascal_voc_0712":
      args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
      args.imdbval_name = "voc_2007_test"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "coco":
      args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
      args.imdbval_name = "coco_2014_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "imagenet":
      args.imdb_name = "imagenet_train"
      args.imdbval_name = "imagenet_val"
      args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']
  elif args.dataset == "vg":
      args.imdb_name = "vg_150-50-50_minitrain"
      args.imdbval_name = "vg_150-50-50_minival"
      args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]']

  args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  cfg.TRAIN.USE_FLIPPED = False

  # Load trained model 
  cur_dir = os.path.dirname(os.path.abspath(__file__)) 
  input_dir = cur_dir + "/" + args.load_dir + "/" + args.net + "/" + args.dataset + "/" + args.date  
  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(imdb.classes, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(nusc_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(imdb.classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(imdb.classes, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  checkpoint = torch.load(load_name)
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')
  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data)
  im_info = Variable(im_info)
  num_boxes = Variable(num_boxes)
  gt_boxes = Variable(gt_boxes)

  if args.cuda:
    cfg.CUDA = True

  if args.cuda:
    fasterRCNN.cuda()

  start = time.time()
  max_per_image = 100

  vis = args.vis

  if vis:
    thresh = 0.05
  else:
    thresh = 0.0

  save_name = 'faster_rcnn_10'
  output_dir = input_dir
  
  # Setup dataloader for test images 
  dataset = nuscenes_dataloader(1, len(nusc_classes), training = False)
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=0,
                            pin_memory=True)

  num_images = len(dataset)
  batch_size = 1 
  data_iter = iter(dataloader)

  print('{:d} entries'.format(num_images))
  
  _t = {'im_detect': time.time(), 'misc': time.time()}
  det_file = os.path.join(output_dir, 'detections.pkl')

  evaluator = Evaluator()
  test_mAP = 0
  test_AP = []
  f = open(output_dir + '/test_AP.txt','w+')
  f.close()
  allBoundingBoxes = BoundingBoxes()
  
  # No grad for testing 
  with torch.no_grad():
      for step in range(num_images):
          data = next(data_iter)
          with torch.no_grad():
                  im_data.resize_(data[0].size()).copy_(data[0])
                  im_info.resize_(data[1].size()).copy_(data[1])
                  gt_boxes.resize_(data[2].size()).copy_(data[2])
                  num_boxes.resize_(data[3].size()).copy_(data[3])
          
          index = 0 
          
          # Saving gt boxes for mAP evaluation 
          for i in range(0, gt_boxes.size()[0]):
              index = step * batch_size + i 
              for box in gt_boxes[i].cpu().numpy():
                  if box[0]==0.0 and box[1]==0.0 and box[2]==0.0 and box[3]==0.0:
                      break
                  cls_i = int(box[4])
                  box = list(int(np.round(x/ data[1][0][2].item())) for x in box[:4])
                  bb= BoundingBox(index,cls_i,box[0],box[1],box[2],box[3],CoordinatesType.Absolute, None, BBType.GroundTruth, format=BBFormat.XYWH)
                  allBoundingBoxes.addBoundingBox(bb)
          
          det_tic = time.time()
          rois, cls_prob, bbox_pred, \
          rpn_loss_cls, rpn_loss_box, \
          RCNN_loss_cls, RCNN_loss_bbox, \
          rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)
          
          det_toc = time.time()
          detect_time = det_toc - det_tic
          misc_tic = time.time()
          
          # Saving predicted boxes for evaluation 
          for i in range(0, batch_size):
              index = step*batch_size + i
              allBoundingBoxes = get_bounding_boxes(rois[i].unsqueeze(0), cls_prob[i].unsqueeze(0), bbox_pred[i].unsqueeze(0), im_info[i].unsqueeze(0), allBoundingBoxes, index)
                
          misc_toc = time.time()
          nms_time = misc_toc - misc_tic

          sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
              .format(step + 1, num_images, detect_time, nms_time))
          sys.stdout.flush()

  # Evaluation 
  print('Evaluating detections')
  metricsPerClass = evaluator.GetPascalVOCMetrics(allBoundingBoxes, IOUThreshold=0.5)
  pdb.set_trace()
  for mc in metricsPerClass:
      # Get metric values per each class
      c = mc['class']
      average_precision = mc['AP']
      test_mAP = test_mAP + average_precision
      test_AP = test_AP + [average_precision]
      # Print AP per class
      print('%s: %f' % (c, average_precision))
        
  test_mAP = test_mAP/ 10
  print(test_mAP)
  
  # Saving test results 
  f = open(output_dir + '/test_AP.txt','a+')
  f.write("AP by class: %.4f    %.4f    %.4f    %.4f    %.4f    %.4f    %.4f    %.4f    %.4f    %.4f\n" % (test_AP[0], test_AP[1], test_AP[2], test_AP[3], test_AP[4], test_AP[5], test_AP[6], test_AP[7], test_AP[8], test_AP[9]))
  f.write("mAP: %.4f\n" % (test_mAP))
  f.close()

  end = time.time()
  print("test time: %0.4fs" % (end - start))
