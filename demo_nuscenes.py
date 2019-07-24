# --------------------------------------------------------
# Tensorflow Faster R-CNN
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
import matplotlib.pyplot as plt 
import matplotlib.patches as patches 

import torchvision.transforms as transforms
import torchvision.datasets as dset
from imageio import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes

# from model.nms.nms_wrapper import nms
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
import pdb

from nuscenes import NuScenes 
from nuscenes import NuScenesExplorer 
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility

nusc = NuScenes(version='v1.0-mini', dataroot='/data/sets/nuscenes', verbose=True)
explorer = NuScenesExplorer(nusc)

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
                      help='directory to load models',
                      default="/srv/share/jyang375/models")
  parser.add_argument('--image_dir', dest='image_dir',
                      help='directory to load images for demo',
                      default="images")
  parser.add_argument('--cuda', dest='cuda',
                      help='whether use CUDA',
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
                      default=1, type=int)
  parser.add_argument('--checkpoint', dest='checkpoint',
                      help='checkpoint to load network',
                      default=10021, type=int)
  parser.add_argument('--bs', dest='batch_size',
                      help='batch_size',
                      default=1, type=int)
  parser.add_argument('--vis', dest='vis',
                      help='visualization mode',
                      action='store_true')
  parser.add_argument('--webcam_num', dest='webcam_num',
                      help='webcam ID number',
                      default=-1, type=int)
  parser.add_argument('--nuscenes', dest='nuscenes',
                      help='using nuscenes library or not',
                      default=False, type=bool)
  args = parser.parse_args()
  return args

lr = cfg.TRAIN.LEARNING_RATE
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY

def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)

def get_boxes(im_token):
  global nusc
  classes = ('__background__', 
                           'pedestrian', 'barrier', 'trafficcone', 'bicycle', 'bus', 'car', 'construction', 'motorcycle', 'trailer', 'truck')
  data_path, boxes, camera_intrinsic = nusc.get_sample_data(im_token, box_vis_level=BoxVisibility.ANY)
  gt_boxes = np.zeros((50, 5), dtype=np.float32)
  i = 0
  for box in boxes:
      if i == 50:
          break
      corners= view_points(box.corners(), view=camera_intrinsic, normalize=True)[:2,:]

      gt_boxes[i,0]= np.min(corners[0])
      gt_boxes[i,1]= np.min(corners[1])
      gt_boxes[i,2]= np.max(corners[0])
      gt_boxes[i,3]= np.max(corners[1])
      if box.name.split('.')[0] == 'vehicle':
          if box.name.split('.')[1] != 'emergency':
              name = box.name.split('.')[1]
          else:
              name = ''
      elif box.name.split('.')[0] == 'human':
          name = 'pedestrian'
      elif box.name.split('.')[0] == 'movable_object':
          if box.name.split('.')[1] != 'debris' and box.name.split('.')[1] != 'pushable_pullable': 
              name = box.name.split('.')[1]
          else:
              name = ''
      else:
          name = ''
      if name != '': 
          gt_boxes[i,4]=classes.index(name)
          i += 1 
  return gt_boxes

def predict_boxes(cls_prob):
  thresh = 0.5
  class_prob = cls_prob.cpu().numpy()[:,1:]

  # get frequency of classes in predictions 
  mask = (class_prob.max(axis=1) > thresh)
  pred_classes = np.argmax(class_prob, axis=1)[mask]
  #pdb.set_trace()
  return (pred_classes + 1) 

def get_accuracy(all_boxes, gtruth_boxes):
  #pdb.set_trace()
  classes = ('__background__', 
                           'pedestrian', 'barrier', 'trafficcone', 'bicycle', 'bus', 'car', 'construction', 'motorcycle', 'trailer', 'truck')
  correct_count = np.zeros(len(classes))
  actual_count = np.zeros(len(classes))
  for cls_i in range(len(classes)):
    actual_count[cls_i] = actual_count[cls_i] + len(gtruth_boxes[cls_i])
    if len(gtruth_boxes[cls_i])>0: 
      print(classes[cls_i])
      correct_count[cls_i] = correct_count[cls_i] + get_correct_count(all_boxes[cls_i], gtruth_boxes[cls_i]) 
  print(correct_count)
  print(actual_count)
  return correct_count/actual_count

def get_correct_count(pred, truth):
  #pdb.set_trace()
  counter = 0
  for box in truth:
    if len(pred)!=0:
      """
      fig, ax = plt.subplots(1)
      ax.add_patch(patches.Rectangle((box[0],box[1]),(box[2]-box[0]), (box[3]-box[1]), linewidth=1, edgecolor='b', facecolor ='none'))
      ax.set_xlim([0,1600])
      ax.set_ylim([0,900])
      """
      for pbox in pred:
        #ax.add_patch(patches.Rectangle((pbox[0],pbox[1]),(pbox[2]-pbox[0]), (pbox[3]-pbox[1]), linewidth=1, edgecolor='r', facecolor ='none'))
        if iou(box, pbox)>0.3:
          counter+=1
          break
      #plt.show()
  return counter

def iou(boxA, boxB): 
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

  boxAArea = (boxA[2] - boxA[0] + 1)* (boxA[3]- boxA[1] +1)
  boxBArea = (boxB[2] - boxB[0] + 1)* (boxB[3]- boxB[1] +1)

  iou = interArea/ float(boxAArea + boxBArea - interArea)
  return iou 

if __name__ == '__main__':

  args = parse_args()

  print('Called with args:')
  print(args)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  cfg.USE_GPU_NMS = args.cuda

  print('Using config:')
  pprint.pprint(cfg)
  np.random.seed(cfg.RNG_SEED)

  # train set
  # -- Note: Use validation set and disable the flipped to enable faster loading.
  root_dir = os.path.dirname(os.path.abspath(__file__))
  input_dir = root_dir + "/data/trained_model/res101/nuscenes/diffcam/07_23_01"

  if not os.path.exists(input_dir):
    raise Exception('There is no input directory for loading network from ' + input_dir)
  load_name = os.path.join(input_dir,
    'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))

  pascal_classes = np.asarray(['__background__',
                       'aeroplane', 'bicycle', 'bird', 'boat',
                       'bottle', 'bus', 'car', 'cat', 'chair',
                       'cow', 'diningtable', 'dog', 'horse',
                       'motorbike', 'person', 'pottedplant',
                       'sheep', 'sofa', 'train', 'tvmonitor'])

  nuscenes_classes = np.asarray(['__background__', 
                           'pedestrian', 'barrier', 'trafficcone', 'bicycle', 'bus', 'car', 'construction', 'motorcycle', 'trailer', 'truck'])
  # initilize the network here.
  if args.net == 'vgg16':
    fasterRCNN = vgg16(nuscenes_classes, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res101':
    fasterRCNN = resnet(nuscenes_classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res50':
    fasterRCNN = resnet(nuscenes_classes, 50, pretrained=False, class_agnostic=args.class_agnostic)
  elif args.net == 'res152':
    fasterRCNN = resnet(nuscenes_pclasses, 152, pretrained=False, class_agnostic=args.class_agnostic)
  else:
    print("network is not defined")
    pdb.set_trace()

  fasterRCNN.create_architecture()

  print("load checkpoint %s" % (load_name))
  if args.cuda > 0:
    checkpoint = torch.load(load_name)
  else:
    checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
  fasterRCNN.load_state_dict(checkpoint['model'])
  if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']


  print('load model successfully!')

  # pdb.set_trace()

  print("load checkpoint %s" % (load_name))

  # initilize the tensor holder here.
  im_data = torch.FloatTensor(1)
  im_info = torch.FloatTensor(1)
  num_boxes = torch.LongTensor(1)
  gt_boxes = torch.FloatTensor(1)

  # ship to cuda
  if args.cuda > 0:
    im_data = im_data.cuda()
    im_info = im_info.cuda()
    num_boxes = num_boxes.cuda()
    gt_boxes = gt_boxes.cuda()

  # make variable
  im_data = Variable(im_data, volatile=True)
  im_info = Variable(im_info, volatile=True)
  num_boxes = Variable(num_boxes, volatile=True)
  gt_boxes = Variable(gt_boxes, volatile=True)

  if args.cuda > 0:
    cfg.CUDA = True

  if args.cuda > 0:
    fasterRCNN.cuda()

  fasterRCNN.eval()

  start = time.time()
  max_per_image = 100
  thresh = 0.05
  vis = True

  imglist = []
  PATH = '/data/sets/nuscenes/train_mini_diffcam.txt'
  with open(PATH) as f:
      image_token = [x.strip() for x in f.readlines()]
  for im_token in image_token: 
      sample_data = nusc.get('sample_data', im_token)
      sample_token = sample_data['sample_token']
      sample = nusc.get('sample', sample_token)
      image_name = sample_data['filename']
      imglist += [image_name[8:]]
  num_images = len(imglist)

  print('Loaded Photo: {} images.'.format(num_images))

  while (num_images >= 0):
      all_boxes = [[] for _ in range(len(nuscenes_classes))]
      gtruth_boxes = [[] for _ in range(len(nuscenes_classes))]
      total_tic = time.time()
      num_images -= 1

      im_file = os.path.join(args.image_dir, imglist[num_images])
      im_in = np.array(imread(im_file))
      truth_boxes = get_boxes(image_token[num_images])
      for box in truth_boxes:
          if box[0]==0.0 and box[1]==0.0 and box[2]==0.0 and box[3]==0.0:
              break
          box = list(int(np.round(x)) for x in box)
          gtruth_boxes[int(box[4])] = gtruth_boxes[int(box[4])] + [box[:4]]

      if len(im_in.shape) == 2:
        im_in = im_in[:,:,np.newaxis]
        im_in = np.concatenate((im_in,im_in,im_in), axis=2)
      # rgb -> bgr
      im = im_in[:,:,::-1]

      blobs, im_scales = _get_image_blob(im)
      assert len(im_scales) == 1, "Only single-image batch implemented"
      im_blob = blobs
      im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

      im_data_pt = torch.from_numpy(im_blob)
      im_data_pt = im_data_pt.permute(0, 3, 1, 2)
      im_info_pt = torch.from_numpy(im_info_np)

      with torch.no_grad():
              im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
              im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
              gt_boxes.resize_(1, 1, 5).zero_()
              num_boxes.resize_(1).zero_()
    

      # pdb.set_trace()
      det_tic = time.time()

      rois, cls_prob, bbox_pred, \
      rpn_loss_cls, rpn_loss_box, \
      RCNN_loss_cls, RCNN_loss_bbox, \
      rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

      #pdb.set_trace()

      scores = cls_prob.data
      boxes = rois.data[:, :, 1:5]

      predicted_boxes = predict_boxes(scores.squeeze()) 

      if cfg.TEST.BBOX_REG:
          # Apply bounding-box regression deltas
          box_deltas = bbox_pred.data
          if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
          # Optionally normalize targets by a precomputed mean and stdev
            if args.class_agnostic:
                if args.cuda > 0:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)

                box_deltas = box_deltas.view(1, -1, 4)
            else:
                if args.cuda > 0:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS) \
                               + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS)
                box_deltas = box_deltas.view(1, -1, 4 * len(nuscenes_classes))
          #pdb.set_trace()
          pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
          pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
      else:
          # Simply repeat the boxes, once for each class
          pred_boxes = np.tile(boxes, (1, scores.shape[1]))

      pred_boxes /= im_scales[0]

      scores = scores.squeeze()
      pred_boxes = pred_boxes.squeeze()
      det_toc = time.time()
      detect_time = det_toc - det_tic
      misc_tic = time.time()
      
      if vis:
          im2show = np.copy(im)
      bounding_boxes = []
      for j in xrange(1, len(nuscenes_classes)):
          inds = torch.nonzero(scores[:,j]).view(-1)#scores[:,j]>thresh).view(-1)
          # if there is det
          if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            if args.class_agnostic:
              cls_boxes = pred_boxes[inds, :]
            else:
              cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
            cls_dets = cls_dets[order]
            prev_cls_dets = cls_dets
            # keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
            keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
            cls_dets = cls_dets[keep.view(-1).long()]
            #pdb.set_trace()
            dets = cls_dets.cpu().numpy()
            for i in range(np.minimum(10, dets.shape[0])):
                bbox = list(int(np.round(x)) for x in cls_dets.cpu().numpy()[i, :4])
                bbox = bbox + [j]
                score = dets[i,-1]
                if score>0.5:
                   bounding_boxes += [bbox]
            if vis:
              im2show = vis_detections(im2show, nuscenes_classes[j], cls_dets.cpu().numpy(), 0.5)
            #all_boxes[j] = cls_dets.cpu().numpy()
      
      for box in bounding_boxes:
          if len(box)!=0:
              print(nuscenes_classes[box[4]])
          all_boxes[box[4]] = all_boxes[box[4]] + [box[:4]]
      print(all_boxes)
      print(get_accuracy(all_boxes, gtruth_boxes))
      #pdb.set_trace()
      plt.figure()
      plt.imshow(im2show)
      plt.show()
      
      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                       .format(num_images + 1, len(imglist), detect_time, nms_time))
      sys.stdout.flush()

      # cv2.imshow('test', im2show)
      # cv2.waitKey(0)
      #result_path = os.path.join("images", imglist[num_images][:-4] + "_det.jpg")
      #cv2.imwrite(result_path, im2show)


