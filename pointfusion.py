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
import numpy.random as npr
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

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
import torchvision.datasets as dset
import torchvision.models as models
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
from model import pointnet 
import pdb

from typing import Tuple, List
import os.path as osp
from pyquaternion import Quaternion
from PIL import Image

from nuscenes import NuScenes 
from nuscenes import NuScenesExplorer 
from roi_data_layer.nuscenes_dataloader import nuscenes_dataloader
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

def preprocess_image(im):
  scale_inds = npr.randint(0, high=len(cfg.TRAIN.SCALES),
                  size=1)
  if len(im.shape) == 2:
    im = im[:,:,np.newaxis]
    im = np.concatenate((im,im,im), axis=2)
  # flip the channel, since the original one using cv2
  # rgb -> bgr
  im = im[:,:,::-1]
  pixel_means = cfg.PIXEL_MEANS
  target_size = cfg.TRAIN.SCALES[scale_inds[0]]   
  im = im.astype(np.float32, copy=False)
  im -= pixel_means
  im_shape = im.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])
  im_scale = float(target_size) / float(im_size_min)
  #print("Image size: " + str(im.shape))
  im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
  #print("Image size: " + str(im.shape))
  im = im.transpose(2,0,1)
  return im 

class sampler(Sampler):
  def __init__(self, train_size, batch_size):
    self.num_data = train_size
    self.num_per_batch = int(train_size / batch_size)
    self.batch_size = batch_size
    self.range = torch.arange(0,batch_size).view(1, batch_size).long()
    self.leftover_flag = False
    if train_size % batch_size:
      self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
      self.leftover_flag = True

  def __iter__(self):
    rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
    self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

    self.rand_num_view = self.rand_num.view(-1)

    if self.leftover_flag:
      self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

    return iter(self.rand_num_view)

  def __len__(self):
    return self.num_data

class ResNet50Bottom(nn.Module):
    def __init__(self, original_model):
        super(ResNet50Bottom, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        
    def forward(self, x):
        x = self.features(x)
        return x

def get_pointcloud(bbox, pointsensor_token: str, camera_token: str, min_dist: float = 1.0) -> Tuple:
  """
  Given a point sensor (lidar/radar) token and camera sample_data token, load point-cloud and map it to the image
  plane.
  :param pointsensor_token: Lidar/radar sample_data token.
  :param camera_token: Camera sample_data token.
  :param min_dist: Distance from the camera below which points are discarded.
  :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
  """

  cam = self.nusc.get('sample_data', camera_token)
  pointsensor = self.nusc.get('sample_data', pointsensor_token)
  pcl_path = osp.join(self.nusc.dataroot, pointsensor['filename'])
  pc = LidarPointCloud.from_file(pcl_path)
  original_pc = np.copy(pc)
  im = Image.open(osp.join(self.nusc.dataroot, cam['filename']))
  bottom_left = bbox[0:2]
  top_right = bbox[2:4] 

  # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
  # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
  cs_record = self.nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
  pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
  pc.translate(np.array(cs_record['translation']))

  # Second step: transform to the global frame.
  poserecord = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
  pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
  pc.translate(np.array(poserecord['translation']))

  # Third step: transform into the ego vehicle frame for the timestamp of the image.
  poserecord = self.nusc.get('ego_pose', cam['ego_pose_token'])
  pc.translate(-np.array(poserecord['translation']))
  pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

  # Fourth step: transform into the camera.
  cs_record = self.nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
  pc.translate(-np.array(cs_record['translation']))
  pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

  # Fifth step: actually take a "picture" of the point cloud.
  # Grab the depths (camera frame z axis points away from the camera).
  depths = pc.points[2, :]

  # Retrieve the color from the depth.
  coloring = depths

  # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
  points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
  crop_points = pc.points[:3, :]
  # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
  # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
  # casing for non-keyframes which are slightly out of sync.
  mask = np.ones(depths.shape[0], dtype=bool)
  mask = np.logical_and(mask, depths > min_dist)
  mask = np.logical_and(mask, points[0, :] > 1)
  mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
  mask = np.logical_and(mask, points[1, :] > 1)
  mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
  points = points[:, mask]
  crop_points = points[:,mask]
  points = points[bottom_left[1]:top_right[1],bottom_left[0]:top_right[0]]

  coloring = coloring[mask]

  return points, coloring, im

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

  input_dir = "/home/julia/faster-rcnn.pytorch/data/pretrained_model"
  #args.load_dir + "/" + args.net + "/" + args.dataset
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

  #nusc_classes = ('__background__',  # always index 0
   #                      'animal', 'human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.construction_worker', 'human.pedestrian.personal_mobility', 'human.pedestrian.police_officer', 'human.pedestrian.stroller', 'human.pedestrian.wheelchair', 'movable_object.barrier', 'movable_object.debris', 'movable_object.pushable_pullable', 'movable_object.trafficcone', 'vehicle.bicycle', 'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.car', 'vehicle.construction', 'vehicle.emergency.ambulance', 'vehicle.emergency.police', 'vehicle.motorcycle', 'vehicle.trailer', 'vehicle.truck', 'static_object.bicycle_rack',)

  nuscenes_classes = np.asarray(['__background__',  # always index 0
                         'animal', 'human', 'movable_object', 'bicycle', 'bus', 'car', 'construction', 'emergency', 'motorcycle', 'trailer', 'truck', 'static_object'])

  # initilize the network here.
  fasterRCNN = resnet(pascal_classess, 101, pretrained=False, class_agnostic=args.class_agnostic)
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
  print("load checkpoint %s" % (load_name))

  output_dir = "/home/julia/faster-rcnn.pytorch/data/trained_model/pointfusion" + args.net + "/" + args.dataset
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  nusc_sampler_batch = sampler(317, args.batch_size)
  nusc_set = nuscenes_dataloader(1, len(nusc_classes), training = True)
  nusc_dataloader = torch.utils.data.DataLoader(nusc_set, batch_size = 1, sampler = nusc_sampler_batch, num_workers = 0)
  nusc_iters_per_epoch = int(len(nusc_set) / args.batch_size)

  # initilize the tensor holder here.
  nusc_im_data = torch.FloatTensor(1)
  nusc_im_info = torch.FloatTensor(1)
  nusc_num_boxes = torch.LongTensor(1)
  nusc_gt_boxes = torch.FloatTensor(1)
  nusc_pcl = torch.FloatTensor(1)

  # ship to cuda

  if args.cuda > 0:
    nusc_im_data = nusc_im_data.cuda()
    nusc_im_info = nusc_im_info.cuda()
    nusc_num_boxes = nusc_num_boxes.cuda()
    nusc_gt_boxes = nusc_gt_boxes.cuda()
    nusc_pcl = nusc_pcl.cuda()

  # make variable
  nusc_im_data = Variable(nusc_im_data, volatile=True)
  nusc_im_info = Variable(nusc_im_info, volatile=True)
  nusc_num_boxes = Variable(nusc_num_boxes, volatile=True)
  nusc_gt_boxes = Variable(nusc_gt_boxes, volatile=True)
  nusc_pcl = Variable(nusc_pcl, volatile = True)

  if args.cuda > 0:
    cfg.CUDA = True

  if args.cuda > 0:
    fasterRCNN.cuda()

  fasterRCNN.eval()

  start = time.time()
  max_per_image = 100
  thresh = 0.05
  vis = True

  """
  imglist = []
  if args.nuscenes == True:
      for scene in nusc.scene:
          token = scene['first_sample_token']
          while token != '':
              sample = nusc.get('sample',token)
              sample_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
              data_path, boxes, camera_intrinsic = explorer.nusc.get_sample_data(sample_data['token'], box_vis_level = BoxVisibility.ANY)
              imglist += [data_path]
             # img = nusc.render_sample_data(sample_data['token'])
              token = sample['next']   
      print(len(imglist))
  else:
      imglist = os.listdir(args.image_dir)
      print(imglist)
      num_images = len(imglist)

  print('Loaded Photo: {} images.'.format(num_images))
  """
  crop_data = torch.FloatTensor(1)
  crop_data = crop_data
  crop_data = Variable(crop_data)

  #while (num_images >= 0):
  for epoch in range(1, 20 + 1):
      #MLP.train()
      loss_temp = 0
      start = time.time()
      """
      if epoch % (5 + 1) == 0:
        adjust_learning_rate(optimizer, 0.1)
        lr *= args.lr_decay_gamma
      """
      nusc_iter = iter(nusc_dataloader)
      for step in range(nusc_iters_per_epoch):
        nusc_data = next(nusc_iter)
        with torch.no_grad():
          nusc_im_data.resize_(nusc_data[0].size()).copy_(nusc_data[0])
          nusc_im_info.resize_(nusc_data[1].size()).copy_(nusc_data[1])
          nusc_gt_boxes.resize_(nusc_data[2].size()).copy_(nusc_data[2])
          nusc_num_boxes.resize_(nusc_data[3].size()).copy_(nusc_data[3])
          nusc_pcl.resize_(nusc_data[4].size()).copy_(nusc_data[4])
        

        """
        im_in = np.array(nusc_im_data.to_cpu())
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
        """
        #det_tic = time.time()
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = fasterRCNN(nusc_im_data, nusc_im_info, nusc_gt_boxes, nusc_num_boxes)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

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
                  box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, nusc_im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))
        pdb.set_trace()
        im_scales = (nusc_im_info.cpu().numpy())[0][2]
        pred_boxes /= im_scales

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        
        if vis:
            im2show = np.copy(im)
        print("Shape of boxes: " + str(pred_boxes.size()))
        print(pred_boxes)
        for j in xrange(1, len(pascal_classes)):
            inds = torch.nonzero(scores[:,j]).view(-1)
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
              # keep = nms(cls_dets, cfg.TEST.NMS, force_cpu=not cfg.USE_GPU_NMS)
              keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
              cls_dets = cls_dets[keep.view(-1).long()]
              
              for i in range(np.minimum(10, cls_dets.cpu().numpy().shape[0])):
                  score = cls_dets.cpu().numpy()[i, -1]
                  if score > 0.8:
                    bbox = tuple(int(np.round(x)) for x in cls_dets.cpu().numpy()[i, :4])
                    bottom_left = bbox[0:2]
                    top_right = bbox[2:4] 
                    crop_img = im[bottom_left[1]:top_right[1],bottom_left[0]:top_right[0]]
                    crop_img = crop_img.astype(np.float32, copy=False)
                    crop_img = preprocess_image(crop_img)
                    crop_img = np.expand_dims(crop_img, axis=0)
                    crop_data.resize_((torch.from_numpy(crop_img)).size()).copy_(torch.from_numpy(crop_img))

                    res50_model = models.resnet50(pretrained=True)
                    res50_conv2 = ResNet50Bottom(res50_model)
                    #for param in res101_conv.parameters():
                    #  param.requires_grad = False
                    outputs = res50_conv2(crop_data)
                    outputs = torch.squeeze(outputs,2)
                    outputs = torch.squeeze(outputs,2)

                    pdb.set_trace()
                    pcl = get_pointcloud(crop_img) 

                
                    base_feat = resnet(crop_data)
                    point_feat = pointNet(pcl)
                    global_feature = base_feat + point_feat 

                    # apply three regression layers 
                    pdb.set_trace()
                    print(base_feat.size())
                    #plt.figure()
                    #plt.imshow(crop_img)
                    #plt.show()
              if vis:
                im2show = vis_detections(im2show, pascal_classes[j], cls_dets.cpu().numpy(), 0.5)
      
      plt.figure()
      print(im2show.shape)
      plt.imshow(im2show)
      plt.show()
      
      misc_toc = time.time()
      nms_time = misc_toc - misc_tic

      if webcam_num == -1:
          sys.stdout.write('im_detect: {:d}/{:d} {:.3f}s {:.3f}s   \r' \
                           .format(num_images + 1, len(imglist), detect_time, nms_time))
          sys.stdout.flush()

      if vis and webcam_num == -1:
          # cv2.imshow('test', im2show)
          # cv2.waitKey(0)
          result_path = os.path.join("images", imglist[num_images][:-4] + "_det.jpg")
          
          cv2.imwrite(result_path, im2show)
      else:
          im2showRGB = cv2.cvtColor(im2show, cv2.COLOR_BGR2RGB)
          cv2.imshow("frame", im2showRGB)
          total_toc = time.time()
          total_time = total_toc - total_tic
          frame_rate = 1 / total_time
          print('Frame rate:', frame_rate)
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
  if webcam_num >= 0:
      cap.release()
      cv2.destroyAllWindows()
