
"""The data layer used during training to train a Fast R-CNN network.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
from imageio import imread 
import torch

from model.utils.config import cfg

import numpy as np
import numpy.random as npr
from PIL import Image
import random
import time
import pdb
from pyquaternion import Quaternion

from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
import os 

import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from mpl_toolkits.mplot3d import Axes3D

import pickle
import cv2 

class nuscenes_dataloader(data.Dataset):
  def __init__(self, batch_size, num_classes, training=True, normalize=None):
    self._num_classes = num_classes
    # we make the height of image consistent to trim_height, trim_width
    self.trim_height = cfg.TRAIN.TRIM_HEIGHT
    self.trim_width = cfg.TRAIN.TRIM_WIDTH
    self.max_num_box = cfg.MAX_NUM_GT_BOXES
    self.training = training
    self.normalize = normalize
    self.batch_size = batch_size
    self.classes = ('__background__', 
                           'pedestrian', 'barrier', 'trafficcone', 'bicycle', 'bus', 'car', 'construction', 'motorcycle', 'trailer', 'truck')
    self.max_num_box = 50
    
    # Checks if pickle file of dataset already exists. If it doesn't exist, creates the file 
    if os.path.exists('lib/roi_data_layer/roidb_CAMFRONT.pkl'):
      print("Reading roidb..")
      pickle_in = open("lib/roi_data_layer/roidb_CAMFRONT.pkl","rb")
      self.roidb = pickle.load(pickle_in)
      trainsize = 20000
      if training == True:
        self.roidb = self.roidb[:trainsize]
        print("roidb size: " + str(len(self.roidb)))
      else:
        self.roidb = self.roidb[trainsize:]
    else:   
      nusc_path = '/data/sets/nuscenes'
      nusc= NuScenes(version='v1.0-trainval', dataroot = nusc_path, verbose= True)
      file_dir = os.path.dirname(os.path.abspath(__file__))
      roots = file_dir.split('/')[:-2]
      root_dir = ""
      for folder in roots:
        if folder != "":
          root_dir = root_dir + "/" + folder
      
      PATH = root_dir + '/data/CAMFRONT.txt'
      with open(PATH) as f:
       image_token = [x.strip() for x in f.readlines()]
      
      roidb = [] 
      
      # Loads information on images and ground truth boxes and saves it as pickle file for faster loading 
      print("Loading roidb...")
      for i in range(len(image_token)):
        im_token = image_token[i] 
        sample_data = nusc.get('sample_data', im_token)
        image_name = sample_data['filename']
        image_path = nusc_path + '/' + image_name
        
        data_path, boxes, camera_intrinsic = nusc.get_sample_data(im_token, box_vis_level=BoxVisibility.ALL)
        
        # Only accepts boxes with above level 3 or 4 visibility and classes with more than 1000 instances
        gt_boxes = [] 
        gt_cls = [] 
        for box in boxes:
            visibility_token = nusc.get('sample_annotation', box.token)['visibility_token']
            vis_level = int(nusc.get('visibility', visibility_token)['token'])
            if (vis_level == 3) or (vis_level == 4):
                visible = True
            else:
                visible = False 
            if visible == True: 
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
                    corners= view_points(box.corners(), view=camera_intrinsic, normalize=True)[:2,:]
                    box = np.zeros(4)
                    box[0]= np.min(corners[0])
                    box[1]= np.min(corners[1])
                    box[2]= np.max(corners[0])
                    box[3]= np.max(corners[1])
                    gt_boxes = gt_boxes + [box]
                    gt_cls = gt_cls + [self.classes.index(name)]
        
        # Only accepts images with at least one object
        if len(gt_boxes)>= 2:
            image = {}
            image['image'] = image_path 
            image['width'] = 1600
            image['height'] = 900
            image['boxes'] = np.asarray(gt_boxes)
            image['gt_classes'] = np.asarray(gt_cls)
            roidb = roidb + [image]
      
      print(len(roidb))
      print("Saving roidb")
      pickle_out = open("lib/roi_data_layer/roidb_CAMFRONT.pkl","wb")
      pickle.dump(roidb, pickle_out)
      pickle_out.close()
      self.roidb = roidb
      trainsize = int(len(self.roidb)*0.8)
      if training == True:
        self.roidb = self.roidb[:trainsize] 
      else:
        self.roidb = self.roidb[trainsize:]

  def __getitem__(self, index):
    item = self.roidb[index]
   
    im = imread(item['image'])
    # get the sample_data for the image batch
    im = np.array(im)
    
    # acquire and process the image data 
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
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    im = im.transpose(2,0,1)
    im = torch.from_numpy(im)
    
    # save im_info
    im_info = np.array([im.shape[1], im.shape[2], im_scale])
    im_info = torch.from_numpy(im_info)

    # get ground truth boxes 
    #data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(im_token, box_vis_level=BoxVisibility.ALL)
    gt_inds = np.where(item['gt_classes'] != 0)[0]
    gt_boxes = np.empty((len(gt_inds), 5), dtype=np.float32)
    gt_boxes[:, 0:4] = item['boxes'][gt_inds, :] * im_scale
    gt_boxes[:, 4] = item['gt_classes'][gt_inds]
    np.random.shuffle(gt_boxes)
    gt_boxes = torch.from_numpy(gt_boxes)

    not_keep = (gt_boxes[:,0] == gt_boxes[:,2]) | (gt_boxes[:,1] == gt_boxes[:,3])
    keep = torch.nonzero(not_keep == 0).view(-1)

    gt_boxes_padding = torch.FloatTensor(self.max_num_box, gt_boxes.size(1)).zero_()
    if keep.numel() != 0:
        gt_boxes = gt_boxes[keep]
        num_boxes = min(gt_boxes.size(0), self.max_num_box)
        gt_boxes_padding[:num_boxes,:] = gt_boxes[:num_boxes]
    else:
        num_boxes = 0
       
    image_path = ''
    return im, im_info, gt_boxes_padding, num_boxes, image_path

  def __len__(self):
    return len(self.roidb)
