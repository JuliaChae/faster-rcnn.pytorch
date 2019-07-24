
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
from nuscenes import NuScenesExplorer 
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
import os 

import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from mpl_toolkits.mplot3d import Axes3D

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
    self.nusc= NuScenes(version='v1.0-trainval', dataroot = '/data/sets/nuscenes', verbose= True)
    self.explorer = NuScenesExplorer(self.nusc)
    self.classes = ('__background__', 
                           'pedestrian', 'barrier', 'trafficcone', 'bicycle', 'bus', 'car', 'construction', 'motorcycle', 'trailer', 'truck')
    file_dir = os.path.dirname(os.path.abspath(__file__))
    roots = file_dir.split('/')[:-2]
    root_dir = ""
    for folder in roots:
        if folder != "":
            root_dir = root_dir + "/" + folder
    pdb.set_trace()
    if training == True: 
        PATH = root_dir + '/data/train_mini.txt'
    else:
        PATH = root_dir + '/data/test_mini.txt'
    with open(PATH) as f:
        self.image_token = [x.strip() for x in f.readlines()]

  def __getitem__(self, index):

    # get the sample_data for the image batch
    im_token = self.image_token[index]
    sample_data = self.nusc.get('sample_data', im_token)
    sample_token = sample_data['sample_token']
    sample = self.nusc.get('sample', sample_token)
    image_name = sample_data['filename']
    image_path = '/data/sets/nuscenes/' + image_name
    img = imread('/data/sets/nuscenes/' + image_name)
    im = np.array(img)
    
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
    
    # save im_info
    im_info = np.array([im.shape[1], im.shape[2], im_scale])

    # get ground truth boxes 
    data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(im_token, box_vis_level=BoxVisibility.ANY)

    gt_boxes = np.zeros((50, 5), dtype=np.float32)
    i = 0
    for box in boxes:
        if i == 50:
            break
        corners= view_points(box.corners(), view=camera_intrinsic, normalize=True)[:2,:]
        corners = corners*im_scale

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
            gt_boxes[i,4]=self.classes.index(name)
            i += 1 
            
    # save the number of boxes 
    num_boxes = len(boxes)
 
    # load in the LiDAR data for the sample 
    sample_data_pcl = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    sample_rec = self.explorer.nusc.get('sample', sample_data_pcl['sample_token'])

    # get LiDAR point cloud 
    chan = sample_data_pcl['channel']
    ref_chan = 'LIDAR_TOP'
    pc, times = LidarPointCloud.from_file_multisweep(self.nusc, sample_rec, chan, ref_chan)

    radius = np.sqrt((pc.points[0])**2 + (pc.points[1])**2 + (pc.points[2])**2)
    pc = pc.points.transpose() 
    pc = pc[np.where(radius<20)]
    pc = pc.transpose()[:3]

    return im, im_info, gt_boxes, num_boxes, image_path

  def draw_rect(self,axis, selected_corners):
    prev = selected_corners[-1]
    for corner in selected_corners:
        axis.plot([prev[0], corner[0]], [prev[1], corner[1]])
        prev = corner
    # Function checks if boxes appear in image and converts it to image frame 

  def display(self,pc):
    # 3D plotting of point cloud 
    fig=plt.figure()
    ax = fig.gca(projection='3d')

    #ax.set_aspect('equal')
    X = pc[0]
    Y = pc[1]
    Z = pc[2]
    c = pc[3]

    """
    radius = np.sqrt(X**2 + Y**2 + Z**2)
    X = X[np.where(radius<20)]
    Y = Y[np.where(radius<20)]
    Z = Z[np.where(radius<20)]
    c = pc.points[3][np.where(radius<20)]
    print(radius)
    """
    ax.scatter(X, Y, Z, s=1, c=cm.hot((c/100)))

    max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
    Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
    Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
    Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())

    i=0
    for xb, yb, zb in zip(Xb, Yb, Zb):
        i = i+1 
        ax.plot([xb], [yb], [zb], 'w')

  def __len__(self):
    return len(self.image_token)
