"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datasets
import numpy as np
from model.utils.config import cfg
import PIL
import pdb
import os 

from nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points, BoxVisibility
import pickle 

def prepare_roidb():
  """Enrich the imdb's roidb by adding some derived quantities that
  are useful for training. This function precomputes the maximum
  overlap, taken over ground-truth boxes, between each ROI and
  each ground-truth box. The class with maximum overlap is also
  recorded.
  """
  classes = ('__background__', 
                           'pedestrian', 'barrier', 'trafficcone', 'bicycle', 'bus', 'car', 'construction', 'motorcycle', 'trailer', 'truck')
                           
  nusc_path = '/data/sets/nuscenes'
  nusc= NuScenes(version='v1.0-trainval', dataroot = nusc_path, verbose= True)
  
  if os.path.exists('lib/roi_data_layer/roidb_nuscenes_mini.pkl'):
      print("Reading roidb..")
      pickle_in = open("lib/roi_data_layer/roidb_nuscenes_mini.pkl","rb")
      roidb = pickle.load(pickle_in)
      return nusc, roidb
  else:   
      file_dir = os.path.dirname(os.path.abspath(__file__))
      roots = file_dir.split('/')[:-2]
      root_dir = ""
      for folder in roots:
        if folder != "":
          root_dir = root_dir + "/" + folder
      
      PATH = root_dir + '/data/train_mini.txt'
      with open(PATH) as f:
       image_token = [x.strip() for x in f.readlines()]
      
      roidb = [] 
      
      print("Loading roidb...")
      for i in range(len(image_token)):
        im_token = image_token[i] 
        sample_data = nusc.get('sample_data', im_token)
        image_name = sample_data['filename']
        image_path = nusc_path + '/' + image_name
        
        data_path, boxes, camera_intrinsic = nusc.get_sample_data(im_token, box_vis_level=BoxVisibility.ALL)
        
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
                    gt_cls = gt_cls + [classes.index(name)]
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
      pickle_out = open("lib/roi_data_layer/roidb_nuscenes_mini.pkl","wb")
      pickle.dump(roidb, pickle_out)
      pickle_out.close()
      return nusc, roidb 
