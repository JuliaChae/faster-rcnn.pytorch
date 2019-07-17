from nuscenes import NuScenes 
from nuscenes import NuScenesExplorer 

from imageio import imread 

from roi_data_layer.nuscenes_dataloader import nuscenes_dataloader
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility

import pdb

from typing import Tuple, List
import os.path as osp
from pyquaternion import Quaternion
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init

from model.pointnet import PointNetfeat

import torch


nusc= NuScenes(version='v1.0-trainval', dataroot = '/data/sets/nuscenes', verbose= True)
explorer = NuScenesExplorer(nusc)

"""
class InputTransformNet(nn.Module):
    def __init__(self):
        super(InputTransformNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn5 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 9)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        B, N = x.shape[0], x.shape[2]
        x = self.relu(self.bn1(self.conv1(x))) #[B, 64, N]
        x = self.relu(self.bn2(self.conv2(x))) #[B, 128, N]
        x = self.relu(self.bn3(self.conv3(x))) #[B, 1024, N]
        x = nn.MaxPool1d(N)(x) #[B, 1024, 1]
        x = x.view(B, 1024) #[B, 1024]
        x = self.relu(self.bn4(self.fc1(x))) #[B, 512]
        x = self.relu(self.bn5(self.fc2(x))) #[B, 256]
        x = self.transform(x) #[B, 9]
        x = x.view(B, 3, 3) #[B, 3, 3]
        return x

class PointNet(nn.Module):
    def __init__(self, global_feature=True):
        super(PointNet, self).__init__()
        self.global_feature = global_feature
        self.input_transform = InputTransformNet()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.bn2 = nn.BatchNorm1d(64)
        self.feature_transform = FeatureTransformNet()
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.bn4 = nn.BatchNorm1d(128)
        self.conv5 = nn.Conv1d(128, 1024, 1)
        self.bn5 = nn.BatchNorm1d(1024)
    
    def forward(self, x):
        B, N = x.shape[0], x.shape[2]
        input_transform = self.input_transform(x) #[B, 3, 3]
        x = torch.matmul(x.permute(0, 2, 1), input_transform.permute(0, 2, 1)).permute(0, 2, 1) #[B, 3, N]
        x = F.relu(self.bn1(self.conv1(x))) #[B, 64, N]
        x = F.relu(self.bn2(self.conv2(x))) #[B, 64, N]
        feature_transform = self.feature_transform(x) #[B, 64, 64]
        x = torch.matmul(x.permute(0, 2, 1), feature_transform.permute(0, 2, 1)).permute(0, 2, 1) #[B, 64, N]
        point_feature = x
        x = F.relu(self.bn3(self.conv3(x))) #[B, 64, N]
        x = F.relu(self.bn4(self.conv4(x))) #[B, 128, N]
        x = F.relu(self.bn5(self.conv5(x))) #[B, 1024, N]
        x = nn.MaxPool1d(N)(x) #[B, 1024, 1]
        if not self.global_feature:
            x = x.repeat([1, 1, N]) #[B, 1024, N]
            x = torch.cat([point_feature, x], 1) #[B, 1088, N]
        return x
"""

def map_pointcloud_to_image(pointsensor_token: str, camera_token: str) -> Tuple:
        """
        Given a point sensor (lidar/radar) token and camera sample_data token, load point-cloud and map it to the image
        plane.
        :param pointsensor_token: Lidar/radar sample_data token.
        :param camera_token: Camera sample_data token.
        :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
        """
        global nusc
        cam = nusc.get('sample_data', camera_token)
        pointsensor = nusc.get('sample_data', pointsensor_token)
        pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
        if pointsensor['sensor_modality'] == 'lidar':
            pc = LidarPointCloud.from_file(pcl_path)
        else:
            pc = RadarPointCloud.from_file(pcl_path)
        im = Image.open(osp.join(nusc.dataroot, cam['filename']))

        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
        # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
        cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
        pc.translate(np.array(cs_record['translation']))

        # Second step: transform to the global frame.
        poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
        pc.translate(np.array(poserecord['translation']))

        # Third step: transform into the ego vehicle frame for the timestamp of the image.
        poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
        pc.translate(-np.array(poserecord['translation']))
        pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

        # Fourth step: transform into the camera.
        cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
        pc.translate(-np.array(cs_record['translation']))
        pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = pc.points[2, :]

        # Retrieve the color from the depth.
        coloring = depths

        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)
        crop_points = pc.points[:3,:]
        pdb.set_trace()
        # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > 0)
        mask = np.logical_and(mask, points[0, :] > 1)
        mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
        mask = np.logical_and(mask, points[1, :] > 1)
        mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
        points = points[:, mask]
        crop_points = crop_points[:, mask]
        coloring = coloring[mask]
        pdb.set_trace()
        return points, coloring, im

PATH = '/data/sets/nuscenes/train_mini.txt'

with open(PATH) as f:
    image_token = [x.strip() for x in f.readlines()]

#model = PointNetCls(k=16)
#model.cuda()
#model.load_state_dict(torch.load('/home/julia/pointnet.pytorch/utils/cls/cls_model_22.pth'))
#model.eval()

im_token = image_token[0]
sample_data = nusc.get('sample_data', im_token)
sample_token = sample_data['sample_token']
sample =nusc.get('sample', sample_token)
image_name = sample_data['filename']
img = imread('/data/sets/nuscenes/' + image_name)

# load in the LiDAR data for the sample 
sample_data_pcl = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
sample_rec = explorer.nusc.get('sample', sample_data_pcl['sample_token'])

# get LiDAR point cloud 
chan = sample_data_pcl['channel']
ref_chan = 'LIDAR_TOP'
pc, times = LidarPointCloud.from_file_multisweep(nusc, sample_rec, chan, ref_chan)

points= pc.points[:3,:]
points = points.astype(np.float32, copy=False)
points = np.expand_dims(points, axis=0)
points= torch.from_numpy(points)

pc2, c, im = map_pointcloud_to_image(sample_data_pcl['token'], im_token)
pointfeat = PointNetfeat(global_feat=True)
_, global_feat= pointfeat(points)
#classifier = classifier.train()
#global_feat = model(points)
pdb.set_trace()

