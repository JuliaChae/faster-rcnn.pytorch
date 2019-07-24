import os.path as osp 

import matplotlib.pyplot as plt 
import matplotlib.cm as cm 
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

from nuscenes import NuScenes 
from nuscenes import NuScenesExplorer 
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import BoxVisibility

def get_pcl():
	nusc = NuScenes(version='v1.0-mini', dataroot='/data/sets/nuscenes', verbose=True)
	explorer = NuScenesExplorer(nusc)

	imglist = []

	count = 0
	for scene in nusc.scene:
		token = scene['first_sample_token']
		while token != '':
			count += 1
			sample = nusc.get('sample',token)
			sample_data_cam = nusc.get('sample_data', sample['data']['CAM_FRONT'])
			sample_data_pcl = nusc.get('sample_data', sample['data']['LIDAR_TOP'])

			# get CAM_FRONT datapath 
			data_path, boxes, camera_intrinsic = explorer.nusc.get_sample_data(sample_data_cam['token'], box_vis_level = BoxVisibility.ANY)
			imglist += [data_path]

			# get LiDAR point cloud 
			sample_rec = explorer.nusc.get('sample', sample_data_pcl['sample_token'])
			chan = sample_data_pcl['channel']
			ref_chan = 'LIDAR_TOP'
			pc, times = LidarPointCloud.from_file_multisweep(nusc, sample_rec, chan, ref_chan)

			radius = np.sqrt((pc.points[0])**2 + (pc.points[1])**2 + (pc.points[2])**2)
			pc = pc.points.transpose() 
			pc = pc[np.where(radius<20)]
			print(pc)
			display(pc.transpose(), count)

			token = sample['next']   
			if count == 2:
				break
		if count == 2:
			break
	plt.show()
	print(len(imglist))


def display(pc, i):
	# 3D plotting of point cloud 
	fig=plt.figure(i)
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

get_pcl()
