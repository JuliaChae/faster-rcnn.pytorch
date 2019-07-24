# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
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
import functools 
import operator 

import cv2

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataset import random_split 

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.nuscenes_dataloader import nuscenes_dataloader
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
	  adjust_learning_rate, save_checkpoint, clip_gradient, vis_detections

from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet

import matplotlib.pyplot as plt 
import matplotlib.patches as patches 
from model.utils.config import cfg

import pdb

def parse_args():
	"""
	Parse input arguments
	"""
	parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
	parser.add_argument('--dataset', dest='dataset',
						help='training dataset',
						default='nuscenes', type=str)
	parser.add_argument('--net', dest='net',
						help='vgg16, res101',
						default='vgg16', type=str)
	parser.add_argument('--date', dest='timestamp',
						help='date & time information',
						default='07_23', type=str)
	parser.add_argument('--start_epoch', dest='start_epoch',
						help='starting epoch',
						default=1, type=int)
	parser.add_argument('--epochs', dest='max_epochs',
						help='number of epochs to train',
						default=50, type=int)
	parser.add_argument('--disp_interval', dest='disp_interval',
						help='number of iterations to display',
						default=100, type=int)
	parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
						help='number of iterations to display',
						default=10000, type=int)

	parser.add_argument('--save_dir', dest='save_dir',
						help='directory to save models', default="models",
						type=str)
	parser.add_argument('--nw', dest='num_workers',
						help='number of worker to load data',
						default=0, type=int)
	parser.add_argument('--cuda', dest='cuda',
						help='whether use CUDA',
						action='store_true')
	parser.add_argument('--ls', dest='large_scale',
						help='whether use large imag scale',
						action='store_true')					  
	parser.add_argument('--mGPUs', dest='mGPUs',
						help='whether use multiple GPUs',
						action='store_true')
	parser.add_argument('--bs', dest='batch_size',
						help='batch_size',
						default=1, type=int)
	parser.add_argument('--cag', dest='class_agnostic',
						help='whether perform class_agnostic bbox regression',
						action='store_true')

# config optimization
	parser.add_argument('--o', dest='optimizer',
						help='training optimizer',
						default="sgd", type=str)
	parser.add_argument('--lr', dest='lr',
						help='starting learning rate',
						default=0.001, type=float)
	parser.add_argument('--lr_decay_step', dest='lr_decay_step',
						help='step to do learning rate decay, unit is epoch',
						default=5, type=int)
	parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
						help='learning rate decay ratio',
						default=0.1, type=float)

# set training session
	parser.add_argument('--s', dest='session',
						help='training session',
						default=1, type=int)

# resume trained model
	parser.add_argument('--r', dest='resume',
						help='resume checkpoint or not',
						default=False, type=bool)
	parser.add_argument('--checksession', dest='checksession',
						help='checksession to load model',
						default=1, type=int)
	parser.add_argument('--checkepoch', dest='checkepoch',
						help='checkepoch to load model',
						default=1, type=int)
	parser.add_argument('--checkpoint', dest='checkpoint',
						help='checkpoint to load model',
						default=0, type=int)
# log and diaplay
	parser.add_argument('--use_tfb', dest='use_tfboard',
						help='whether use tensorboard',
						action='store_true')

	args = parser.parse_args()
	return args


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

def get_accuracy(all_boxes, gtruth_boxes):
	correct_count = np.zeros(len(nusc_classes))
	actual_count = np.zeros(len(nusc_classes))
	for cls_i in range(len(nusc_classes)):
		for img_i in range(len(gtruth_boxes[cls_i])):
			actual_count[cls_i] = actual_count[cls_i] + len(gtruth_boxes[cls_i][img_i])
			if len(gtruth_boxes[cls_i][img_i])>0:
				correct_count[cls_i] = correct_count[cls_i] + get_correct_count(all_boxes[cls_i][img_i], gtruth_boxes[cls_i][img_i]) 
	print(correct_count)
	print(actual_count)
	return correct_count/actual_count

def get_correct_count(pred, truth):
	counter = 0
	for box in truth:
		if len(pred)!=0:
			for pbox in pred:
				if iou(box, pbox)>0.3:
					counter+=1
					break
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

	if args.dataset == "pascal_voc":
		args.imdb_name = "voc_2007_trainval"
		args.imdbval_name = "voc_2007_test"
		args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
	elif args.dataset == 'nuscenes':
		args.imdb_name = "nuscenes_train"
		args.imdbval_name = "nuscenes_val"
		args.set_cfgs = ['ANCHOR_SCALES', '[8, 6, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

	nusc_classes = ('__background__', 
						   'pedestrian', 'barrier', 'trafficcone', 'bicycle', 'bus', 'car', 'construction', 'motorcycle', 'trailer', 'truck')
	args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

	if args.cfg_file is not None:
		cfg_from_file(args.cfg_file)
	if args.set_cfgs is not None:
		cfg_from_list(args.set_cfgs)

	print('Using config:')
	pprint.pprint(cfg)
	np.random.seed(cfg.RNG_SEED)

	#torch.backends.cudnn.benchmark = True
	if torch.cuda.is_available() and not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")

	# train set
	# -- Note: Use validation set and disable the flipped to enable faster loading.
	cfg.TRAIN.USE_FLIPPED = True
	cfg.USE_GPU_NMS = args.cuda

	out_dir = os.path.dirname(os.path.abspath(__file__))
	output_dir = out_dir + "/data/trained_model/" + args.net + "/" + args.dataset + "/diffcam/" + args.timestamp
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	nusc_sampler_batch = sampler(317, args.batch_size)
	train_sampler_batch = sampler(254, args.batch_size)
	val_sampler_batch = sampler(63, args.batch_size)
  
	nusc_set = nuscenes_dataloader(args.batch_size, len(nusc_classes), training = True)
	training_set, validation_set = random_split(nusc_set, [254,63])#[1900, 39])
	print(training_set)
	print(validation_set)
	nusc_dataloader = torch.utils.data.DataLoader(nusc_set, batch_size = args.batch_size , num_workers = args.num_workers, sampler = nusc_sampler_batch)
	train_loader = torch.utils.data.DataLoader(training_set, batch_size = args.batch_size , num_workers = args.num_workers, sampler = train_sampler_batch)
	val_loader = torch.utils.data.DataLoader(validation_set, batch_size = args.batch_size , num_workers = args.num_workers, sampler = val_sampler_batch)

	with open(output_dir + '/args.txt', 'w') as f:
		for key, value in vars(args).items():
			f.write("%s" % key + " " + str(value))
			f.write("\n")
		f.write("Training set size: " + str(len(training_set)) + "\n")
		f.write("Validation set size: " + str(len(validation_set)) + "\n")

	with open(output_dir + '/cfgs.txt', 'w') as f:
		for key, value in cfg.items():
			f.write("%s" % key + " " + str(value))
			f.write("\n")

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

	im_data = Variable(im_data)
	im_info = Variable(im_info)
	num_boxes = Variable(num_boxes)
	gt_boxes = Variable(gt_boxes)

	if args.cuda:
		cfg.CUDA = True

	# initilize the network here.
	if args.net == 'vgg16':
		fasterRCNN = vgg16(nusc_classes, pretrained=True, class_agnostic=args.class_agnostic)
	elif args.net == 'res101':
		fasterRCNN = resnet(nusc_classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
	elif args.net == 'res50':
		fasterRCNN = resnet(nusc_classes, 50, pretrained=True, class_agnostic=args.class_agnostic)
	elif args.net == 'res152':
		fasterRCNN = resnet(nusc_classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
	else:
		print("network is not defined")

	fasterRCNN.create_architecture()

	lr = cfg.TRAIN.LEARNING_RATE
	lr = args.lr

	params = []
	for key, value in dict(fasterRCNN.named_parameters()).items():
		if value.requires_grad:
			if 'bias' in key:
				params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
					'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
			else:
				params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

	if args.cuda:
		fasterRCNN.cuda()
	  
	if args.optimizer == "adam":
		lr = lr * 0.1
		optimizer = torch.optim.Adam(params)
	elif args.optimizer == "sgd":
		optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

	if args.resume:
		load_name = os.path.join(output_dir,
		  'faster_rcnn_{}_{}_{}.pth'.format(args.checksession, args.checkepoch, args.checkpoint))
		print("loading checkpoint %s" % (load_name))
		checkpoint = torch.load(load_name)
		args.session = checkpoint['session']
		args.start_epoch = checkpoint['epoch']
		fasterRCNN.load_state_dict(checkpoint['model'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		lr = optimizer.param_groups[0]['lr']
		if 'pooling_mode' in checkpoint.keys():
		  cfg.POOLING_MODE = checkpoint['pooling_mode']
		print("loaded checkpoint %s" % (load_name))

	if args.mGPUs:
		fasterRCNN = nn.DataParallel(fasterRCNN)

	nusc_iters_per_epoch = int(len(nusc_set) / args.batch_size)

	if args.use_tfboard:
		from tensorboardX import SummaryWriter
		logger = SummaryWriter("logs")

	accuracy = 0

	f1 = open(output_dir + '/loss.txt','w+')
	f1.write("epoch	train	val\n")
	f1.close()
	f2 = open(output_dir + '/train_accuracy.txt','w+')
	f2.close()
	f3 = open(output_dir + '/val_accuracy.txt','w+')
	f3.close()

	min_loss = 100
	for epoch in range(args.start_epoch, args.max_epochs + 1):
		train_accuracy = []
		val_accuracy = [] 

		# setting to train mode
		fasterRCNN.train()
		train_loss_temp = 0
		train_loss_out = 0
		val_loss_temp = 0
		start = time.time()
		accuracy = 0

		if epoch % (args.lr_decay_step + 1) == 0:
			adjust_learning_rate(optimizer, args.lr_decay_gamma)
			lr *= args.lr_decay_gamma

		nusc_iter = iter(nusc_dataloader)
		train_iter = iter(train_loader)
		val_iter = iter(val_loader)

		# training 
		all_boxes = [[[] for _ in range(len(nusc_set))]
					 for _ in range(len(nusc_classes))]
		truth_boxes = [[[] for _ in range(len(nusc_set))]
					 for _ in range(len(nusc_classes))]
		vis = True
		max_per_image = 100
		args.class_agnostic = True
		for step in range(len(validation_set)):
			nusc_data = next(val_iter)
			with torch.no_grad():
				im_data.resize_(nusc_data[0].size()).copy_(nusc_data[0])
				im_info.resize_(nusc_data[1].size()).copy_(nusc_data[1])
				gt_boxes.resize_(nusc_data[2].size()).copy_(nusc_data[2])
				num_boxes.resize_(nusc_data[3].size()).copy_(nusc_data[3])
				image_path = functools.reduce(operator.add, (nusc_data[4]))
			fasterRCNN.zero_grad()
			for box in gt_boxes[0].cpu().numpy():
				if box[0]==0.0 and box[1]==0.0 and box[2]==0.0 and box[3]==0.0:
					break
				cls_i = int(box[4])
				box = list(int(np.round(x/ nusc_data[1][0][2].item())) for x in box[:4])
				truth_boxes[cls_i][step] = truth_boxes[cls_i][step] + [box]

			rois, cls_prob, bbox_pred, \
			rpn_loss_cls, rpn_loss_box, \
			RCNN_loss_cls, RCNN_loss_bbox, \
			rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

			#accuracy += get_accuracy(cls_prob.data.squeeze(), gt_boxes.data.squeeze())
			train_loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
			   + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
			train_loss_temp += train_loss.item()
			train_loss_out += train_loss.item()
			if train_loss != train_loss:
				pdb.set_trace()

			# accuracy
			scores = cls_prob.data
			boxes = rois.data[:, :, 1:5]

			if cfg.TRAIN.BBOX_REG:
				# Apply bounding-box regression deltas
				box_deltas = bbox_pred.data
				if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
				# Optionally normalize targets by a precomputed mean and stdev
					if args.class_agnostic:
						box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
								   + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
						box_deltas = box_deltas.view(1, -1, 4)
					else:
						box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
								   + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
						box_deltas = box_deltas.view(1, -1, 4)# * len(nusc_classes))
					pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
					pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
				else:
				# Simply repeat the boxes, once for each class
					pred_boxes = np.tile(boxes, (1, scores.shape[1]))

			pred_boxes /= nusc_data[1][0][2].item()

			scores = scores.squeeze()
			pred_boxes = pred_boxes.squeeze()

			if vis:
				im = cv2.imread(image_path)
				im2show = np.copy(im)
			bounding_boxes = []
			for j in range(1, len(nusc_classes)):
				inds = torch.nonzero(scores[:,j]>0.5).view(-1)
				# if there is det
				if inds.numel() > 0:
					cls_scores = scores[:,j][inds]
					_, order = torch.sort(cls_scores, 0, True)
					if args.class_agnostic:
						cls_boxes = pred_boxes[inds, :]
					else:
						cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
				
					cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
					cls_dets = cls_dets[order]
					keep = nms(cls_boxes[order, :], cls_scores[order], 0.5)
					cls_dets = cls_dets[keep.view(-1).long()]
					dets = cls_dets.cpu().numpy()
					for i in range(np.minimum(10, dets.shape[0])):
						bbox = list(int(np.round(x)) for x in cls_dets.cpu().numpy()[i, :4])
						bbox = bbox + [j]
						score = dets[i,-1]
						if score>0.5:
						   bounding_boxes += [bbox]
					if vis:
					  im2show = vis_detections(im2show, nusc_classes[j], cls_dets.cpu().numpy(), 0.3)
				else:
					all_boxes[j][step] = []
			for box in bounding_boxes:
				all_boxes[box[4]][step] = all_boxes[box[4]][step] + [box[:4]]

			# backward
			optimizer.zero_grad()
			train_loss.backward()
			if args.net == "vgg16":
				clip_gradient(fasterRCNN, 10.)
			optimizer.step()

			if step % args.disp_interval == 0:
				end = time.time()
				if step > 0:
					train_loss_out /=(args.disp_interval + 1)

				if args.mGPUs:
					loss_rpn_cls = rpn_loss_cls.mean().item()
					loss_rpn_box = rpn_loss_box.mean().item()
					loss_rcnn_cls = RCNN_loss_cls.mean().item()
					loss_rcnn_box = RCNN_loss_bbox.mean().item()
					fg_cnt = torch.sum(rois_label.data.ne(0))
					bg_cnt = rois_label.data.numel() - fg_cnt
				else:
					loss_rpn_cls = rpn_loss_cls.item()
					loss_rpn_box = rpn_loss_box.item()
					loss_rcnn_cls = RCNN_loss_cls.item()
					loss_rcnn_box = RCNN_loss_bbox.item()
					fg_cnt = torch.sum(rois_label.data.ne(0))
					bg_cnt = rois_label.data.numel() - fg_cnt

				print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, accuracy: %.4f, lr: %.2e" % (args.session, epoch, step, len(validation_set), train_loss_out, accuracy, lr))
				print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
				print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

				if args.use_tfboard:
					info = {
					'loss': loss_temp,
					'loss_rpn_cls': loss_rpn_cls,
					'loss_rpn_box': loss_rpn_box,
					'loss_rcnn_cls': loss_rcnn_cls,
					'loss_rcnn_box': loss_rcnn_box
					}
					logger.add_scalars("logs_s_{}/losses".format(args.session), info, (epoch - 1) * nusc_iters_per_epoch + step)

				train_loss_out = 0
				start = time.time()

		train_accuracy = get_accuracy(all_boxes, truth_boxes)
		train_loss_temp = train_loss_temp/len(training_set)
		print("Training loss: " + str(train_loss_temp))
		val_iter = iter(val_loader)
		

		# validation
		#fasterRCNN.eval()
		empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
		all_boxes = [[[] for _ in range(len(validation_set))]
					 for _ in range(len(nusc_classes))]
		truth_boxes = [[[] for _ in range(len(validation_set))]
					 for _ in range(len(nusc_classes))]
		vis = True
		max_per_image = 100

		for i in range(int(len(validation_set))):
			data = next(val_iter)
			with torch.no_grad():
				im_data.resize_(data[0].size()).copy_(data[0])
				im_info.resize_(data[1].size()).copy_(data[1])
				gt_boxes.resize_(data[2].size()).copy_(data[2])
				num_boxes.resize_(data[3].size()).copy_(data[3])
				image_path = functools.reduce(operator.add, (data[4]))
			for box in gt_boxes[0].cpu().numpy():
				if box[0]==0.0 and box[1]==0.0 and box[2]==0.0 and box[3]==0.0:
					break
				cls_i = int(box[4])
				box = list(int(np.round(x/ nusc_data[1][0][2].item())) for x in box[:4])
				truth_boxes[cls_i][i] = truth_boxes[cls_i][i] + [box]
			det_tic = time.time()

			rois, cls_prob, bbox_pred, \
			rpn_loss_cls, rpn_loss_box, \
			RCNN_loss_cls, RCNN_loss_bbox, \
			rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

			val_loss = rpn_loss_cls.mean()+ rpn_loss_box.mean()\
			   + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
			val_loss_temp += val_loss.item()

			if val_loss != val_loss:
				pdb.set_trace()

			scores = cls_prob.data
			boxes = rois.data[:, :, 1:5]

			if cfg.TEST.BBOX_REG:
				# Apply bounding-box regression deltas
				box_deltas = bbox_pred.data
				if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
				# Optionally normalize targets by a precomputed mean and stdev
					box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
						   + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
					box_deltas = box_deltas.view(1, -1, 4)#* len(nusc_classes))
				pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
				pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
			else:
				# Simply repeat the boxes, once for each class
				pred_boxes = np.tile(boxes, (1, scores.shape[1]))

			pred_boxes /= data[1][0][2].item()

			scores = scores.squeeze()
			pred_boxes = pred_boxes.squeeze()
			det_toc = time.time()
			detect_time = det_toc - det_tic
			misc_tic = time.time()

			if vis:
				im = cv2.imread(image_path)
				im2show = np.copy(im)
			bounding_boxes = []
			for j in range(1, len(nusc_classes)):
				inds = torch.nonzero(scores[:,j]>0.5).view(-1)
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
					keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
					cls_dets = cls_dets[keep.view(-1).long()]
					dets = cls_dets.cpu().numpy()
					for i in range(np.minimum(10, dets.shape[0])):
						bbox = list(int(np.round(x)) for x in cls_dets.cpu().numpy()[i, :4])
						bbox = bbox + [j]
						score = dets[i,-1]
						if score>0.5:
						   bounding_boxes += [bbox]
					if vis:
						im2show = vis_detections(im2show, nusc_classes[j], cls_dets.cpu().numpy(), 0.3)			
				else:
					all_boxes[j][i] = []
			for box in bounding_boxes:
				all_boxes[box[4]][i] = all_boxes[box[4]][i] + [box[:4]]

		#print(get_accuracy(all_boxes, truth_boxes))
		val_accuracy = get_accuracy(all_boxes, truth_boxes)
		val_loss_temp = val_loss_temp/len(validation_set)
		print("Validation loss: " + str(val_loss_temp))
		
		f1 = open(output_dir + '/loss.txt','a+')
		f1.write("%2d		%.4f	%.4f\n" % (epoch, train_loss_temp, val_loss_temp))
		f1.close()
		f2 = open(output_dir + '/train_accuracy.txt','a+')
		f2.write("%.4f	%.4f	%.4f	%.4f	%.4f	%.4f	%.4f	%.4f	%.4f	%.4f\n" % (train_accuracy[1], train_accuracy[2], train_accuracy[3], train_accuracy[4], train_accuracy[5], train_accuracy[6], train_accuracy[7], train_accuracy[8], train_accuracy[9], train_accuracy[10]))
		f2.close()
		f3 = open(output_dir + '/val_accuracy.txt','a+')
		f3.write("%.4f	%.4f	%.4f	%.4f	%.4f	%.4f	%.4f	%.4f	%.4f	%.4f\n" % (val_accuracy[1], val_accuracy[2], val_accuracy[3], val_accuracy[4], val_accuracy[5], val_accuracy[6], val_accuracy[7], val_accuracy[8], val_accuracy[9], val_accuracy[10]))
		f3.close()_temp 

		if val_loss_temp < min_loss:
			min_loss = val_loss_temp 
			save_name = os.path.join(output_dir, 'faster_rcnn_{}_{}_{}.pth'.format(args.session, epoch, step))
			save_checkpoint({
			  'session': args.session,
			  'epoch': epoch + 1,
			  'model': fasterRCNN.module.state_dict() if args.mGPUs else fasterRCNN.state_dict(),
			  'optimizer': optimizer.state_dict(),
			  'pooling_mode': cfg.POOLING_MODE,
			  'class_agnostic': args.class_agnostic,
			}, save_name)
			print('saved model: {}'.format(save_name))

	if args.use_tfboard:
		logger.close()

