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

from utils import sampler,  get_bounding_boxes

from detection_metric.utils import *
from detection_metric.BoundingBox import BoundingBox
from detection_metric.BoundingBoxes import BoundingBoxes
from detection_metric.Evaluator import *

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
                        default='res101', type=str)
    parser.add_argument('--date', dest='timestamp',
                        help='date & time information',
                        default='date', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=15, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--val_interval', dest='val_interval',
                        help='number of epochs to perform validation',
                        default=1, type=int)
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


if __name__ == '__main__':

    args = parse_args()

    print('Called with args:')
    print(args)

    # Setting and loading training configurations 
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

    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # train set
    # -- Note: Use validation set and disable the flipped to enable faster loading.
    cfg.TRAIN.USE_FLIPPED = True
    cfg.USE_GPU_NMS = args.cuda

    out_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = out_dir + "/" + args.save_dir + "/" + args.net + "/" + args.dataset + "/" + args.timestamp
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loading train and val dataset using NuScenes dataloader 
    dataset_size= 20000
    train_size = int(dataset_size*0.8)
    val_size = dataset_size - train_size
    nusc_sampler_batch = sampler(dataset_size, args.batch_size)
    train_sampler_batch = sampler(train_size, args.batch_size)
    val_sampler_batch = sampler(val_size, args.batch_size)
  
    nusc_set = nuscenes_dataloader(args.batch_size, len(nusc_classes), training = True)
    training_set, validation_set = random_split(nusc_set, [train_size, val_size])
    nusc_dataloader = torch.utils.data.DataLoader(nusc_set, batch_size = args.batch_size , num_workers = args.num_workers, sampler = nusc_sampler_batch)
    train_loader = torch.utils.data.DataLoader(training_set, batch_size = args.batch_size , num_workers = args.num_workers, sampler = train_sampler_batch)
    val_loader = torch.utils.data.DataLoader(validation_set, batch_size = args.batch_size , num_workers = args.num_workers, sampler = val_sampler_batch)

    # Recording the args and configs for the experiment 
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
    train_iters_per_epoch = int(len(training_set) / args.batch_size)
    val_iters_per_epoch = int(len(validation_set) / args.batch_size)

    if args.use_tfboard:
        from tensorboardX import SummaryWriter
        logger = SummaryWriter("logs")

    accuracy = 0

    # Loading the experiment documentation files 
    f1 = open(output_dir + '/loss.txt','w+')
    f1.write("epoch    train    val\n")
    f1.close()
    f2 = open(output_dir + '/train_AP.txt','w+')
    f2.close()
    f3 = open(output_dir + '/val_AP.txt','w+')
    f3.close()
    f4 = open(output_dir + '/mAP.txt','w+')
    f4.close()

    evaluator = Evaluator()
    max_mAP = 0
    for epoch in range(args.start_epoch, args.max_epochs + 1):
        train_accuracy = []
        val_accuracy = [] 

        # setting to train mode
        fasterRCNN.train()
        train_loss_temp = 0
        train_loss_out = 0
        val_loss_temp = 0
        val_mAP = 0
        train_mAP = 0
        val_AP = []
        train_AP = []
        start = time.time()

        if epoch % (args.lr_decay_step + 1) == 0:
            adjust_learning_rate(optimizer, args.lr_decay_gamma)
            lr *= args.lr_decay_gamma

        nusc_iter = iter(nusc_dataloader)
        train_iter = iter(train_loader)
        val_iter = iter(val_loader)

        # training 
        allBoundingBoxes = BoundingBoxes()
        args.class_agnostic = True
        for step in range(train_iters_per_epoch):
            nusc_data = next(train_iter)
            with torch.no_grad():
                im_data.resize_(nusc_data[0].size()).copy_(nusc_data[0])
                im_info.resize_(nusc_data[1].size()).copy_(nusc_data[1])
                gt_boxes.resize_(nusc_data[2].size()).copy_(nusc_data[2])
                num_boxes.resize_(nusc_data[3].size()).copy_(nusc_data[3])
                image_path = functools.reduce(operator.add, (nusc_data[4]))
            
            fasterRCNN.zero_grad()
            
            # Saving gt_boxes for mAP evaluation 
            index = 0 
            for i in range(0, gt_boxes.size()[0]):
                index = step * args.batch_size + i 
                for box in gt_boxes[i].cpu().numpy():
                    if box[0]==0.0 and box[1]==0.0 and box[2]==0.0 and box[3]==0.0:
                        break
                    cls_i = int(box[4])
                    box = list(int(np.round(x/ nusc_data[1][0][2].item())) for x in box[:4])
                    bb= BoundingBox(index,cls_i,box[0],box[1],box[2],box[3],CoordinatesType.Absolute, None, BBType.GroundTruth, format=BBFormat.XYWH)
                    allBoundingBoxes.addBoundingBox(bb)

            rois, cls_prob, bbox_pred, \
            rpn_loss_cls, rpn_loss_box, \
            RCNN_loss_cls, RCNN_loss_bbox, \
            rois_label = fasterRCNN(im_data, im_info, gt_boxes, num_boxes)

            train_loss = rpn_loss_cls.mean() + rpn_loss_box.mean() \
               + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
            train_loss_temp += train_loss.item()
            train_loss_out += train_loss.item()

            # Saving predicted bounding boxes 
            for i in range(0, args.batch_size):
                index = step*args.batch_size + i
                allBoundingBoxes = get_bounding_boxes(rois[i].unsqueeze(0), cls_prob[i].unsqueeze(0), bbox_pred[i].unsqueeze(0), im_info[i].unsqueeze(0), allBoundingBoxes, index)
            
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
                    lsoss_rcnn_cls = RCNN_loss_cls.mean().item()
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

                print("[session %d][epoch %2d][iter %4d/%4d] loss: %.4f, lr: %.2e" % (args.session, epoch, step, train_iters_per_epoch, train_loss_out, lr))
                print("\t\t\tfg/bg=(%d/%d), time cost: %f" % (fg_cnt, bg_cnt, end-start))
                print("\t\t\trpn_cls: %.4f, rpn_box: %.4f, rcnn_cls: %.4f, rcnn_box %.4f" % (loss_rpn_cls, loss_rpn_box, loss_rcnn_cls, loss_rcnn_box))

                train_loss_out = 0
                start = time.time()

        # Evaluating mAP and loss 
        train_loss_temp = train_loss_temp/len(training_set)
        metricsPerClass = evaluator.GetPascalVOCMetrics(allBoundingBoxes, IOUThreshold=0.5)
        for mc in metricsPerClass:
            # Get metric values per each class
            c = mc['class']
            average_precision = mc['AP']
            train_mAP = train_mAP + average_precision
            train_AP = train_AP + [average_precision]
            # Print AP per class
            print('%s: %f' % (c, average_precision))
        
        train_mAP = train_mAP/ len(nusc_classes)
        print(train_mAP)
        
        # Validations
        if epoch % val_interval == 0:
            
            # Set to no grad for validation 
            with torch.no_grad():  
                val_iter = iter(val_loader)
                allBoundingBoxes = BoundingBoxes()
                
                for i in range(val_iters_per_epoch):   
                    data = next(val_iter)
                    im_data.resize_(data[0].size()).copy_(data[0])
                    im_info.resize_(data[1].size()).copy_(data[1])
                    gt_boxes.resize_(data[2].size()).copy_(data[2])
                    num_boxes.resize_(data[3].size()).copy_(data[3])
                    
                    # Saving gt boxes for mAP evaluation 
                    for ind in range(0, gt_boxes.size()[0]):
                        index = i*args.batch_size + ind 
                        for box in gt_boxes[ind].cpu().numpy():
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

                    val_loss = rpn_loss_cls.mean()+ rpn_loss_box.mean()\
                       + RCNN_loss_cls.mean() + RCNN_loss_bbox.mean()
                    val_loss_temp += val_loss.item()

                    # Saving predicted boxes for mAP evaluation 
                    for x in range(0, args.batch_size):
                        index = i*args.batch_size + x
                        allBoundingBoxes = get_bounding_boxes(rois[x].unsqueeze(0), cls_prob[x].unsqueeze(0), bbox_pred[x].unsqueeze(0), im_info[x].unsqueeze(0), allBoundingBoxes, index)

                # Evaluating mAP and loss 
                val_loss_temp = val_loss_temp/val_iters_per_epoch
                metricsPerClass = evaluator.GetPascalVOCMetrics(allBoundingBoxes, IOUThreshold=0.5)
                for mc in metricsPerClass:
                    c = mc['class']
                    average_precision = mc['AP']
                    val_mAP = val_mAP + average_precision
                    val_AP = val_AP + [average_precision]
                    # Print AP per class
                    print('%s: %f' % (c, average_precision))
                val_mAP = val_mAP/ len(nusc_classes)
                
                # Save validation results 
                f3 = open(output_dir + '/val_AP.txt','a+')
                f3.write("%.4f    %.4f    %.4f    %.4f    %.4f    %.4f    %.4f    %.4f    %.4f    %.4f\n" % (val_AP[0], val_AP[1], val_AP[2], val_AP[3], val_AP[4], val_AP[5], val_AP[6], val_AP[7], val_AP[8], val_AP[9]))
                f3.close()
                
                # Save model only if mAP is recorded maximum 
                if val_mAP > max_mAP:
                    max_mAP = val_mAP 
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

        # Save experiment loss and AP results 
        f1 = open(output_dir + '/loss.txt','a+')
        f1.write("%2d        %.4f    %.4f\n" % (epoch, train_loss_temp, val_loss_temp))
        f1.close()
        f2 = open(output_dir + '/train_AP.txt','a+')
        f2.write("%.4f    %.4f    %.4f    %.4f    %.4f    %.4f    %.4f    %.4f    %.4f    %.4f\n" % (train_AP[0], train_AP[1], train_AP[2], train_AP[3], train_AP[4], train_AP[5], train_AP[6], train_AP[7], train_AP[8], train_AP[9]))
        f2.close()
        f4 = open(output_dir + '/mAP.txt','a+')
        f4.write("%2d        %.4f    %.4f\n" % (epoch, train_mAP, val_mAP))
        f4.close()
     
    if args.use_tfboard:
        logger.close()

