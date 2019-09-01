import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from model.utils.config import cfg
from model.roi_layers import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.rpn.bbox_transform import clip_boxes
import pdb

from detection_metric.utils import *
from detection_metric.BoundingBox import BoundingBox
from detection_metric.BoundingBoxes import BoundingBoxes
from detection_metric.Evaluator import *

nusc_classes = ('__background__', 
                           'pedestrian', 'barrier', 'trafficcone', 'bicycle', 'bus', 'car', 'construction', 'motorcycle', 'trailer', 'truck')
                           
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

# Saves bounding boxes predicted by model in allBoundingBoxes object 
def get_bounding_boxes(rois, cls_prob, bbox_pred, im_info, allBoundingBoxes, index):
    global nusc_classes
    scores = cls_prob.data
    boxes = rois.data[:, :, 1:5]
    thresh = 0.05
    if cfg.TRAIN.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data
        if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
            box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                       + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
            box_deltas = box_deltas.view(1, -1, 4)
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
        # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    pred_boxes /= im_info[0][2].item()

    scores = scores.squeeze()
    pred_boxes = pred_boxes.squeeze()

    bounding_boxes = []
    for j in range(1, len(nusc_classes)):
        inds = torch.nonzero(scores[:,j]>thresh).view(-1)
        # if there is det
        if inds.numel() > 0:
            cls_scores = scores[:,j][inds]
            _, order = torch.sort(cls_scores, 0, True)
            cls_boxes = pred_boxes[inds, :]
        
            cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
            cls_dets = cls_dets[order]
            keep = nms(cls_boxes[order, :], cls_scores[order], 0.3)
            cls_dets = cls_dets[keep.view(-1).long()]
            dets = cls_dets.cpu().numpy()
            #pdb.set_trace()
            for i in range(np.minimum(10, dets.shape[0])):
                bbox = list(int(np.round(x)) for x in cls_dets.cpu().numpy()[i, :4])
                bbox = bbox + [j]
                score = dets[i,-1]
                if score>0.3:
                   bounding_boxes += [bbox]
                   bb= BoundingBox(index,j,bbox[0],bbox[1],bbox[2],bbox[3],CoordinatesType.Absolute, None, BBType.Detected, score, format=BBFormat.XYWH)
                   allBoundingBoxes.addBoundingBox(bb)
    return allBoundingBoxes
