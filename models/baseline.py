import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import Conv, ResizeConv, DilateEncoder, SPP
from utils import box_ops, loss
from backbone import *
import numpy as np

import os
import cv2


class Baseline(nn.Module):
    def __init__(self, device, input_size=None, trainable=False, num_classes=None, backbone='r18', conf_thresh=0.05, nms_thresh=0.45, topk=100, gs=1.0, use_nms=False):
        super(Baseline, self).__init__()
        self.device = device
        self.input_size = input_size
        self.trainable = trainable
        self.num_classes = num_classes
        self.bk = backbone
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.stride = 4
        self.topk = topk
        self.gs = gs
        self.use_nms = use_nms
        self.grid_cell = self.create_grid(input_size)


        # backbone
        if self.bk == 'r18':
            print("Use backbone : resnet-18")
            self.backbone = resnet18(pretrained=trainable)
            c2, c3, c4, c5 = 64, 128, 256, 512
            p2, p3, p4, p5 = 256, 256, 256, 256
            act = 'relu'
        
        elif self.bk == 'r50':
            print("Use backbone : resnet-50")
            self.backbone = resnet50(pretrained=trainable)
            c2, c3, c4, c5 = 256, 512, 1024, 2048
            p2, p3, p4, p5 = 256, 256, 256, 256
            act = 'relu'

        elif self.bk == 'r101':
            print("Use backbone : resnet-101")
            self.backbone = resnet101(pretrained=trainable)
            c2, c3, c4, c5 = 256, 512, 1024, 2048
            p2, p3, p4, p5 = 256, 256, 256, 256
            act = 'relu'

        elif self.bk == 'rx50':
            print("Use backbone : resnext-50")
            self.backbone = resnext50_32x4d(pretrained=trainable)
            c2, c3, c4, c5 = 256, 512, 1024, 2048
            p2, p3, p4, p5 = 256, 256, 256, 256
            act = 'relu'

        elif self.bk == 'rx101':
            print("Use backbone : resnext-101")
            self.backbone = resnext101_32x8d(pretrained=trainable)
            c2, c3, c4, c5 = 256, 512, 1024, 2048
            p2, p3, p4, p5 = 256, 256, 256, 256
            act = 'relu'

        else:
            print("Only support r18, r50, r101, rx50, rx101, d53, cspd53 as backbone !!")
            exit()
            
        # neck
        # # dilate encoder
        self.neck = Conv(c1=c5, c2=p5, k=1, act=act)
        # self.neck = DilateEncoder(c1=c5, c2=p5, act=act)
        
        # upsample
        self.deconv4 = ResizeConv(c1=p5, c2=p4, act=act, scale_factor=2) # 32 -> 16
        self.latter4 = Conv(c4, p4, k=1, act=None)
        self.smooth4 = Conv(p4, p4, k=3, p=1, act=act)

        self.deconv3 = ResizeConv(c1=p4, c2=p3, act=act, scale_factor=2) # 16 -> 8
        self.latter3 = Conv(c3, p3, k=1, act=None)
        self.smooth3 = Conv(p3, p3, k=3, p=1, act=act)

        self.deconv2 = ResizeConv(c1=p3, c2=p2, act=act, scale_factor=2) #  8 -> 4
        self.latter2 = Conv(c2, p2, k=1, act=None)
        self.smooth2 = Conv(p2, p2, k=3, p=1, act=act)
        

        # detection head
        self.cls_pred = nn.Sequential(
            Conv(p2, 64, k=3, p=1, act=act),
            nn.Conv2d(64, self.num_classes, kernel_size=1)
        )

        self.txty_pred = nn.Sequential(
            Conv(p2, 64, k=3, p=1, act=act),
            nn.Conv2d(64, 2, kernel_size=1)
        )
       
        self.twth_pred = nn.Sequential(
            Conv(p2, 64, k=3, p=1, act=act),
            nn.Conv2d(64, 2, kernel_size=1)
        )

        # init weight of cls_pred
        init_prob = 0.01
        bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
        nn.init.constant_(self.cls_pred[-1].bias, bias_value)


    def create_grid(self, input_size):
        h = w = input_size
        # generate grid cells
        ws, hs = w // self.stride, h // self.stride
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)])
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()
        grid_xy = grid_xy.view(1, hs*ws, 2).to(self.device)
        
        return grid_xy


    def set_grid(self, input_size):
        self.grid_cell = self.create_grid(input_size)
        self.input_size = input_size


    def decode_boxes(self, pred):
        """
        input box :  [delta_x, delta_y, sqrt(w), sqrt(h)]
        output box : [xmin, ymin, xmax, ymax]
        """
        output = torch.zeros_like(pred)
        pred[:, :, :2] = (self.grid_cell + self.gs * torch.sigmoid(pred[:, :, :2]) - (self.gs - 1.0) / 2) * self.stride
        pred[:, :, 2:] = (torch.exp(pred[:, :, 2:])) * self.stride

        # [c_x, c_y, w, h] -> [xmin, ymin, xmax, ymax]
        output[:, :, 0] = pred[:, :, 0] - pred[:, :, 2] / 2
        output[:, :, 1] = pred[:, :, 1] - pred[:, :, 3] / 2
        output[:, :, 2] = pred[:, :, 0] + pred[:, :, 2] / 2
        output[:, :, 3] = pred[:, :, 1] + pred[:, :, 3] / 2
        
        return output


    def _gather_feat(self, feat, ind, mask=None):
        dim  = feat.size(2)
        ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat


    def _topk(self, scores):
        B, C, H, W = scores.size()
        
        topk_scores, topk_inds = torch.topk(scores.view(B, C, -1), self.topk)

        topk_inds = topk_inds % (H * W)
        
        topk_score, topk_ind = torch.topk(topk_scores.view(B, -1), self.topk)
        topk_clses = (topk_ind / self.topk).int()
        topk_inds = self._gather_feat(topk_inds.view(B, -1, 1), topk_ind).view(B, self.topk)

        return topk_score, topk_inds, topk_clses


    def nms(self, dets, scores):
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)                 # the size of bbox
        order = scores.argsort()[::-1]                        # sort bounding boxes by decreasing order

        keep = []                                             # store the final bounding boxes
        while order.size > 0:
            i = order[0]                                      #the index of the bbox with highest confidence
            keep.append(i)                                    #save it to keep
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h

            # Cross Area / (bbox + particular area - Cross Area)
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]

        return keep


    def vis_fmap(self, fmap, normal=True, name='p3'):
        """ fmap = [C, H, W] """
        save_path = os.path.join('vis_feat/Baseline/' + name)
        os.makedirs(save_path, exist_ok=True)
        f = fmap
        
        f = torch.sum(fmap, dim=0)
        if normal:
            # normalization
            max_val = torch.max(f)
            min_val = torch.min(f)
            f = (f - min_val) / (max_val - min_val)
        f = f.cpu().numpy()
        # resize
        f = cv2.resize(f, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)
        # plt.imsave(os.path.join(save_path, name+'.jpg'), f)
        cv2.imwrite(os.path.join(save_path, name+'.jpg'), (f*255).astype(np.uint8))


    def forward(self, x, target=None):
        # backbone
        c2, c3, c4, c5 = self.backbone(x)
        B = c5.size(0)

        # bottom-up
        p5 = self.neck(c5)
        p4 = self.smooth4(self.latter4(c4) + self.deconv4(p5))
        p3 = self.smooth3(self.latter3(c3) + self.deconv3(p4))
        p2 = self.smooth2(self.latter2(c2) + self.deconv2(p3))

        # detection head
        cls_pred = self.cls_pred(p2)
        txty_pred = self.txty_pred(p2)
        twth_pred = self.twth_pred(p2)
        
        # train
        if self.trainable:
            # [B, H*W, num_classes]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, self.num_classes)
            # [B, H*W, 2]
            txty_pred = txty_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 2)
            # [B, H*W, 2]
            twth_pred = twth_pred.permute(0, 2, 3, 1).contiguous().view(B, -1, 2)

            # compute iou between pred bboxes and gt bboxes
            txtytwth_pred = torch.cat([txty_pred, twth_pred], dim=-1)
            x1y1x2y2_pred = (self.decode_boxes(txtytwth_pred) / self.input_size).view(-1, 4)
            x1y1x2y2_gt = target[:, :, -4:].view(-1, 4)
            iou_pred = box_ops.iou_score(x1y1x2y2_pred, x1y1x2y2_gt, batch_size=B)

            # compute loss
            cls_loss, txty_loss, twth_loss, iou_loss, iou_aware_loss = loss.loss_base(
                                                                        pred_cls=cls_pred, 
                                                                        pred_txty=txty_pred, 
                                                                        pred_twth=twth_pred, 
                                                                        pred_iou=iou_pred,
                                                                        label=target, 
                                                                        num_classes=self.num_classes
                                                                        )
            
            return cls_loss, txty_loss, twth_loss, iou_loss, iou_aware_loss  

        # test
        else:
            with torch.no_grad():
                # batch_size = 1
                cls_pred = torch.sigmoid(cls_pred)

                # # visual class prediction
                # self.vis_fmap(p2[0], normal=True, name='p2')    
                # self.vis_fmap(p3[0], normal=True, name='p3')    
                # self.vis_fmap(p4[0], normal=True, name='p4')    
                # self.vis_fmap(p5[0], normal=True, name='p5')    
                # self.vis_fmap(c5[0], normal=True, name='c5')    
                # self.vis_fmap(cls_pred[0], normal=False, name='cls_pred')    

                # simple nms
                hmax = F.max_pool2d(cls_pred, kernel_size=5, padding=2, stride=1)
                keep = (hmax == cls_pred).float()
                cls_pred *= keep

                # decode box
                txtytwth_pred = torch.cat([txty_pred, twth_pred], dim=1).permute(0, 2, 3, 1).contiguous().view(B, -1, 4)
                # [B, H*W, 4] -> [H*W, 4]
                bbox_pred = torch.clamp((self.decode_boxes(txtytwth_pred) / self.input_size)[0], 0., 1.)

                # topk
                topk_scores, topk_inds, topk_clses = self._topk(cls_pred)

                topk_scores = topk_scores[0].cpu().numpy()
                topk_cls_inds = topk_clses[0].cpu().numpy()
                topk_bbox_pred = bbox_pred[topk_inds[0]].cpu().numpy()

                if self.use_nms:
                    # nms
                    keep = np.zeros(len(topk_bbox_pred), dtype=np.int)
                    for i in range(self.num_classes):
                        inds = np.where(topk_cls_inds == i)[0]
                        if len(inds) == 0:
                            continue
                        c_bboxes = topk_bbox_pred[inds]
                        c_scores = topk_scores[inds]
                        c_keep = self.nms(c_bboxes, c_scores)
                        keep[inds[c_keep]] = 1

                    keep = np.where(keep > 0)
                    topk_bbox_pred = topk_bbox_pred[keep]
                    topk_scores = topk_scores[keep]
                    topk_cls_inds = topk_cls_inds[keep]

                return topk_bbox_pred, topk_scores, topk_cls_inds
                