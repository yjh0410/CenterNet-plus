import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatmapLoss(nn.Module):
    def __init__(self, alpha=2, beta=4, reduction='mean'):
        super(HeatmapLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

    def forward(self, logits, targets):
        inputs = torch.clamp(torch.sigmoid(logits), min=1e-4, max=1.0 - 1e-4)
        pos_ind = (targets == 1.0).float()
        neg_ind = (targets != 1.0).float()
        pos_loss = -pos_ind * (1.0 - inputs)**self.alpha * torch.log(inputs)
        neg_loss = -neg_ind * (1.0 - targets)**self.beta * (inputs)**self.alpha * torch.log(1.0 - inputs)
        loss = pos_loss + neg_loss

        if self.reduction == 'mean':
            batch_size = loss.size(0)
            loss = torch.sum(loss) / batch_size

        if self.reduction == 'sum':
            loss = torch.sum(loss) / batch_size

        return loss


def gaussian_radius(det_size, min_overlap=0.7):
    box_w, box_h  = det_size
    a1 = 1
    b1 = (box_w + box_h)
    c1 = box_w * box_h * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2
    # r1 = (b1 + sq1) / (2*a1)

    a2 = 4
    b2 = 2 * (box_w + box_h)
    c2 = (1 - min_overlap) * box_w * box_h
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2
    # r2 = (b2 + sq2) / (2*a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (box_w + box_h)
    c3 = (min_overlap - 1) * box_w * box_h
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    # r3 = (b3 + sq3) / (2*a3)

    return min(r1, r2, r3)


def generate_txtytwth(gt_label, w, h, s, gauss=False):
    x1, y1, x2, y2 = gt_label[:-1]
    # compute the center, width and height
    c_x = (x2 + x1) / 2 * w
    c_y = (y2 + y1) / 2 * h
    box_w = (x2 - x1) * w
    box_h = (y2 - y1) * h

    box_w_s = box_w / s
    box_h_s = box_h / s

    if gauss:
        r = gaussian_radius([box_w_s, box_h_s])
        r = max(int(r), 1)
        rw = rh = r
        # rw = max(int(box_w_s / 2), 1)
        # rh = max(int(box_h_s / 2), 1)
    else:
        r = None

    if box_w < 1e-4 or box_h < 1e-4:
        # print('A dirty data !!!')
        return False    

    # map center point of box to the grid cell
    c_x_s = c_x / s
    c_y_s = c_y / s
    grid_x = int(c_x_s)
    grid_y = int(c_y_s)
    # compute the (x, y, w, h) for the corresponding grid cell
    tx = c_x_s - grid_x
    ty = c_y_s - grid_y
    tw = np.log(box_w_s)
    th = np.log(box_h_s)
    weight = 2.0 - (box_w / w) * (box_h / h)

    return grid_x, grid_y, tx, ty, tw, th, weight, rw, rh, x1, y1, x2, y2


def gt_creator(input_size, stride, num_classes, label_lists=[], gauss=True):
    # prepare the all empty gt datas
    batch_size = len(label_lists)
    h = w = input_size
    
    ws = w // stride
    hs = h // stride
    s = stride
    gt_tensor = np.zeros([batch_size, hs, ws, num_classes+4+1+4])

    for batch_index in range(batch_size):
        for gt_label in label_lists[batch_index]:
            gt_cls = gt_label[-1]

            result = generate_txtytwth(gt_label, w, h, s, gauss=gauss)
            if result:
                grid_x, grid_y, tx, ty, tw, th, weight, rw, rh, x1, y1, x2, y2 = result

                gt_tensor[batch_index, grid_y, grid_x, int(gt_cls)] = 1.0
                gt_tensor[batch_index, grid_y, grid_x, num_classes:num_classes + 4] = np.array([tx, ty, tw, th])
                gt_tensor[batch_index, grid_y, grid_x, num_classes + 4] = weight
                gt_tensor[batch_index, grid_y, grid_x, num_classes + 5:] = np.array([x1, y1, x2, y2])

                if gauss:
                    # get the x1x2y1y2 for the target
                    x1, y1, x2, y2 = gt_label[:-1]
                    x1s, x2s = int(x1 * ws), int(x2 * ws)
                    y1s, y2s = int(y1 * hs), int(y2 * hs)
                    # create the grid
                    grid_x_mat, grid_y_mat = np.meshgrid(np.arange(x1s, x2s), np.arange(y1s, y2s))
                    # create a Gauss Heatmap for the target
                    heatmap = np.exp(-(grid_x_mat - grid_x)**2 / (2*(rw/3)**2) - \
                                      (grid_y_mat - grid_y)**2 / (2*(rh/3)**2))
                    p = gt_tensor[batch_index, y1s:y2s, x1s:x2s, int(gt_cls)]
                    gt_tensor[    batch_index, y1s:y2s, x1s:x2s, int(gt_cls)] = np.maximum(heatmap, p)
                
    gt_tensor = gt_tensor.reshape(batch_size, -1, num_classes+4+1+4)

    return torch.from_numpy(gt_tensor).float()


def iou_score(bboxes_a, bboxes_b, batch_size):
    """
        Input:\n
        bboxes_a : [B*N, 4] = [x1, y1, x2, y2] \n
        bboxes_b : [B*N, 4] = [x1, y1, x2, y2] \n

        Output:\n
        iou : [B, N] = [iou, ...] \n
    """
    tl = torch.max(bboxes_a[:, :2], bboxes_b[:, :2])
    br = torch.min(bboxes_a[:, 2:], bboxes_b[:, 2:])
    area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
    area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)

    en = (tl < br).type(tl.type()).prod(dim=1)
    area_i = torch.prod(br - tl, 1) * en  # * ((tl < br).all())
    iou = area_i / (area_a + area_b - area_i + 1e-14)

    return iou.view(batch_size, -1)


def loss(pred_cls, pred_txty, pred_twth, pred_iou, pred_iou_aware, label, num_classes):
    # create loss_f
    cls_loss_function = HeatmapLoss(reduction='mean')
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    twth_loss_function = nn.SmoothL1Loss(reduction='none')
    iou_loss_function = nn.SmoothL1Loss(reduction='none')
    iou_aware_loss_function = nn.BCEWithLogitsLoss(reduction='none')

    # pred
    pred_iou_aware = pred_iou_aware[:, :, 0]
    
    # groundtruth    
    gt_cls = label[:, :, :num_classes]
    gt_txty = label[:, :, num_classes : num_classes + 2]
    gt_twth = label[:, :, num_classes + 2 : num_classes + 4]
    gt_box_scale_weight = label[:, :, num_classes + 4]
    gt_mask = (gt_box_scale_weight > 0.).float()
    gt_iou = gt_mask.clone()
    # we use pred iou as the target of the iou aware.
    with torch.no_grad():
        gt_iou_aware = pred_iou.clone()

    batch_size = pred_cls.size(0)
    # obj loss
    cls_loss = cls_loss_function(pred_cls, gt_cls)
        
    # box loss
    txty_loss = torch.sum(torch.sum(txty_loss_function(pred_txty, gt_txty), dim=-1) * gt_box_scale_weight) / batch_size
    twth_loss = torch.sum(torch.sum(twth_loss_function(pred_twth, gt_twth), dim=-1) * gt_box_scale_weight) / batch_size

    # iou loss
    iou_loss = torch.sum(iou_loss_function(pred_iou, gt_iou) * gt_mask) / batch_size

    # iou aware loss
    iou_aware_loss = torch.sum(iou_aware_loss_function(pred_iou_aware, gt_iou_aware) * gt_mask) / batch_size

    return cls_loss, txty_loss, twth_loss, iou_loss, iou_aware_loss


if __name__ == "__main__":
    pass