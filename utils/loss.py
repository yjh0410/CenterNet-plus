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


def loss_base(pred_cls, pred_txty, pred_twth, pred_iou, label, num_classes):
    # create loss_f
    cls_loss_function = HeatmapLoss(reduction='mean')
    txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')
    twth_loss_function = nn.SmoothL1Loss(reduction='none')
    # iou_loss_function = nn.SmoothL1Loss(reduction='none')

    # groundtruth    
    gt_cls = label[:, :, :num_classes]
    gt_txty = label[:, :, num_classes : num_classes + 2]
    gt_twth = label[:, :, num_classes + 2 : num_classes + 4]
    gt_box_scale_weight = label[:, :, num_classes + 4]
    gt_mask = (gt_box_scale_weight > 0.).float()
    gt_iou = gt_mask.clone()

    batch_size = pred_cls.size(0)
    # obj loss
    cls_loss = cls_loss_function(pred_cls, gt_cls)
        
    # box loss
    txty_loss = torch.sum(torch.sum(txty_loss_function(pred_txty, gt_txty), dim=-1) * gt_box_scale_weight) / batch_size
    twth_loss = torch.sum(torch.sum(twth_loss_function(pred_twth, gt_twth), dim=-1) * gt_box_scale_weight) / batch_size

    # iou loss
    iou_loss = torch.tensor([0], requires_grad=False).to(pred_txty.device) #torch.sum(iou_loss_function(pred_iou, gt_iou) * gt_mask) / batch_size

    # iou aware loss (For baseline model, we dont consider iou-aware loss)
    iou_aware_loss = torch.tensor([0], requires_grad=False).to(pred_txty.device)

    return cls_loss, txty_loss, twth_loss, iou_loss, iou_aware_loss



if __name__ == "__main__":
    pass