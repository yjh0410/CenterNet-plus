import os
import argparse
from numpy import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from data import *
import numpy as np
import cv2
import time


parser = argparse.ArgumentParser(description='CenterNet-Plus')
parser.add_argument('-v', '--version', default='centernet_plus',
                    help='centernet_plus')
parser.add_argument('-bk', '--backbone', default='r18',
                    help='r18, r34, r50, r101')
parser.add_argument('-d', '--dataset', default='voc',
                    help='voc, coco-val.')
parser.add_argument('-size', '--input_size', default=512, type=int,
                    help='input_size')
parser.add_argument('--topk', default=100, type=int,
                    help='input_size')
parser.add_argument('--trained_model', default='weight/',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--show', action='store_true', default=False,
                    help='show the visulization results.')
parser.add_argument('-vs', '--visual_threshold', default=0.5, type=float,
                    help='Final confidence threshold')
parser.add_argument('--nms_thresh', default=0.45, type=float,
                    help='NMS threshold')
parser.add_argument('--cuda', action='store_true', default=False, 
                    help='use cuda.')
parser.add_argument('-nms', '--use_nms', action='store_true', default=False,
                    help='use nms.')
parser.add_argument('--save_folder', default='det_results/', type=str,
                    help='Dir to save results')

args = parser.parse_args()


def plot_bbox_labels(img, bbox, label, cls_color, text_scale=0.4):
    x1, y1, x2, y2 = bbox
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
    # plot bbox
    cv2.rectangle(img, (x1, y1), (x2, y2), cls_color, 2)
    # plot title bbox
    cv2.rectangle(img, (x1, y1-t_size[1]), (int(x1 + t_size[0] * text_scale), y1), cls_color, -1)
    # put the test on the title bbox
    cv2.putText(img, label, (int(x1), int(y1 - 5)), 0, text_scale, (0, 0, 0), 1, lineType=cv2.LINE_AA)

    return img


def visualize(img, bboxes, scores, cls_inds, vis_thresh, class_colors, class_names, class_indexs=None, dataset='voc'):
    ts = 0.4
    for i, bbox in enumerate(bboxes):
        if scores[i] > vis_thresh:
            if dataset == 'coco-val' or 'coco-test':
                cls_color = class_colors[int(cls_inds[i])]
                cls_id = class_indexs[int(cls_inds[i])]
            else:
                cls_id = int(cls_inds[i])
                cls_color = class_colors[cls_id]
            mess = '%s: %.2f' % (class_names[cls_id], scores[i])
            img = plot_bbox_labels(img, bbox, mess, cls_color, text_scale=ts)

    return img
        

def test(net, 
         device, 
         testset,
         transform, 
         vis_thresh, 
         class_colors=None, 
         class_names=None, 
         class_indexs=None,
         show=False,
         dataset='voc'):
    num_images = len(testset)
    save_path = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(save_path, exist_ok=True)

    for index in range(num_images):
        print('Testing image {:d}/{:d}....'.format(index+1, num_images))
        img, _ = testset.pull_image(index)
        h, w, _ = img.shape

        # to tensor
        x = torch.from_numpy(transform(img)[0][:, :, (2, 1, 0)]).permute(2, 0, 1)
        x = x.unsqueeze(0).to(device)

        t0 = time.time()
        # forward
        # inference
        bboxes, scores, cls_inds = net(x)
        print("detection time used ", time.time() - t0, "s")
        
        # scale each detection back up to the image
        scale = np.array([[w, h, w, h]])
        # map the boxes to origin image scale
        bboxes *= scale

        # vis detection
        img_processed = visualize(img=img,
                            bboxes=bboxes,
                            scores=scores,
                            cls_inds=cls_inds,
                            vis_thresh=vis_thresh,
                            class_colors=class_colors,
                            class_names=class_names,
                            class_indexs=class_indexs,
                            dataset=dataset
                            )
        if show:
            cv2.imshow('detection', img_processed)
            cv2.waitKey(0)
        # save result
        cv2.imwrite(os.path.join(save_path, str(index).zfill(6) +'.jpg'), img_processed)


if __name__ == '__main__':
    random.seed(0)
    # get device
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # img size
    input_size = args.input_size

    # dataset
    if args.dataset == 'voc':
        print('test on voc ...')
        class_names = VOC_CLASSES
        class_indexs = None
        num_classes = 20
        dataset = VOCDetection(root=VOC_ROOT, 
                               image_sets=[('2007', 'test')], 
                               transform=None)

    elif args.dataset == 'coco-val':
        print('test on coco-val ...')
        class_names = coco_class_labels
        class_indexs = coco_class_index
        num_classes = 80
        dataset = COCODataset(
                    data_dir=coco_root,
                    json_file='instances_val2017.json',
                    name='val2017',
                    img_size=input_size)

    class_colors = [(np.random.randint(255),
                     np.random.randint(255),
                     np.random.randint(255)) for _ in range(num_classes)]

    # load net
    if args.version == 'centernet_plus':
        from models.centernet_plus import CenterNetPlus
        net = CenterNetPlus(device=device, 
                            input_size=input_size, 
                            num_classes=num_classes,
                            backbone=args.backbone,
                            nms_thresh=args.nms_thresh, 
                            use_nms=args.use_nms)
                                 
    net.load_state_dict(torch.load(args.trained_model, map_location=device), strict=False)
    net.to(device).eval()
    print('Finished loading model!')

    # evaluation
    test(net=net, 
        device=device, 
        testset=dataset,
        transform=BaseTransform(input_size),
        vis_thresh=args.visual_threshold,
        class_colors=class_colors,
        class_names=class_names,
        class_indexs=class_indexs,
        show=args.show,
        dataset=args.dataset
        )
