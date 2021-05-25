from __future__ import division

import os
import random
import argparse
import time
import math
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn

from data import *
import tools

from utils.augmentations import SSDAugmentation, ColorAugmentation
from utils.cocoapi_evaluator import COCOAPIEvaluator
from utils.vocapi_evaluator import VOCAPIEvaluator
from utils.modules import ModelEMA

def parse_args():
    parser = argparse.ArgumentParser(description='CenterNetv2 Detection')
    parser.add_argument('-v', '--version', default='centernetv2',
                        help='centernetv2')
    parser.add_argument('-bk', '--backbone', default='r18',
                        help='r18, r34, r50, r101')
    parser.add_argument('-d', '--dataset', default='voc',
                        help='voc or coco')
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='use multi-scale trick')                  
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--mosaic', action='store_true', default=False,
                        help='use mosaic augmentation.')
    parser.add_argument('--gauss', action='store_true', default=False,
                        help='use gauss smooth.')
    parser.add_argument('--ema', action='store_true', default=False,
                        help='use ema.')
    parser.add_argument('--batch_size', default=32, type=int, 
                        help='Batch size for training')
    parser.add_argument('--lr', default=1e-3, type=float, 
                        help='initial learning rate')
    parser.add_argument('-no_wp', '--no_warm_up', action='store_true', default=False,
                        help='yes or no to choose using warmup strategy to train')
    parser.add_argument('--wp_epoch', type=int, default=1,
                        help='The upper bound of warm-up')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch to train')
    parser.add_argument('-r', '--resume', default=None, type=str, 
                        help='keep training')
    parser.add_argument('--momentum', default=0.9, type=float, 
                        help='Momentum value for optim')
    parser.add_argument('--weight_decay', default=5e-4, type=float, 
                        help='Weight decay for SGD')
    parser.add_argument('--gamma', default=0.1, type=float, 
                        help='Gamma update for SGD')
    parser.add_argument('--num_workers', default=8, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--eval_epoch', type=int,
                            default=10, help='interval between evaluations')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug mode where only one image is trained')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='Gamma update for SGD')

    return parser.parse_args()


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # gauss smooth
    if args.gauss:
        print('use Gauss Smooth Labels ...')

    # mosaic ema
    if args.ema:
        print('use EMA ...')

    # mosaic augmentation
    if args.mosaic:
        print('use Mosaic Augmentation ...')

    # multi-scale
    if args.multi_scale:
        print('use the multi-scale trick ...')
        train_size = 640
        val_size = 512
    else:
        train_size = 512
        val_size = 512

    # config
    cfg = train_cfg

    # dataset and evaluator
    if args.dataset == 'voc':
        data_dir = VOC_ROOT
        num_classes = 20
        dataset = VOCDetection(root=data_dir, 
                                transform=SSDAugmentation(train_size),
                                base_transform=ColorAugmentation(train_size),
                                mosaic=args.mosaic
                                )

        evaluator = VOCAPIEvaluator(data_root=data_dir,
                                    device=device,
                                    transform=BaseTransform(val_size),
                                    labelmap=VOC_CLASSES
                                    )

    elif args.dataset == 'coco':
        data_dir = coco_root
        num_classes = 80
        dataset = COCODataset(
                    data_dir=data_dir,
                    img_size=train_size,
                    transform=SSDAugmentation(train_size),
                    base_transform=ColorAugmentation(train_size),
                    debug=args.debug,
                    mosaic=args.mosaic
                    )


        evaluator = COCOAPIEvaluator(
                        data_dir=data_dir,
                        img_size=val_size,
                        device=device,
                        transform=BaseTransform(val_size)
                        )
    
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)
    
    print('Training model on:', dataset.name)
    print('The dataset size:', len(dataset))
    print("----------------------------------------------------------")

    # dataloader
    dataloader = torch.utils.data.DataLoader(
                    dataset, 
                    batch_size=args.batch_size, 
                    shuffle=True, 
                    collate_fn=detection_collate,
                    num_workers=args.num_workers,
                    pin_memory=True
                    )

    # build model
    if args.version == 'centernetv2':
        from models.centernetv2 import CenterNetv2
        
        net = CenterNetv2(device=device, 
                          input_size=train_size, 
                          num_classes=num_classes, 
                          trainable=True, 
                          backbone=args.backbone)
        print('Let us train centernet on the %s dataset ......' % (args.dataset))

    else:
        print('Unknown version !!!')
        exit()

    model = net
    model.to(device).train()
    ema = ModelEMA(model) if args.ema else None

    # use tfboard
    if args.tfboard:
        print('use tensorboard')
        from torch.utils.tensorboard import SummaryWriter
        c_time = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        log_path = os.path.join('log/coco/', args.version, c_time)
        os.makedirs(log_path, exist_ok=True)

        writer = SummaryWriter(log_path)
    
    # keep training
    if args.resume is not None:
        print('keep training model: %s' % (args.resume))
        model.load_state_dict(torch.load(args.resume, map_location=device))

    # optimizer setup
    base_lr = args.lr
    tmp_lr = base_lr
    optimizer = optim.SGD(model.parameters(), 
                            lr=args.lr, 
                            momentum=args.momentum,
                            weight_decay=args.weight_decay
                            )

    max_epoch = cfg['max_epoch']
    epoch_size = len(dataset) // args.batch_size

    # start training loop
    t0 = time.time()

    for epoch in range(args.start_epoch, max_epoch):

        # use step lr
        if epoch in cfg['lr_epoch']:
            tmp_lr = tmp_lr * 0.1
            set_lr(optimizer, tmp_lr)
    

        for iter_i, (images, targets) in enumerate(dataloader):
            # WarmUp strategy for learning rate
            if not args.no_warm_up:
                if epoch < args.wp_epoch:
                    tmp_lr = base_lr * pow((iter_i+epoch*epoch_size)*1. / (args.wp_epoch*epoch_size), 4)
                    # tmp_lr = 1e-6 + (base_lr-1e-6) * (iter_i+epoch*epoch_size) / (epoch_size * (args.wp_epoch))
                    set_lr(optimizer, tmp_lr)

                elif epoch == args.wp_epoch and iter_i == 0:
                    tmp_lr = base_lr
                    set_lr(optimizer, tmp_lr)
        

            # multi-scale trick
            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                # randomly choose a new size
                train_size = random.randint(10, 20) * 32
                model.set_grid(train_size)
            if args.multi_scale:
                # interpolate
                images = torch.nn.functional.interpolate(images, size=train_size, mode='bilinear', align_corners=False)
            
            # make train label
            targets = [label.tolist() for label in targets]
            # vis data
            # 可视化数据，以便查看预处理部分是否有问题，将下面两行取消注释即可
            # vis_data(images, targets, train_size)
            # continue
            targets = tools.gt_creator(input_size=train_size, 
                                        stride=net.stride,
                                        num_classes=num_classes,
                                        label_lists=targets, 
                                        gauss=args.gauss
                                        )
            # 可视化高斯热力图
            # vis_heatmap(targets)
            # continue
            # to device
            images = images.to(device)
            targets = targets.to(device)

            # forward and loss
            cls_loss, txty_loss, twth_loss, iou_loss, iou_aware_loss = model(images, target=targets)

            # loss
            total_loss = cls_loss + txty_loss + twth_loss + iou_loss + iou_aware_loss

            # backprop
            total_loss.backward()        
            optimizer.step()
            optimizer.zero_grad()

            # ema
            if args.ema:
                ema.update(model)

            if iter_i % 10 == 0:
                if args.tfboard:
                    # viz loss
                    writer.add_scalar('class loss', cls_loss.item(),   iter_i + epoch * epoch_size)
                    writer.add_scalar('txty loss',  txty_loss.item(),  iter_i + epoch * epoch_size)
                    writer.add_scalar('twth loss',  twth_loss.item(),  iter_i + epoch * epoch_size)
                    writer.add_scalar('iou loss',   iou_loss.item(),   iter_i + epoch * epoch_size)
                    writer.add_scalar('iou-aw loss', iou_aware_loss.item(),   iter_i + epoch * epoch_size)
                    writer.add_scalar('total loss', total_loss.item(), iter_i + epoch * epoch_size)
                
                t1 = time.time()
                print('[Epoch %d/%d][Iter %d/%d][lr %.6f]'
                    '[Loss: cls %.2f || txty %.2f || twth %.2f || iou %.2f || aw %.2f || size %d || time: %.2f]'
                        % (epoch+1, max_epoch, iter_i, epoch_size, tmp_lr,
                            cls_loss.item(), 
                            txty_loss.item(), 
                            twth_loss.item(), 
                            iou_loss.item(),
                            iou_aware_loss.item(),
                            train_size, 
                            t1-t0),
                            flush=True
                            )

                t0 = time.time()

        # evaluation
        if (epoch + 1) % args.eval_epoch == 0:
            if args.ema:
                model_eval = ema.ema
            else:
                model_eval = model

            model_eval.trainable = False
            model_eval.set_grid(val_size)
            model_eval.eval()

            # evaluate
            evaluator.evaluate(model_eval)

            # convert to training mode.
            model_eval.trainable = True
            model_eval.set_grid(train_size)
            model_eval.train()

            # save model
            print('Saving state, epoch:', epoch + 1)
            torch.save(model_eval.state_dict(), os.path.join(path_to_save, 
                        args.version + '_' + repr(epoch + 1) + '.pth'),
                        _use_new_zipfile_serialization=False
                        )  


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def vis_heatmap(targets):
    # vis heatmap
    HW = targets.shape[1]
    h = int(np.sqrt(HW))
    for c in range(20):
        heatmap = targets[0, :, c].reshape(h, h).cpu().numpy()
        name = VOC_CLASSES[c]
        heatmap = cv2.resize(heatmap, (640, 640))
        cv2.imshow(name, heatmap)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def vis_data(images, targets, input_size):
    # vis data
    mean=(0.406, 0.456, 0.485)
    std=(0.225, 0.224, 0.229)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    img = images[0].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
    img = ((img * std + mean)*255).astype(np.uint8)
    cv2.imwrite('1.jpg', img)

    img_ = cv2.imread('1.jpg')
    for box in targets[0]:
        xmin, ymin, xmax, ymax = box[:-1]
        # print(xmin, ymin, xmax, ymax)
        xmin *= input_size
        ymin *= input_size
        xmax *= input_size
        ymax *= input_size
        cv2.rectangle(img_, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)

    cv2.imshow('img', img_)
    cv2.waitKey(0)


if __name__ == '__main__':
    train()
