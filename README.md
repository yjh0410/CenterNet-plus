# CenterNetv2
A Simple Baseline for Object Detection

CenterNet is a very simple yet efficient object detector. Based on this supreme work,
I rebuild it with PyTorch.

CenterNet is an encoder-decoder network, but I won't consider Hourglass-101 or DLA-34 in this
project as they are both too heavy and time consuming.

## Backbone
For backbone, I use ResNet including ResNet-18, ResNet-50 and ResNet-101. I also make use of
DarkNet-53 and CSPDarkNet-53 reproduced by myself with PyTorch.

## Neck
For neck, I use the DilateEncoder proposed by YOLOF. DCN or DCNv2 don't be considered as they
are not easy to deploy.

## Decoder
For decoder, I just use nearest interpolate following a convolutional layer rather than deconvolutional
layer to avoid the effect of checkerboard.

## Detection head
For detection head, there are totally 4 branches including class branch, offset(tx and ty) branch, size(tw and th)
branch and iou-aware branch. During training stage, how to get the labels of offset and size is different from
CenterNet. For more details, see my codes in ```tools.py``` file.

The whole structure of my CenterNetv2 is shown in the following picture:

![Image](https://github.com/yjh0410/CenterNetv2/blob/main/img_files/centernetv2.jpg)

## Train on COCO
For example, you can use the following line of command to train my CenterNetv2 with ResNet-18 backbone.

```Shell
python train.py -d coco --cuda --gauss --ema -bk r18 
```


COCO:

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> backbone </td><td bgcolor=white> data </td><td bgcolor=white> AP </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white> AP_S </td><td bgcolor=white> AP_M </td><td bgcolor=white> AP_L </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> CenterNetv2</th><td bgcolor=white> ResNet-18 </td><td bgcolor=white> COCO val </td><td bgcolor=white> 21.6 </td><td bgcolor=white> 44.0 </td><td bgcolor=white> 19.2 </td><td bgcolor=white> 5.0 </td><td bgcolor=white> 22.4 </td><td bgcolor=white> 35.5 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> CenterNetv2</th><td bgcolor=white> ResNet-50 </td><td bgcolor=white> COCO val </td><td bgcolor=white> 21.6 </td><td bgcolor=white> 44.0 </td><td bgcolor=white> 19.2 </td><td bgcolor=white> 5.0 </td><td bgcolor=white> 22.4 </td><td bgcolor=white> 35.5 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> CenterNetv2</th><td bgcolor=white> ResNet-101 </td><td bgcolor=white> COCO val </td><td bgcolor=white> 21.6 </td><td bgcolor=white> 44.0 </td><td bgcolor=white> 19.2 </td><td bgcolor=white> 5.0 </td><td bgcolor=white> 22.4 </td><td bgcolor=white> 35.5 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> CenterNetv2</th><td bgcolor=white> DarkNet-53 </td><td bgcolor=white> COCO val </td><td bgcolor=white> 21.6 </td><td bgcolor=white> 44.0 </td><td bgcolor=white> 19.2 </td><td bgcolor=white> 5.0 </td><td bgcolor=white> 22.4 </td><td bgcolor=white> 35.5 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> CenterNetv2</th><td bgcolor=white> CSPDarkNet-53 </td><td bgcolor=white> COCO val </td><td bgcolor=white> 21.6 </td><td bgcolor=white> 44.0 </td><td bgcolor=white> 19.2 </td><td bgcolor=white> 5.0 </td><td bgcolor=white> 22.4 </td><td bgcolor=white> 35.5 </td></tr>

</table></tbody>
