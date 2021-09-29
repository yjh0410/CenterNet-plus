# CenterNet-plus
A Simple Baseline for Object Detection based on CenterNet with ResNet backbone.

CenterNet is a very simple yet efficient object detector. Based on this supreme work,
I rebuild it with PyTorch.

CenterNet is an encoder-decoder network, but I won't consider Hourglass-101 in this
project as it is heavy and time consuming.

I will try DLA-34 in the future, but let us focus ResNet for now.

## Weight
You can download all my models from my BaiDuYunDisk:

Link: https://pan.baidu.com/s/1yaAhIT6NErzv_QaDHqfYrg 

Password: zr8m

I will upload them to Google Drive.

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

The whole structure of my CenterNet-plus is shown in the following picture:

![Image](https://github.com/yjh0410/CenterNet-plus/blob/main/img_files/centernet-plus.jpg)

I dont deploy any DCN in my CenterNet-plus.

## Train on COCO
For example, you can use the following line of command to train my CenterNet-plus with ResNet-18 backbone.

```Shell
python train.py -d coco --cuda --ema -bk r18 
```

## Experimental results

### COCO
Experimental results on COCO:

<table><tbody>
<tr><th align="left" bgcolor=#f8f8f8> </th>     <td bgcolor=white> backbone </td><td bgcolor=white> data </td><td bgcolor=white> size </td><td bgcolor=white> AP </td><td bgcolor=white> AP50 </td><td bgcolor=white> AP75 </td><td bgcolor=white> AP_S </td><td bgcolor=white> AP_M </td><td bgcolor=white> AP_L </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> CenterNet-plus</th><td bgcolor=white> ResNet-18 </td><td bgcolor=white> COCO val </td><td bgcolor=white> 512x512 </td><td bgcolor=white> 29.9 </td><td bgcolor=white> 49.1 </td><td bgcolor=white> 31.8 </td><td bgcolor=white> 14.4 </td><td bgcolor=white> 31.0 </td><td bgcolor=white> 43.1 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> CenterNet-plus</th><td bgcolor=white> ResNet-50 </td><td bgcolor=white> COCO val </td><td bgcolor=white> 512x512 </td><td bgcolor=white> 36.0 </td><td bgcolor=white> 56.2 </td><td bgcolor=white> 38.9 </td><td bgcolor=white> 18.7 </td><td bgcolor=white> 38.9 </td><td bgcolor=white> 51.1 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> CenterNet-plus</th><td bgcolor=white> ResNet-101 </td><td bgcolor=white> COCO val </td><td bgcolor=white> 512x512 </td><td bgcolor=white> 37.5 </td><td bgcolor=white> 57.7 </td><td bgcolor=white> 41.0 </td><td bgcolor=white> 19.5 </td><td bgcolor=white> 41.4 </td><td bgcolor=white> 53.0 </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> CenterNet</th><td bgcolor=white> ResNet-18 </td><td bgcolor=white> COCO val </td><td bgcolor=white> 512x512 </td><td bgcolor=white> 28.1 </td><td bgcolor=white> 44.9 </td><td bgcolor=white> 29.6 </td><td bgcolor=white> - </td><td bgcolor=white> - </td><td bgcolor=white> - </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> CenterNet</th><td bgcolor=white> ResNet-101 </td><td bgcolor=white> COCO val </td><td bgcolor=white> 512x512 </td><td bgcolor=white> 34.6 </td><td bgcolor=white> 53.0 </td><td bgcolor=white> 36.9 </td><td bgcolor=white> - </td><td bgcolor=white> - </td><td bgcolor=white> - </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> CenterNet</th><td bgcolor=white> DLA-34 </td><td bgcolor=white> COCO val </td><td bgcolor=white> 512x512 </td><td bgcolor=white> 37.4 </td><td bgcolor=white> 55.1 </td><td bgcolor=white> 40.8 </td><td bgcolor=white> - </td><td bgcolor=white> - </td><td bgcolor=white> - </td></tr>

<tr><th align="left" bgcolor=#f8f8f8> CenterNet</th><td bgcolor=white> Hourglass-104 </td><td bgcolor=white> COCO val </td><td bgcolor=white> 512x512 </td><td bgcolor=white> 40.3 </td><td bgcolor=white> 59.1 </td><td bgcolor=white> 44.0 </td><td bgcolor=white> - </td><td bgcolor=white> - </td><td bgcolor=white> - </td></tr>

</table></tbody>

With ResNet backbone, my CenterNet-plus works better.
