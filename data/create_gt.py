import numpy as np
import torch


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


def generate_txtytwth(gt_label, w, h, s):
    x1, y1, x2, y2 = gt_label[:-1]
    # compute the center, width and height
    c_x = (x2 + x1) / 2 * w
    c_y = (y2 + y1) / 2 * h
    box_w = (x2 - x1) * w
    box_h = (y2 - y1) * h

    box_w_s = box_w / s
    box_h_s = box_h / s

    r = gaussian_radius([box_w_s, box_h_s])
    r = max(int(r), 1)
    rw = rh = r

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


def gt_creator(img_size, stride, num_classes, label_lists=[]):
    # prepare the all empty gt datas
    h = w = img_size
    
    ws = w // stride
    hs = h // stride
    s = stride
    gt_tensor = np.zeros([hs, ws, num_classes+4+1+4])

    for gt_label in label_lists:
        cls_id = int(gt_label[-1])

        result = generate_txtytwth(gt_label, w, h, s)
        if result:
            grid_x, grid_y, tx, ty, tw, th, weight, rw, rh, x1, y1, x2, y2 = result

            gt_tensor[grid_y, grid_x, cls_id] = 1.0
            gt_tensor[grid_y, grid_x, num_classes:num_classes + 4] = np.array([tx, ty, tw, th])
            gt_tensor[grid_y, grid_x, num_classes + 4] = weight
            gt_tensor[grid_y, grid_x, num_classes + 5:] = np.array([x1, y1, x2, y2])

            # get the x1x2y1y2 for the target
            x1, y1, x2, y2 = gt_label[:-1]
            x1s, x2s = int(x1 * ws), int(x2 * ws)
            y1s, y2s = int(y1 * hs), int(y2 * hs)
            # create the grid
            grid_x_mat, grid_y_mat = np.meshgrid(np.arange(x1s, x2s), np.arange(y1s, y2s))
            # create a Gauss Heatmap for the target
            heatmap = np.exp(-(grid_x_mat - grid_x)**2 / (2*(rw/3)**2) - \
                                (grid_y_mat - grid_y)**2 / (2*(rh/3)**2))
            p = gt_tensor[y1s:y2s, x1s:x2s, cls_id]
            gt_tensor[    y1s:y2s, x1s:x2s, cls_id] = np.maximum(heatmap, p)
            
    gt_tensor = gt_tensor.reshape(-1, num_classes+4+1+4)

    return torch.from_numpy(gt_tensor).float()

