# TODO: transfer numpy to tensorflow operations
import os
import cv2
import random
import numpy as np
import tensorflow as tf
from yolov3.utils import read_class_names, image_preprocess
from yolov3.yolov3 import bbox_iou
from yolov3.configs import *
import  math


def gaussian_label(label, num_class, u=0, sig=4.0):
    '''
    转换成CSL Labels：
        用高斯窗口函数根据角度θ的周期性赋予gt labels同样的周期性，使得损失函数在计算边界处时可以做到“差值很大但loss很小”；
        并且使得其labels具有环形特征，能够反映各个θ之间的角度距离
    @param label: 当前box的θ类别  shape(1)
    @param num_class: θ类别数量=180
    @param u: 高斯函数中的μ
    @param sig: 高斯函数中的σ
    @return: 高斯离散数组:将高斯函数的最高值设置在θ所在的位置，例如label为45，则将高斯分布数列向右移动直至x轴为45时，取值为1 shape(180)
    '''
    # floor()返回数字的下舍整数   ceil() 函数返回数字的上入整数  range(-90,90)
    # 以num_class=180为例，生成从-90到89的数字整形list  shape(180)
    x = np.array(range(math.floor(-num_class / 2), math.ceil(num_class / 2), 1))
    #     print(x)
    y_sig = np.exp(-(x - u) ** 2 / (2 * sig ** 2))  # shape(180) 为-90到89的经高斯公式计算后的数值
    # 将高斯函数的最高值设置在θ所在的位置，例如label为45，则将高斯分布数列向右移动直至x轴为45时，取值为1
    #     print(y_sig)
    return np.concatenate([y_sig[math.ceil(num_class / 2) - label:],
                           y_sig[:math.ceil(num_class / 2) - label]], axis=0)


def up_down_augment(img, bboxes):
    img = cv2.flip(img, 0)
    w, h = img.shape[:2]
    bboxes[:, 1] = w - bboxes[:, 1]
    bboxes[:, -1] = 180 - bboxes[:, -1]
    bboxes[bboxes[:, -1] == 180, -1] = 0
    return img, bboxes


def left_right_augment(img, bboxes):
    img = cv2.flip(img, 1)
    w, h = img.shape[:2]
    bboxes[:, 0] = h - bboxes[:, 0]
    # 竖直翻转,x不变，y相对图片大小变
    bboxes[:, -1] = 180 - bboxes[:, -1]
    bboxes[bboxes[:, -1] == 180, -1] = 0
    return img, bboxes


def rand(a=0, b=1):
    return np.random.rand() * (b - a) + a


def xyxy2xywh(bbox):
    x_min = bbox[:, 0:1]
    y_min = bbox[:, 1:2]
    x_max = bbox[:, 2:3]
    y_max = bbox[:, 3:4]
    angel = bbox[:, 4:5]
    classes_idx = bbox[:, 5:6]

    x = (x_min + x_max) / 2
    y = (y_min + y_max) / 2
    w = (x_max - x_min)
    h = (y_max - y_min)

    out = np.concatenate((x, y, w, h, angel, classes_idx), axis=-1)
    return out


def xywh2xyxy(bbox):
    x = bbox[:, 0:1]
    y = bbox[:, 1:2]
    w = bbox[:, 2:3]
    h = bbox[:, 3:4]
    angel = bbox[:, 4:5]
    classes_idx = bbox[:, 5:6]

    x_min = x - w / 2
    y_min = y - h / 2
    x_max = x + w / 2
    y_max = y + h / 2

    out = np.concatenate((x_min, y_min, x_max, y_max, angel, classes_idx), axis=-1)
    return out


def xywh2xyls(bbox):
    # [-90,0)----->[-180,0)
    w = bbox[:, 2:3]
    h = bbox[:, 3:4]
    theta = bbox[:, 4:5]
    classed_idx = bbox[:, 5:6]

    wh_flag = w > h
    long_side = np.where(wh_flag, w, h)
    short_side = np.where(wh_flag, h, w)
    theta = np.where(wh_flag, theta, theta - 90)

    theta = (theta + 180.5)
    theta = np.array(theta, np.uint8)
    flag = (theta == 180)
    theta = np.where(flag, 179, theta)

    out = np.concatenate((bbox[:, :2], long_side, short_side, classed_idx, theta), axis=-1)

    return out


def xyls2xywh(pred):
    '''
    :param pred:  [num1,4+1+] xyls theta
    :return:
    '''

    long_side = pred[:, 2:3]
    short_side = pred[:, 3:4]
    classes_idx = pred[:, 4:5]
    theta = pred[:, 5:6] - 180

    # [0,179]------->[-180,0)
    flag = theta < (-90)
    theta = np.where(flag, theta + 90, theta)
    # [-90,0)

    w = np.where(flag, short_side, long_side)
    h = np.where(flag, long_side, short_side)

    out = np.concatenate((pred[:, :2], w, h, theta, classes_idx), axis=-1)

    return out


def ls2wh_format(x, y, l, s, theta):
    if theta >= -180 and theta < -90:
        w = s
        h = l
        theta = theta + 90
    else:
        w = l
        h = s
        theta = theta
    return ((x, y), (w, h), theta)


def wh2ls_format(x, y, w, h, theta):
    if theta == 0:
        theta = -90
    if theta > 0:
        if theta == 90:
            print('theta 出现异常情况，导致该现象的原因是w或者h为零')
        return False
    if theta < -90:
        print('角度范围超出了opencv的表示法')
        return False
    if w >= h:
        long_side = w
        short_side = h
        theta_long = theta
    else:
        long_side = h
        short_side = w
        theta_long = theta - 90
    return x, y, long_side, short_side, theta_long


def image_preprocess(image, target_size, gt_boxes):
    ih, iw = target_size
    h, w, _ = image.shape
    '''
    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2

    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
    image_paded = image_paded / 255.

    # print(scale, ih, iw, h, w, nh, nw, dh, dw)
    #     print(scale,dw,dh)
    #     x,y,l,s,c,angel    angel-->[0,179]
    #     print(np.unique(gt_boxes[:,-1]))
    gt_boxes = xyls2xywh(gt_boxes)
    #     x,y,w,h,theta,c     theta--->[-90,0)
    #     print(np.unique(gt_boxes[:,-2]))
    gt_boxes = xywh2xyxy(gt_boxes)
    #     x,y,x,y,theta,c     theta-->[-90,0)
    #     print(np.unique(gt_boxes[:,-2]))

    gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
    gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh

    gt_boxes = xyxy2xywh(gt_boxes)
    #     print(np.unique(gt_boxes[:,-2]))
    #     x,y,w,h,theta,c     theta-->[-90,0)
    gt_boxes = xywh2xyls(gt_boxes)
    #    x,y,l,s,c,theta     theta--->[0,179]
    #     print(np.unique(gt_boxes[:,-1]))
    '''
    image_paded = image/255.
    gt_boxes = gt_boxes
    return image_paded, gt_boxes


def rotate_augment(angel, scale, image, labels):
    Pi_angle = -angel * np.math.pi / 180.0
    rows, cols = image.shape[:2]
    a, b = cols / 2, rows / 2
    M = cv2.getRotationMatrix2D(center=(a, b), angle=angel, scale=scale)
    rotated_img = cv2.warpAffine(image, M, (cols, rows))  # 旋转后的图像保持大小不变
    rotated_labels = []
    for label in labels:
        x, y, l, s, classes_idx,theta = label
            # 标签中的角度信息为[0,179]变为对应的[-180,0)
        rect = ls2wh_format(x, y, l, s, theta - 179.9)
            # 由ls对应的[-180,0)变回对应的[-90,0)
        poly = cv2.boxPoints(rect)

        X0 = (poly[0][0] - a) * math.cos(Pi_angle) - (poly[0][1] - b) * math.sin(Pi_angle) + a
        Y0 = (poly[0][0] - a) * math.sin(Pi_angle) + (poly[0][1] - b) * math.cos(Pi_angle) + b

        X1 = (poly[1][0] - a) * math.cos(Pi_angle) - (poly[1][1] - b) * math.sin(Pi_angle) + a
        Y1 = (poly[1][0] - a) * math.sin(Pi_angle) + (poly[1][1] - b) * math.cos(Pi_angle) + b

        X2 = (poly[2][0] - a) * math.cos(Pi_angle) - (poly[2][1] - b) * math.sin(Pi_angle) + a
        Y2 = (poly[2][0] - a) * math.sin(Pi_angle) + (poly[2][1] - b) * math.cos(Pi_angle) + b

        X3 = (poly[3][0] - a) * math.cos(Pi_angle) - (poly[3][1] - b) * math.sin(Pi_angle) + a
        Y3 = (poly[3][0] - a) * math.sin(Pi_angle) + (poly[3][1] - b) * math.cos(Pi_angle) + b

        poly_rotated = np.array([(X0, Y0), (X1, Y1), (X2, Y2), (X3, Y3)])
        ((x, y), (w, h), theta) = cv2.minAreaRect(np.float32(poly_rotated))
        x, y, l, s, theta = wh2ls_format(x, y, w, h, theta)
            # 由[-90,0)变回对应的[-180,0)
            # 再由对应的[-180,0)变回分类所需的[0,179]
        if x <= 1 or y <= 1 or l >= 416 or s >= 416 or x >= 416 or y >= 416 or l <= 1 or s <= 1:
            continue

        theta = int(theta + 180.5)  # range int[0,180] 四舍五入
        if theta == 180:  # range int[0,179]
            theta = 179
        rotated_labels.append([x, y, l, s, classes_idx,theta])

    return rotated_img, np.array(rotated_labels)


def distoy_img(image, hue=.1, sat=1.5, val=1.5):
    hue = rand(-hue, hue)
    sat = rand(1, sat) if rand() < .5 else 1 / rand(1, sat)
    val = rand(1, val) if rand() < .5 else 1 / rand(1, val)
    x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_RGB2HSV)
    x[..., 0] += hue * 360
    x[..., 0][x[..., 0] > 1] -= 1
    x[..., 0][x[..., 0] < 0] += 1
    x[..., 1] *= sat
    x[..., 2] *= val
    x[x[:, :, 0] > 360, 0] = 360
    x[:, :, 1:][x[:, :, 1:] > 1] = 1
    x[x < 0] = 0

    image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

    image_data = np.array(image_data, np.uint8)

    return image_data


class Dataset(object):
    # Dataset preprocess implementation
    def __init__(self, dataset_type, TEST_INPUT_SIZE=TEST_INPUT_SIZE, mosaic_augment=False):
        self.annot_path  = TRAIN_ANNOT_PATH if dataset_type == 'train' else TEST_ANNOT_PATH
        self.input_sizes = TRAIN_INPUT_SIZE if dataset_type == 'train' else TEST_INPUT_SIZE
        self.batch_size  = TRAIN_BATCH_SIZE if dataset_type == 'train' else TEST_BATCH_SIZE
        self.data_aug    = TRAIN_DATA_AUG   if dataset_type == 'train' else TEST_DATA_AUG
        self.train_input_sizes = TRAIN_INPUT_SIZE
        self.strides = np.array(YOLO_STRIDES)
        self.classes = read_class_names(TRAIN_CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = (np.array(YOLO_ANCHORS).T/self.strides).T
        self.anchor_per_scale = YOLO_ANCHOR_PER_SCALE
        self.max_bbox_per_scale = YOLO_MAX_BBOX_PER_SCALE
        self.annotations = self.load_annotations(dataset_type)
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0
        self.mosaic_augment = mosaic_augment
    #     r-box无法处理越界的边界,否则直接4合1,不进行裁剪,模拟增大BS,降低BN带来的影响,或者梯度累计法模拟更大的BS

    def load_annotations(self, dataset_type):
        final_annotations = []
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        
        for annotation in annotations:
            # fully parse annotations
            line = annotation.split()
            image_path, index = "", 1
            image_path = line[0]
            # for i, one_line in enumerate(line):
            #     if not one_line.replace(",","").isnumeric():
            #         if image_path != "": image_path += " "
            #         image_path += one_line
            #     else:
            #         index = i
            #         break
            # print(image_path)
            if not os.path.exists(image_path):
                print(image_path)
                raise KeyError("%s does not exist ... " %image_path)
            if TRAIN_LOAD_IMAGES_TO_RAM:
                image = cv2.imread(image_path)
            else:
                image = ''
            final_annotations.append([image_path, line[index:], image])
            # 【图片路径，目标信息，图片（w,h,3）的一个数组】
            # 【图片路径，目标信息，''】未读取到内存时
        return final_annotations

    def __iter__(self):
        return self

    def Delete_bad_annotation(self, bad_annotation):
        print(f'Deleting {bad_annotation} annotation line')
        bad_image_path = bad_annotation[0]
        bad_image_name = bad_annotation[0].split('/')[-1] # can be used to delete bad image
        bad_xml_path = bad_annotation[0][:-3]+'xml' # can be used to delete bad xml file

        # remove bad annotation line from annotation file
        with open(self.annot_path, "r+") as f:
            d = f.readlines()
            f.seek(0)
            for i in d:
                if bad_image_name not in i:
                    f.write(i)
            f.truncate()
    
    def __next__(self):
        with tf.device('/cpu:0'):
            self.train_input_size = random.choice([self.train_input_sizes])
            self.train_output_sizes = self.train_input_size // self.strides

            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3), dtype=np.float32)

            batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                          self.anchor_per_scale, 5 + 180 + self.num_classes), dtype=np.float32)

            batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                          self.anchor_per_scale, 5 + 180 + self.num_classes), dtype=np.float32)

            batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                          self.anchor_per_scale, 5 + 180 + self.num_classes), dtype=np.float32)

            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 5), dtype=np.float32)
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 5), dtype=np.float32)
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 5), dtype=np.float32)

            exceptions = False
            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples: index -= self.num_samples
                    if self.mosaic_augment:
                        annotation = self.annotations[index:index+4]
                        image = self.mosaic_parse_annotation(annotation)
                    else:
                        annotation = self.annotations[index]
                        image, bboxes = self.parse_annotation(annotation)

                    '''
                    if len(bboxes)==0:
                        continue
                    '''

                    bboxes =  np.array(bboxes,np.int)
                    try:
                        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes)

                    except IndexError:
                        exceptions = True
                        self.Delete_bad_annotation(annotation)
                        print("IndexError, something wrong with", annotation[0], "removed this line from annotation file")

                    batch_image[num, :, :, :] = image

                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    # [batch_size,52,52,3,6+num_classes]
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    # [batch_size,26,26,3,6+num_classes]
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    # [batch_size,13,13,3,6+num_classes]

                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1

                if exceptions: 
                    print('\n')
                    raise Exception("There were problems with dataset, I fixed them, now restart the training process.")
                    # 数据集标注的问题产生报错
                self.batch_count += 1
                # 开启下一个批次的迭代
                batch_smaller_target = batch_label_sbbox, batch_sbboxes
                batch_medium_target  = batch_label_mbbox, batch_mbboxes
                batch_larger_target  = batch_label_lbbox, batch_lbboxes

                return batch_image, (batch_smaller_target, batch_medium_target, batch_larger_target)


            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def random_horizontal_flip(self, image, bboxes):
        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]]

        return image, bboxes

    def random_crop(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def merge_bboxes(bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                x1, y1, x2, y2, classes_idx, angel = box[0], box[1], box[2], box[3], box[4], box[5]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                        if y2 - y1 < 5:
                            continue
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx
                        if x2 - x1 < 5:
                            continue

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue

                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                        if y2 - y1 < 5:
                            continue

                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx
                        if x2 - x1 < 5:
                            continue

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue

                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                        if y2 - y1 < 5:
                            continue

                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                        if x2 - x1 < 5:
                            continue

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue

                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                        if y2 - y1 < 5:
                            continue

                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                        if x2 - x1 < 5:
                            continue

                x,y,long_side,short_side = x1,y1,x2,y2
                merge_bbox.append([x,y,long_side,short_side,classes_idx,angel])
        return merge_bbox

    def mosaic_parse_annotation(self,annotation,mAP=False):
        h ,w =self.input_sizes,self.input_sizes

        min_offset_x = self.rand(0.4, 0.6)
        min_offset_y = self.rand(0.4, 0.6)

        place_x = [0, 0, int(w * min_offset_x), int(w * min_offset_x)]
        place_y = [0, int(h * min_offset_y), int(h * min_offset_y), 0]
        # 坐标的起始点

        nws = [int(w * min_offset_x), int(w * min_offset_x), w - int(w * min_offset_x), w - int(w * min_offset_x)]
        nhs = [int(h * min_offset_y), h - int(h * min_offset_y), h - int(h * min_offset_y), int(h * min_offset_y)]

        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])

        if TRAIN_LOAD_IMAGES_TO_RAM:
            image1_path = annotation[0][0]
            image2_path = annotation[1][0]
            image3_path = annotation[2][0]
            image4_path = annotation[3][0]

            image1 = annotation[0][2]
            image2 = annotation[1][2]
            image3 = annotation[2][2]
            image4 = annotation[3][2]

            bboxes1 = np.array([list(map(int, box.split(','))) for box in annotation[0][1]])
            bboxes2 = np.array([list(map(int, box.split(','))) for box in annotation[1][1]])
            bboxes3 = np.array([list(map(int, box.split(','))) for box in annotation[2][1]])
            bboxes4 = np.array([list(map(int, box.split(','))) for box in annotation[3][1]])

            image1, bboxes1 = image_preprocess(np.copy(image1), [self.input_sizes, self.input_sizes], np.copy(bboxes1))
            image2, bboxes2 = image_preprocess(np.copy(image2), [self.input_sizes, self.input_sizes], np.copy(bboxes2))
            image3, bboxes3 = image_preprocess(np.copy(image3), [self.input_sizes, self.input_sizes], np.copy(bboxes3))
            image4, bboxes4 = image_preprocess(np.copy(image4), [self.input_sizes, self.input_sizes], np.copy(bboxes4))

        else:
            image1_path = annotation[0][0]
            image2_path = annotation[1][0]
            image3_path = annotation[2][0]
            image4_path = annotation[3][0]

            image1 = cv2.imread(image1_path)
            image2 = cv2.imread(image2_path)
            image3 = cv2.imread(image3_path)
            image4 = cv2.imread(image4_path)

            bboxes1 = np.array([list(map(int, box.split(','))) for box in annotation[0][1]])
            bboxes2 = np.array([list(map(int, box.split(','))) for box in annotation[1][1]])
            bboxes3 = np.array([list(map(int, box.split(','))) for box in annotation[2][1]])
            bboxes4 = np.array([list(map(int, box.split(','))) for box in annotation[3][1]])

            image1, bboxes1 = image_preprocess(np.copy(image1), [self.input_sizes, self.input_sizes], np.copy(bboxes1))
            image2, bboxes2 = image_preprocess(np.copy(image2), [self.input_sizes, self.input_sizes], np.copy(bboxes2))
            image3, bboxes3 = image_preprocess(np.copy(image3), [self.input_sizes, self.input_sizes], np.copy(bboxes3))
            image4, bboxes4 = image_preprocess(np.copy(image4), [self.input_sizes, self.input_sizes], np.copy(bboxes4))

        new_image[:cuty, :cutx, :] = image1[:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image2[cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image3[cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image4[:cuty, cutx:, :]

        bboxes = [bboxes1, bboxes2, bboxes3, bboxes4]

        bboxes = self.merge_bboxes(bboxes, cutx, cuty)

        return new_image, bboxes

    def parse_annotation(self, annotation, mAP = 'False'):
        if TRAIN_LOAD_IMAGES_TO_RAM:
            image_path = annotation[0]
            image = annotation[2]
        else:
            image_path = annotation[0]
            image = cv2.imread(image_path)
            # print(image.shape)


        bboxes = np.array([list(map(int, box.split(','))) for box in annotation[1]])
        # [singal_img_target_num,6]
        # [x,y,w,h,class_idx,angel]
        # print(image_path, image.shape,bboxes)
        '''
        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))
        '''

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # [singal_img_target_num,6]
        # [x_min,y_min,x_max,y_max,class_idx,angel]
        if mAP == True:
            return image, bboxes

        '''
        if len(bboxes)==0:
            return  image, bboxes
        '''

        image, bboxes = image_preprocess(np.copy(image), [self.input_sizes, self.input_sizes], np.copy(bboxes))
        return image, bboxes

    def preprocess_true_boxes(self, bboxes):
        # bboxes======>[taget,6]
        # [x_min,y_min,x_max,y_max,class_idx,angel]
        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + 180 + self.num_classes)) for i in range(3)]

        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 5)) for _ in range(3)]

        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]
            bbox_class_ind = bbox[4]

            bbox_angel = bbox[5]

            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
            # label_smooth,Inception-V2提出
            # bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            bbox_xywh = bbox_coor
            #  x,y,l,s,theta

            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False
            # print('*'*50)
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 5))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                anchors_xywh[:, 2:4] = self.anchors[i]

                def ls2wh_format(x, y, l, s, theta):
                    if theta >= -180 and theta < -90:
                        w = s
                        h = l
                        theta = theta + 90
                    else:
                        w = l
                        h = s
                        theta = theta
                    return ((x, y), (w, h), theta)

                def iou_rotate_calculate(boxes1, boxes2, gt_angel, prior_angel=-90, multi_scale=3):
                    area1 = boxes1[:, 2] * boxes1[:, 3]
                    area2 = boxes2[:, 2] * boxes2[:, 3]
                    iou_s = []
                    for i, box1 in enumerate(boxes1):
                        temp_ious = []

                        r1 = ls2wh_format(box1[0], box1[1], box1[2], box1[3], gt_angel)
                        for j, box2 in enumerate(boxes2):
                            r2 = ((box2[0], box2[1]), (box2[2], box2[3]), prior_angel)

                            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]

                            if int_pts is not None:
                                order_pts = cv2.convexHull(int_pts, returnPoints=True)

                                int_area = cv2.contourArea(order_pts)
                                # print(int_area)

                                inter = int_area * 1.0 / (area1[i] + area2[j] - int_area)
                                temp_ious.append(inter)
                            else:
                                temp_ious.append(0.0)
                        iou_s.append(temp_ious)
                    # print(iou_s)
                    # print('='*50)
                    iou_s = np.array(iou_s, dtype=np.float32)
                    iou_s = np.reshape(iou_s, multi_scale)
                    return np.array(iou_s, dtype=np.float32)

                iou_scale = iou_rotate_calculate(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh, bbox_angel)

                # print('*50')
                # print(iou_scale)

                # print(iou_scale,iou_scale.shape)
                iou.append(iou_scale)
                '''
                iou阈值可以适当减小
                '''
                iou_mask = iou_scale > 0.4

                if np.any(iou_mask):
                    # 如果至少有一个结果是大于0.3的
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0

                    angel_label = gaussian_label(label=bbox_angel,num_class=180,u=0, sig=6)
                    label[i][yind, xind, iou_mask, 5:(5+180)] = angel_label
                    label[i][yind, xind, iou_mask, (5+180):] = smooth_onehot
                    if yind-1>=0:
                        label[i][yind-1, xind, iou_mask, 0:4] = bbox_xywh
                        label[i][yind-1, xind, iou_mask, 4:5] = 1.0
                        label[i][yind-1, xind, iou_mask, 5:(5 + 180)] = angel_label
                        label[i][yind-1, xind, iou_mask, (5 + 180):] = smooth_onehot
                    if xind-1>=0:
                        label[i][yind, xind-1, iou_mask, 0:4] = bbox_xywh
                        label[i][yind, xind-1, iou_mask, 4:5] = 1.0
                        label[i][yind, xind-1, iou_mask, 5:(5 + 180)] = angel_label
                        label[i][yind, xind-1, iou_mask, (5 + 180):] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    # 记录这是第几个符合条件的先验框，[0,100)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bboxes_xywh[i][bbox_ind, 4:5] = bbox_angel
                    bbox_count[i] += 1
                    exist_positive = True
            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0

                angel_label = gaussian_label(label=bbox_angel, num_class=180, u=0, sig=6)
                label[best_detect][yind, xind, best_anchor, 5:(5+180)] = angel_label

                label[best_detect][yind, xind, best_anchor, (5+180):] = smooth_onehot
                if yind-1>=0:
                    label[best_detect][yind-1, xind, best_anchor, :] = 0
                    label[best_detect][yind-1, xind, best_anchor, 0:4] = bbox_xywh
                    label[best_detect][yind-1, xind, best_anchor, 4:5] = 1.0
                    label[best_detect][yind-1, xind, best_anchor, 5:(5 + 180)] = angel_label
                    label[best_detect][yind-1, xind, best_anchor, (5 + 180):] = smooth_onehot
                if xind-1>=0:
                    label[best_detect][yind , xind-1, best_anchor, :] = 0
                    label[best_detect][yind , xind-1, best_anchor, 0:4] = bbox_xywh
                    label[best_detect][yind , xind-1, best_anchor, 4:5] = 1.0
                    label[best_detect][yind , xind-1, best_anchor, 5:(5 + 180)] = angel_label
                    label[best_detect][yind , xind-1, best_anchor, (5 + 180):] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bboxes_xywh[best_detect][bbox_ind, 4:5] = bbox_angel

                bbox_count[best_detect] += 1

        # 迭代完了一张图片中的所有目标
        label_sbbox, label_mbbox, label_lbbox = label
        #  得到该图片所对应的掩码图
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batchs
