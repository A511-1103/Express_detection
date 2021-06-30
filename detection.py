import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import  cv2 as cv
import  time
from yolov3.yolov4 import Create_Yolo
from yolov3.configs import *
import glob


def postprocess_boxes(pred_bbox, original_image, input_size, score_threshold):
    pred_bbox = np.array(pred_bbox)
    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_angel = pred_bbox[:, 5:(5+180)]
    pred_prob = pred_bbox[:, (5+180):]
    pred_coor = pred_xywh
    pred_angel = np.argmax(pred_angel,axis=-1)
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    mask = scores > score_threshold
    coors, scores, angel, classes = pred_coor[mask], scores[mask], pred_angel[mask], classes[mask]
    pred_coor, scores, pred_angel, classes=coors, scores, angel,classes
    mask_l = pred_coor[:, 2] < input_size
    pred_coor, scores, pred_angel, classes = pred_coor[mask_l], scores[mask_l], pred_angel[mask_l], classes[mask_l]
    mask_h = pred_coor[:, 3] < input_size
    pred_coor, scores, pred_angel, classes = pred_coor[mask_h], scores[mask_h], pred_angel[mask_h], classes[mask_h]
    return np.concatenate([pred_coor, pred_angel[:, np.newaxis], classes[:, np.newaxis], scores[:, np.newaxis]], axis=-1)


def xyls2xywh(pred):
    long_side = pred[:,2:3]
    short_side = pred[:,3:4]
    theta = pred[:,4:5] - 180
    flag = theta < (-90)
    theta = np.where(flag, theta+90, theta)
    w = np.where(flag, short_side, long_side)
    h = np.where(flag, long_side, short_side)
    out = np.concatenate([pred[:, :2], w, h, theta, pred[:, 5:]], axis=-1)
    return out


def compute_rotate_iou(best_bbox, other_bbox, thre=0.3):
    area1 = best_bbox[2] * best_bbox[3]
    area2 = other_bbox[:, 2] * other_bbox[:, 3]
    all_iou_result = []
    r1 = ((best_bbox[0], best_bbox[1]), (best_bbox[2], best_bbox[3]), best_bbox[4])
    for i in range(len(other_bbox)):
        r2 = ((other_bbox[i, 0], other_bbox[i, 1]), (other_bbox[i, 2], other_bbox[i, 3]), other_bbox[i, 4])
        int_pts = cv.rotatedRectangleIntersection(r1, r2)[1]
        if int_pts is not None:
            order_pts = cv.convexHull(int_pts, returnPoints=True)
            int_area = cv.contourArea(order_pts)
            inter = int_area * 1.0 / (area1 + area2[i] - int_area)
            all_iou_result.append(inter)
        else:
            all_iou_result.append(0.0)
    all_iou_result = np.array(all_iou_result)
    return all_iou_result < thre


def NMS(bboxes, thre=0.3):
    class_in_img = set(list(bboxes[:, 5]))
    best_bboxes = []
    for cls in class_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]
        while len(cls_bboxes) > 0:
            score_max_idx = np.argmax(cls_bboxes[:, 6])
            cur_best_bboxes = cls_bboxes[score_max_idx]
            best_bboxes.append(cur_best_bboxes)
            cls_bboxes = np.concatenate((cls_bboxes[:score_max_idx, :], cls_bboxes[(score_max_idx + 1):, :]), axis=0)
            iou_mask = compute_rotate_iou(cur_best_bboxes, cls_bboxes,thre=thre)
            cls_bboxes = cls_bboxes[iou_mask]
    return best_bboxes


def draw_roate_bboxes1(images,scale_images, bboxes, save_path,input_size=416):
    rotate_image = images.copy()
    ih, iw, _ = images.shape
    for i in range(len(bboxes)):
        gray_label = np.zeros((input_size, input_size), np.uint8)
        x, y, w, h, theta = bboxes[i, :5]
        rect = ((x, y), (w, h), theta)
        poly = np.float32(cv.boxPoints(rect))
        poly = np.int0(poly)
        cv.drawContours(gray_label, [poly], -1, 255, cv.FILLED)
        cv.drawContours(image=scale_images,
                        contours=[poly],
                        contourIdx=-1,
                        color=[255, 0, 0],
                        thickness=2)
        gray_label = cv.cvtColor(gray_label, cv.COLOR_GRAY2BGR)
        gray_label = cv.resize(gray_label, (iw, ih))
        gray_label = np.where(gray_label > 0, 1, 0)
        target = rotate_image * gray_label
        rows, cols = target.shape[:2]
        a, b = cols / 2, rows / 2
        target = np.array(target, np.uint8)
        gray_label = np.array(gray_label, np.uint8)
        M = cv.getRotationMatrix2D(center=(a, b), angle=theta, scale=1)
        rotated_img = cv.warpAffine(target, M, (cols, rows))  # 旋转后的图像保持大小不变
        rotate_lab = cv.warpAffine(gray_label, M, (cols, rows))  # 旋转后的图像保持大小不变
        cur_lab = rotate_lab[:, :, 0].copy()
        res = cv.findContours(cur_lab, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_NONE)
        cont, idx = res
        if len(cont) == 0:
            continue
        x_1, y_1, w_1, h_1 = cv.boundingRect(cont[0])
        final_crop_img = rotated_img[int(y_1):int(y_1 + h_1), int(x_1):int(x_1 + w_1), :]
        name=save_path+'/'+'{}.png'.format(i)
        cv.imwrite(name, final_crop_img)
        # 对快递单区域进行截取,得到快递单区域,对其区域进行旋转,得到水平矩形区域

    name = save_path + '/' + 'plot.jpg'
    cv.imwrite(name, scale_images)
    # 原始图片,进行了pad处理,需要pad的原始是有些快递单位于边界处,将导致坐标越界
    return images


def image_preprocess(image,target_size=[416,416],fill_value=128):
    ih, iw = target_size
    h, w, _ = image.shape
    scale = min(iw / w, ih / h)
    oh, ow = image.shape[:2]
    oh, ow = max(oh, ow), max(oh, ow)
    org_scale = min(ow/image.shape[1], oh/image.shape[0])
    # long side scale
    nw, nh = int(scale * w), int(scale * h)
    now, noh = int(org_scale*image.shape[1]),int(org_scale*image.shape[0])
    # scale size
    image_resized = cv.resize(image, (nw, nh))
    org_resized = cv.resize(image,(now, noh))
    # resize
    image_paded = np.full(shape=[ih, iw, 3], fill_value=fill_value)
    org_paded = np.full(shape=[oh,ow,3], fill_value=fill_value)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    odw, odh = (ow-now)//2, (oh-noh)//2
    # pad short
    image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
    org_paded[odh:odh + noh, odw:odw + now,:] = org_resized
    scale_images = image_paded
    image_paded = image_paded/255.0
    return org_paded, scale_images, image_paded


def detect_image(Yolo, image_path, input_size=416,
                 score_threshold=0.5, iou_threshold=0.1, save_path='org'):
    original_image = cv.imread(image_path)
    # original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
    initial_scale, scale_images, image_data = image_preprocess(np.copy(original_image))
    # 对图片进行letter_bbox
    image_data = image_data[np.newaxis, ...].astype(np.float32)
    # [416,416,3]----------->[1,416,4163]
    t1 = time.time()
    pred_bbox = Yolo(image_data)
    #  num_classes = 2
    # 【[1,52,52,3,(5 + num_classes + 180)],[1,52,52,3,(5 + num_classes + 180)],[1,52,52,3,(5 + num_classes + 180 )]】
    pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
    pred_bbox = tf.concat(pred_bbox, axis=0)
    # [10647,5+180+num_classes]
    bboxes = postprocess_boxes(pred_bbox, original_image, input_size, score_threshold)
    # 1.删除置信度较低的结果
    # 2.删除无效的框
    bboxes = xyls2xywh(bboxes)
    bboxes = NMS(bboxes,thre=iou_threshold)
    bboxes = np.array(bboxes)
    direct_path = save_path
    # 检测结果的保存路径,当前路径下创建一个子文件夹,名称为org
    name = os.path.basename(image_path)
    name = name[:-4]
    # name = image_path.split('.jpg')[0].split('\\')[-1]
    #  获取当前检测图片的名字
    direct = direct_path + '/' + name
    flag = os.path.exists(direct)
    if not flag:
        os.makedirs(direct,exist_ok=True)
    # 创建org子文件夹以及图片名称子文件夹
    plot_images = draw_roate_bboxes1(initial_scale, scale_images, bboxes, direct)
    # x,y,w,h,theta,classes_idx,score
    t2 = time.time()
    print('图片{}检测完成,其检测结果保存至文件夹:{},检测时间花费:{}秒'.format(name, direct, round(t2-t1,2)))
    # return plot_images, bboxes
    return round(t2-t1, 2), time.ctime()


if __name__ == '__main__':
    all_image_path = glob.glob("D:/ydata/images_dir/*.png")
    # 待检测图片的路径

    print('待检测图片的数量为:{}'.format(len(all_image_path)))
    yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
    yolo.load_weights(YOLO_CUSTOM_WEIGHTS)  # use custom weights
    for i in range(len(all_image_path)):
        detect_image(yolo,
                    all_image_path[i],
                    input_size=416,
                    score_threshold=0.7,
                    iou_threshold=0.1)