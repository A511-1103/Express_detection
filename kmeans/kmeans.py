import glob
import random
import xml.etree.ElementTree as ET
import numpy as np

def cas_iou(box, cluster):
    x = np.minimum(cluster[:, 0], box[0])
    y = np.minimum(cluster[:, 1], box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cluster[:, 0] * cluster[:, 1]
    iou = intersection / (area1 + area2 - intersection)

    return iou


def avg_iou(box, cluster):
    return np.mean([np.max(cas_iou(box[i], cluster)) for i in range(box.shape[0])])


def kmeans(box, k):
    # 取出一共有多少框
    row = box.shape[0]

    # 每个框各个点的位置
    distance = np.empty((row, k))

    # 最后的聚类位置
    last_clu = np.zeros((row,))

    np.random.seed()

    # 随机选5个当聚类中心
    cluster = box[np.random.choice(row, k, replace=False)]
    # cluster = random.sample(row, k)
    while True:
        # 计算每一行距离五个点的iou情况。
        for i in range(row):
            distance[i] = 1 - cas_iou(box[i], cluster)

        # 取出最小点
        near = np.argmin(distance, axis=1)

        if (last_clu == near).all():
            break

        # 求每一个类的中位点
        for j in range(k):
            cluster[j] = np.median(
                box[near == j], axis=0)

        last_clu = near

    return cluster


def xyls2xywh(pred):
    '''
    :param pred:  [num1,4+1+] xyls theta
    :return:
    '''
    long_side = pred[:,2:3]
    short_side = pred[:,3:4]
    classes_idx = pred[:,4:5]
    theta = pred[:,5:6] - 180

    # [0,179]------->[-180,0)
    flag= theta<(-90)
    theta = np.where(flag,theta+90,theta)
    # [-90,0)

    w=np.where(flag,short_side,long_side)
    h=np.where(flag,long_side,short_side)

    out= np.concatenate((pred[:,:2],w,h,theta,classes_idx),axis=-1)

    return out


def load_data():
    txt = open('train.txt', mode='r')
    dataset = []
    txt = txt.readlines()

    annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
    for anno in annotations:
        line = anno.split()
        cur_anno = line[1:]
        bboxes = np.array([list(map(int, box.split(','))) for box in cur_anno])
        bboxes = xyls2xywh(bboxes)
        for i in range(len(bboxes)):
            dataset.append([bboxes[i, 2], bboxes[i, 3]])
    return np.array(dataset)


if __name__ == '__main__':
    # 运行该程序会计算'./VOCdevkit/VOC2007/Annotations'的xml
    # 会生成yolo_anchors.txt
    SIZE = 416

    anchors_num = 9
    # 载入数据集，可以使用VOC的xml
    # path = r'./VOCdevkit/VOC2007/Annotations'

    # 载入所有的xml
    # 存储格式为转化为比例后的width,height
    data = load_data()

    # 使用k聚类算法
    out = kmeans(data, anchors_num)
    out = out[np.argsort(out[:, 0])]
    print('acc:{:.2f}%'.format(avg_iou(data, out) * 100))
    # print(out * SIZE)
    data = out
    data = list(data)
    data.sort(key=lambda x:x[0]*x[1])
    print(data)