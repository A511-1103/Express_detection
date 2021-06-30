# Express_detection
YOLOV4,YOLOV5,rbox, tf2,旋转框检测
![image](./detection_imags/building_detection.png)
# 已经完成的部分
# 1.train.txt 数据所对应的格式为 images_path x,y,long_side,short_side,classes_idx,theta
# 2.角度作为回归任务,采用CSL,论文链接[https://yangxue0827.github.io/files/csl_slides.pdf]
# 3.将IoU匹配准则换为RIoU匹配准则
# 4.网格正负样本极不均衡的缺陷,采用YOLOV5回归机制中采用了YOLOV5中上下左右均作为正样本的思想,对应的中心点坐标解码方式发生改变
# 5.网络可供选择的模型有：YOLOV3、YOLOV4、YOLOV4_Large、YOLOV4_MOBILNET
# 6.IoU、CIoU、DIoU、GIoU,虽然可以检测,但损失函数应用于旋转框检测存在缺陷
# 需要继续改善的部分
# 1.IoU匹配准则中改为了RIoU的准则,但在损失函数中由于RIoU损失可能不可导,但在这篇论文[https://arxiv.org/abs/2011.09670]中提出了相应的改进机制,接下来会尝试利用该GWD损失完善旋转框的损失函数
# 2.置信度的回归与IoU有关,而不是RIoU,这将导致置信度得分出现虚高的情况
# 3.mosaic数据增强应用于旋转框时暂未想到合适的越界处理办法