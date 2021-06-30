import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import shutil
import numpy as np
import tensorflow as tf
from yolov3.dataset import Dataset
from yolov3.yolov4 import Create_Yolo, compute_loss
from yolov3.utils import load_yolo_weights
from yolov3.configs import *


def main():
    global TRAIN_FROM_CHECKPOINT

    if os.path.exists(TRAIN_LOGDIR): shutil.rmtree(TRAIN_LOGDIR)
    writer = tf.summary.create_file_writer(TRAIN_LOGDIR)
    # TRAIN_LOGDIR = "log"

    trainset = Dataset('train')
    # testset = Dataset('test')

    steps_per_epoch = len(trainset)
    global_steps = tf.Variable(0, trainable=False, dtype=tf.int64)
    warmup_steps = TRAIN_WARMUP_EPOCHS * steps_per_epoch
    # 余弦退火学习率

    total_steps = TRAIN_EPOCHS * steps_per_epoch
    if TRAIN_TRANSFER:
        Darknet = Create_Yolo(input_size=YOLO_INPUT_SIZE)
        load_yolo_weights(Darknet, Darknet_weights) # use darknet weights

    yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, training=True, CLASSES=TRAIN_CLASSES)
    # [batch_size,52,52,3*(80+5+180)],[batch_size,52,52,3,80+5+180]
    # [batch_size,26.26,3*(80+5+180)],[batch_size,26,26,3,80+5+180]
    # [batch_size,13,13,3*(80+5+180)],[batch_size,13,13,3,80+5+180]
    # yolo.summary()
    # [x,y,w,h,conf,CSL_Label,classes_prob]
    if TRAIN_FROM_CHECKPOINT:
        try:
            yolo.load_weights(TRAIN_CHECKPOINTS_FOLDER)
            print(TRAIN_CHECKPOINTS_FOLDER)
        except ValueError:
            print("Shapes are incompatible, transfering Darknet weights")
            TRAIN_FROM_CHECKPOINT = False

    '''
    if TRAIN_TRANSFER and not TRAIN_FROM_CHECKPOINT:
        for i, l in enumerate(Darknet.layers):
            layer_weights = l.get_weights()
            if layer_weights != []:
                try:
                    yolo.layers[i].set_weights(layer_weights)
                except:
                    print("skipping", yolo.layers[i].name)
    '''
    
    optimizer = tf.keras.optimizers.Adam()

    def train_step(image_data, target):
        with tf.GradientTape() as tape:
            pred_result = yolo(image_data, training=True)
            giou_loss=conf_loss=prob_loss=angel_loss=0

            # optimizing process
            grid = 3 if not TRAIN_YOLO_TINY else 2

            for i in range(grid):
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                loss_items = compute_loss(pred, conv, *target[i], i, CLASSES=TRAIN_CLASSES)
                # 预测的输出与真实的标签计算损失
                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                angel_loss += loss_items[2]
                prob_loss += loss_items[3]

            total_loss = giou_loss + conf_loss + prob_loss + angel_loss
            gradients = tape.gradient(total_loss, yolo.trainable_variables)
            optimizer.apply_gradients(zip(gradients, yolo.trainable_variables))

            # update learning rate
            # about warmup: https://arxiv.org/pdf/1812.01187.pdf&usg=ALkJrhglKOPDjNt6SHGbphTHyMcT0cuMJg
            global_steps.assign_add(1)

            if global_steps < warmup_steps:
                # and not TRAIN_TRANSFER:
                lr = global_steps / warmup_steps * TRAIN_LR_INIT
            else:
                lr = TRAIN_LR_END + 0.5 * (TRAIN_LR_INIT - TRAIN_LR_END)*(
                    (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi)))

            # lr = tf.Variable(1e-6)
            optimizer.lr.assign(lr.numpy())

            # writing summary data
            with writer.as_default():
                tf.summary.scalar("lr", optimizer.lr, step=global_steps)
                tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
                tf.summary.scalar("loss/ciou_loss", giou_loss, step=global_steps)
                tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
                tf.summary.scalar("loss/angel_loss", angel_loss, step=global_steps)
                tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
            writer.flush()
            
        return global_steps.numpy(), optimizer.lr.numpy(), giou_loss.numpy(), conf_loss.numpy(),angel_loss.numpy() ,prob_loss.numpy(), total_loss.numpy()

    validate_writer = tf.summary.create_file_writer(TRAIN_LOGDIR)

    def validate_step(image_data, target):
        with tf.GradientTape() as tape:
            # image_data====>[batch_size,416,416,3]
            pred_result = yolo(image_data, training=False)
            giou_loss=conf_loss=prob_loss=angel_loss=0

            grid = 3 if not TRAIN_YOLO_TINY else 2
            for i in range(grid):
                conv, pred = pred_result[i*2], pred_result[i*2+1]
                loss_items = compute_loss(pred, conv, *target[i], i, CLASSES=TRAIN_CLASSES)

                giou_loss += loss_items[0]
                conf_loss += loss_items[1]
                angel_loss += loss_items[2]
                prob_loss += loss_items[3]

            total_loss = giou_loss + conf_loss + prob_loss + angel_loss

        return giou_loss.numpy(), conf_loss.numpy(), angel_loss.numpy(), prob_loss.numpy(), total_loss.numpy()

    # mAP_model = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES) # create second model to measure mAP
    # mAP-model

    best_val_loss = 1000 # should be large at start
    for epoch in range(TRAIN_EPOCHS):
        # TRAIN_EPOCHS = 100
        epoch_total_loss=0
        for image_data, target in trainset:
            # t1=time.time()
            results = train_step(image_data, target)
            # t2=time.time()
            # print(t2-t1)
            '''
            TODO:add angel regression  loss
            '''
            cur_step = results[0]%steps_per_epoch
            print("epoch:{:2.0f} step:{:5.0f}/{}, lr:{:.6f}, ciou_loss:{:7.2f}, conf_loss:{:7.2f},angel_loss:{:7.2f}, prob_loss:{:7.2f}, total_loss:{:7.2f}"
                  .format(epoch, cur_step, steps_per_epoch, results[1], results[2], results[3], results[4], results[5],results[6]))
            epoch_total_loss+=results[6]
        print("\n\nepoch_avg_total_loss:{}\n\n".format(epoch_total_loss/(14000//32)))
        '''
        if len(testset) == 0:
            print("configure TEST options to validate model")
            yolo.save_weights(os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME))
            # TRAIN_CHECKPOINTS_FOLDER = "checkpoints"
            # TRAIN_MODEL_NAME = "yolov3_custom"
            continue
            
        count, giou_val, conf_val, angel_val, prob_val, total_val = 0., 0, 0, 0, 0,0
        for image_data, target in testset:
            results = validate_step(image_data, target)
            count += 1
            giou_val += results[0]
            conf_val += results[1]
            angel_val += results[2]
            prob_val += results[3]
            total_val += results[4]
            
        # writing validate summary data
        with validate_writer.as_default():
            tf.summary.scalar("validate_loss/total_val", total_val/count, step=epoch)
            tf.summary.scalar("validate_loss/giou_val", giou_val/count, step=epoch)
            tf.summary.scalar("validate_loss/angel_val", angel_val / count, step=epoch)
            tf.summary.scalar("validate_loss/conf_val", conf_val/count, step=epoch)
            tf.summary.scalar("validate_loss/prob_val", prob_val/count, step=epoch)
        validate_writer.flush()
       
            
        print("\n\ngiou_val_loss:{:7.2f}, conf_val_loss:{:7.2f}, angel_val_loss:{:7.2f},prob_val_loss:{:7.2f}, total_val_loss:{:7.2f}\n\n".
              format(giou_val/count, conf_val/count, angel_val/count, prob_val/count, total_val/count))
        
        '''

        if TRAIN_SAVE_CHECKPOINT and not TRAIN_SAVE_BEST_ONLY:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME+"epoch_{}".format(epoch))
            yolo.save_weights(save_directory)
        if TRAIN_SAVE_BEST_ONLY and best_val_loss>epoch_total_loss/224:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME)
            yolo.save_weights(save_directory)
            best_val_loss = epoch_total_loss/224
        if not TRAIN_SAVE_BEST_ONLY and not TRAIN_SAVE_CHECKPOINT:
            save_directory = os.path.join(TRAIN_CHECKPOINTS_FOLDER, TRAIN_MODEL_NAME)
            yolo.save_weights(save_directory)

    # measure mAP of trained custom model
    # mAP_model.load_weights(save_directory) # use keras weights

    # get_mAP(mAP_model, testset, score_threshold=TEST_SCORE_THRESHOLD, iou_threshold=TEST_IOU_THRESHOLD)


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        print(gpus)
        try:
            tf.config.experimental.set_memory_growth(gpus[0], True)
        except RuntimeError:
            pass

    if YOLO_TYPE == "yolov4":
        Darknet_weights = YOLO_V4_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V4_WEIGHTS

    if YOLO_TYPE == "yolov3":
        Darknet_weights = YOLO_V3_TINY_WEIGHTS if TRAIN_YOLO_TINY else YOLO_V3_WEIGHTS

    if TRAIN_YOLO_TINY:
        TRAIN_MODEL_NAME += "_Tiny"

    main()



