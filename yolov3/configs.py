YOLO_TYPE                   = "yolov4-mobilenet" # yolov4 or yolov3 or yolov4_large or yolov4-mobilenet
# 可以默认用"yolov4-mobilenet"
if YOLO_TYPE                == "yolov4-mobilenet":
    # YOLO_ANCHORS            = [[[12,  16], [19,   36], [40,   28]],
    #                            [[36,  75], [76,   55], [72,  146]],
    #                            [[142,110], [192, 243], [459, 401]]]
    # YOLO_ANCHORS = [[[124,160], [226,158], [160,228]],
    #                 [[210,210], [329,215], [215,330]],
    #                 [[616,884], [887,617], [818,830]]]
    # YOLO_ANCHORS_LS = [[[160,124], [226,158], [228,160]],
    #                    [[210,210], [329,215], [330,215]],
    #                    [[884,616], [887,617], [830,818]]]
    # YOLO_ANCHORS = [[[42,51], [54,75], [74,55]],
    #                 [[71,106], [110,72], [73,126]],
    #                 [[297,207], [207,298], [278,275]]]
    #
    # YOLO_ANCHORS_LS = [[[51,42], [75,54], [74,55]],
    #                    [[106,71], [110,72], [126,73]],
    #                    [[297,207], [298,107], [278,275]]]

    YOLO_ANCHORS = [[[53, 73], [74, 54], [71, 106]],
                    [[110, 72], [73, 126], [268, 182]],
                    [[207, 298], [302, 212], [276, 276]]]
    YOLO_ANCHORS_LS = [[[73, 53], [74, 54], [106, 71]],
                       [[110, 72], [126, 73], [268, 182]],
                       [[298, 207], [302, 212], [276, 276]]]


YOLO_CUSTOM_WEIGHTS         = "checkpoints/yolov4-mobilenet_customepoch_86"
# 检测时需要用到的

TRAIN_CLASSES               = "model_data/kuaidi.txt"
TRAIN_ANNOT_PATH            = "model_data/train.txt"

TRAIN_EPOCHS                = 200
TRAIN_LOAD_IMAGES_TO_RAM    = True

YOLO_MAX_BBOX_PER_SCALE     = 10
YOLO_INPUT_SIZE             = 416
# 416*1.5
TRAIN_INPUT_SIZE            = 416

TRAIN_BATCH_SIZE            = 16
TRAIN_SAVE_CHECKPOINT       = True

# 先默认False

TRAIN_FROM_CHECKPOINT       = False
TRAIN_CHECKPOINTS_FOLDER    = 'checkpoints/yolov4-mobilenet_customepoch_0'


# 如果YOLO_INPUT_SIZE越小,TRAIN_BATCH_SIZE 越大 --->[416，32]

# YOLO_INPUT_SIZE   越大，降采样（4000/628）*（3000/618）==30多倍
# 越小,训练和检测速度更快
# 精度与速度上的权衡

# 不用修改的参数如下:
YOLO_FRAMEWORK              = "tf"
YOLO_V3_WEIGHTS             = "model_data/yolov3.weights"
YOLO_V4_WEIGHTS             = "model_data/yolov4.weights"
YOLO_V3_TINY_WEIGHTS        = "model_data/yolov3-tiny.weights"
YOLO_V4_TINY_WEIGHTS        = "model_data/yolov4-tiny.weights"
YOLO_TRT_QUANTIZE_MODE      = "INT8" # INT8, FP16, FP32
# YOLO_CUSTOM_WEIGHTS         ="checkpoints/yolov4_customepoch_29"
YOLO_COCO_CLASSES           = "model_data/coco/coco.names"
YOLO_STRIDES                = [8, 16, 32]
YOLO_IOU_LOSS_THRESH        = 0.5
YOLO_ANCHOR_PER_SCALE       = 3
TRAIN_YOLO_TINY             = False
TRAIN_SAVE_BEST_ONLY        = False





if YOLO_TYPE                == "yolov4":
    # YOLO_ANCHORS            = [[[12,  16], [19,   36], [40,   28]],
    #                            [[36,  75], [76,   55], [72,  146]],
    #                            [[142,110], [192, 243], [459, 401]]]
    YOLO_ANCHORS = [[[124,160], [226,158], [160,228]],
                    [[210,210], [329,215], [215,330]],
                    [[616,884], [887,617], [818,830]]]
    YOLO_ANCHORS_LS = [[[160,124], [226,158], [228,160]],
                       [[210,210], [329,215], [330,215]],
                       [[884,616], [887,617], [830,818]]]

if YOLO_TYPE                =='yolov4_large':
    YOLO_ANCHORS = [[[42, 51], [54, 75], [74, 55]],
                    [[71, 106], [110, 72], [73, 126]],
                    [[297, 207], [207, 298], [278, 275]]]

    YOLO_ANCHORS_LS = [[[51, 42], [75, 54], [74, 55]],
                       [[106, 71], [110, 72], [126, 73]],
                       [[297, 207], [298, 107], [278, 275]]]
    #
    # YOLO_ANCHORS_LS         = [[[16, 10],  [19, 18], [33, 15]],
    #                             [[40, 23], [46, 34], [67, 44]],
    #                             [[64, 51], [109, 61], [127, 74]]]

if YOLO_TYPE == "yolov3":
    YOLO_ANCHORS            = [[[10,  13], [16,   30], [33,   23]],
                               [[30,  61], [62,   45], [59,  119]],
                               [[116, 90], [156, 198], [373, 326]]]
    YOLO_ANCHORS_LS         = [[[ 13,  10], [30,  16], [33,   23]],
                                [[61,  30], [62,  45], [119,  59]],
                                [[116, 90], [198, 156], [373, 326]]]

# Train options


TRAIN_LOGDIR                = "log"
TRAIN_MODEL_NAME            = f"{YOLO_TYPE}_custom"
TRAIN_DATA_AUG              = False
TRAIN_TRANSFER              = False
TRAIN_LR_INIT               = 1e-4
TRAIN_LR_END                = 1e-6
TRAIN_WARMUP_EPOCHS         = 4



# TEST options
TEST_ANNOT_PATH             = "model_data/test.txt"
TEST_BATCH_SIZE             = 4
TEST_INPUT_SIZE             = 416
TEST_DATA_AUG               = False
TEST_DECTECTED_IMAGE_PATH   = ""
TEST_SCORE_THRESHOLD        = 0.3
TEST_IOU_THRESHOLD          = 0.45

if TRAIN_YOLO_TINY:
    YOLO_STRIDES            = [16, 32, 64]    
    YOLO_ANCHORS            = [[[10,  14], [23,   27], [37,   58]],
                               [[81,  82], [135, 169], [344, 319]],
                               [[0,    0], [0,     0], [0,     0]]]
