import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import base64
from flask import request
from flask import Flask, jsonify, json
import time, math, glob
from yolov3.yolov4 import Create_Yolo
from yolov3.configs import *
from detection import detect_image
import shutil



'''
服务端 Server
对外提供快递单检测的API接口
'''
app = Flask(__name__)

def del_file(path):
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)


def load_model():
    yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=TRAIN_CLASSES)
        # print(YOLO_INPUT_SIZE,TRAIN_CLASSES)
    yolo.load_weights(YOLO_CUSTOM_WEIGHTS)  # use custom weights
    return yolo

def run_model(yolo, img_path, save_path):
    t = detect_image(yolo,
                img_path,
                input_size=416,
                score_threshold=0.7,
                iou_threshold=0.1,
                save_path = save_path)
    return t

yolo = load_model()

# 定义路由
@app.route("/photo", methods=['POST'])
def get_frame():
    try:
        user_IP = request.headers.get('clientID')
        # 接收图片
        upload_file = request.files['file']

        # 获取图片名
        file_name = upload_file.filename
        if upload_file == None:
            return jsonify(status='NG', data=None, error='传入的图片错误', time=None)
        if file_name == None:
            return jsonify(status='NG', data=None, error='传入的图片名字为空', time=None)
        # 文件保存目录（桌面）
        file_path = r'D:\XYLS\receive_file'
        if upload_file:
            # 地址拼接
            file_paths = os.path.join(file_path, file_name)
            # 保存接收的图片到桌面
            upload_file.save(file_paths)
            # user_IP
            if os.path.exists(user_IP): shutil.rmtree(user_IP)
            t = run_model(yolo,file_paths,user_IP)

            all_path = glob.glob(r'D:\XYLS' + '\\' + user_IP  + '\\' + file_name[:-4] + '\\*.png')
            if len(all_path)==0:
                return jsonify(data=None, status='SUCCESS', error='None', time=t)
            res = {}
            for i in range(len(all_path)):
                with open(all_path[i], 'rb') as f:
                    res['{}'.format(i)] = base64.b64encode(f.read())


            return jsonify(data=res, status='SUCCESS', error='None', time=t)
            # return_json = {'data':img_data, 'status':'SUCCESS','error':'None'}
            # return json.dumps(img_data, ensure_ascii=False, encoding='utf-8')
    except Exception as e:
        return jsonify(status='NG', data=None, error=e, time=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)