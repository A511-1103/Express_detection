import requests
import base64
import socket
import json
import os


def get_host_ip():
    """
    查询本机ip地址
    :return: ip
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()

    return ip

user_ip = get_host_ip()
user_ip = str(user_ip).split('.')
user_ip = '_'.join(user_ip)     # 可自行确定该客户端的标识号，但需要保证唯一性

# 检测结果中的图片是否需要保存再本地
is_Save = True      #  如果不需要保存，设置为 is_Save = False
# 检测结果中图片保存位置
save_path = r'D:\dataset\my_test\result'
if is_Save:
    os.makedirs(save_path)

# API地址
url = "http://192.168.2.178:5002/photo"     # 快递单检测URL
# 图片地址
file_path = r'G:\data\111\SF1031550393590.jpg'    # 20092540602038.jpg    SF1031527629397_YT
if not os.path.exists(file_path):
    print('输入图片路径不存在')


# 图片名
file_name = os.path.basename(file_path)   # 图片的名字，如1.png

# 二进制打开图片
file = open(file_path, 'rb')
a = 1
# 拼接参数
files = {'file': (file_name, file, 'image/jpg')}

# 发送post请求到服务器端
r = requests.post(url, files=files,headers={'clientID':'{}'.format(user_ip)})

# 获取服务器返回的图片，字节流返回
result = r.content.decode('utf-8')
my_json = json.loads(result)
print(json.dumps(my_json, sort_keys=True, indent=4,  separators=(',',':'), ensure_ascii=False))
data = my_json['data']
# print(data)
status = my_json['status']

# # 字节转换成图片
if is_Save and status == 'SUCCESS':

    keys = my_json['data'].keys()

    # print(my_json['data'].items())
    # print('img', img)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in keys:
        img = my_json['data'][i]
        img = base64.b64decode(img)
        file = open(save_path + r'\\test_{}.jpg'.format(i), 'wb')
        file.write(img)
        file.close()