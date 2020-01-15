# -*- coding: UTF-8 -*-
"""
构建flask接口服务
接收 files={'image_file': ('captcha.jpg', BytesIO(bytes), 'application')} 参数识别验证码
需要配置参数：
    image_height = 60
    image_width = 100
    max_captcha = 4
"""
import json
from io import BytesIO
import os
from cnnlib.recognition_object import Recognizer
import base64
import time
from flask import Flask, request, jsonify, Response
from PIL import Image

# 默认使用CPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

with open("conf/app_config.json", "r") as f:
    app_conf = json.load(f)
# 配置参数
image_height = app_conf["image_height"]
image_width = app_conf["image_width"]
max_captcha = app_conf["max_captcha"]
api_image_dir = app_conf["api_image_dir"]
image_suffix = app_conf["image_suffix"]  # 文件后缀
use_labels_json_file = app_conf['use_labels_json_file']

if use_labels_json_file:
    with open("tools/labels.json", "r") as f:
        char_set = f.read().strip()
else:
    char_set = app_conf["char_set"]

# Flask对象
app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))

# 生成识别对象，需要配置参数
R_4 = Recognizer(image_height, image_width, 4, char_set, "model4/model4")
R_5 = Recognizer(image_height, image_width, 5, char_set, "model5/model5")

def response_headers(content):
    resp = Response(content)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp


@app.route("/recognize/base64", methods=['POST'])
def recognize_base64():
    if request.method == 'POST':
        get_data = json.loads(request.get_data(as_text=True))
        timec = str(time.time()).replace(".", "")
        img_b64encode = get_data['image_base64']
        img_b64decode = base64.b64decode(img_b64encode)
        image = BytesIO(img_b64decode)
        img = Image.open(image, mode="r")
        print("接收图片尺寸: {}".format(img.size))
        img = img.resize((150, 50), Image.ANTIALIAS)
        s = time.time()
        captcha_length = get_data['captcha_length']
        if captcha_length == 4:
            value = R_4.rec_image(img)
        elif captcha_length == 5:
            value = R_5.rec_image(img)
        else:
            value = R_4.rec_image(img)
        e = time.time()
        print("识别结果: {}".format(value))
        # 保存图片
        # print("保存图片： {}{}_{}.{}".format(api_image_dir, value, timec, image_suffix))
        # file_name = "{}_{}.{}".format(value, timec, image_suffix)
        # file_path = os.path.join(api_image_dir + file_name)
        # img.save(file_path)
        result = {
            'time': timec,  # 时间戳
            'value': value,  # 预测的结果
            'speed_time(ms)': int((e - s) * 1000)  # 识别耗费的时间
        }
        img.close()
        return jsonify(result)
    else:
        content = json.dumps({"error_code": "1001"})
        resp = response_headers(content)
        return resp


@app.route('/recognize/file', methods=['POST'])
def recognize_file():
    if request.method == 'POST' and request.files.get('image_file'):
        timec = str(time.time()).replace(".", "")
        file = request.files.get('image_file')
        captcha_length = request.values.get("captcha_length",type=int)
        img = file.read()
        img = BytesIO(img)
        img = Image.open(img, mode="r")
        print("接收图片尺寸: {}".format(img.size))
        img = img.resize((150, 50), Image.ANTIALIAS)
        s = time.time()
        if captcha_length == 4:
            value = R_4.rec_image(img)
        elif captcha_length == 5:
            value = R_5.rec_image(img)
        else:
            value = R_4.rec_image(img)
        e = time.time()
        print("识别结果: {}".format(value))
        # 保存图片
        # print("保存图片： {}{}_{}.{}".format(api_image_dir, value, timec, image_suffix))
        # file_name = "{}_{}.{}".format(value, timec, image_suffix)
        # file_path = os.path.join(api_image_dir + file_name)
        # img.save(file_path)
        result = {
            'time': timec,  # 时间戳
            'value': value,  # 预测的结果
            'speed_time(ms)': int((e - s) * 1000)  # 识别耗费的时间
        }
        img.close()
        return jsonify(result)
    else:
        content = json.dumps({"error_code": "1001"})
        resp = response_headers(content)
        return resp


if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=6000,
        debug=True
    )
