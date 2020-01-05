#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
使用自建的接口识别来自本地目录的验证码
需要配置参数：
    test_path = "image/test/0401_15440848576253345.png"
"""
import datetime
import requests
from io import BytesIO
import time
import json
import os


def recognize_captcha(test_path, save_path, image_suffix):
    image_file_name = 'captcha.{}'.format(image_suffix)

    with open(test_path, "rb") as f:
        content = f.read()

    # 识别
    s = time.time()
    url = "http://127.0.0.1:6000/recognize"
    files = {'image_file': (image_file_name, BytesIO(content), 'application')}
    r = requests.post(url=url, files=files)
    e = time.time()

    # 识别结果
    print("接口响应: {}".format(r.text))
    predict_text = json.loads(r.text)["value"]
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("【{}】 耗时：{}ms 预测结果：{}".format(now_time, int((e-s)*1000), predict_text))

    # 保存文件
    img_name = "{}_{}.{}".format(predict_text, str(time.time()).replace(".", ""), image_suffix)
    path = os.path.join(save_path, img_name)
    with open(path, "wb") as f:
        f.write(content)
    print("============== end ==============")


def main():
    with open("conf/app_config.json", "r") as f:
        app_conf = json.load(f)

    # 配置相关参数
    test_path = "image/test/0401_15440848576253345.png"  # 测试识别的图片路径
    save_path = app_conf["local_image_dir"]  # 保存的地址
    image_suffix = app_conf["image_suffix"]  # 文件后缀
    recognize_captcha(test_path, save_path, image_suffix)


if __name__ == '__main__':
    main()
    

