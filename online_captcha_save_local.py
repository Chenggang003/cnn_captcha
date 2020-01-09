"""
通过输入线上的获取第三方验证码图片地址，将其保存到本地image/origin
"""
import json
import requests
import os
import time
import base64

def main():
    with open("conf/app_config.json", "r") as f:
        app_conf = json.load(f)

    # 图片路径
    origin_dir = app_conf["origin_image_dir"]

    headers = {
        'user-agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/65.0.3325.146 Safari/537.36",
    }

    for index in range(1):
        # 请求
        while True:
            try:
                response = requests.request("GET", "http://zxgk.court.gov.cn/shixin/captchaNew.do?captchaId"
                                                   "=U3qUb6wgg93Hcs9TDXm6CFPTc8uL9BM0&random=0.6914391412872873",
                                            headers=headers, timeout=6)
                if response.text:
                    break
                else:
                    print("retry, response.text is empty")
            except Exception as ee:
                print(ee)

        # 识别对应的验证码
        headers = {
            'Content-Type': "application/json",
        }
        base64_data = base64.b64encode(response.content).decode()
        params_json = json.dumps({'channelId': '1', 'imageBase64': base64_data})
        res = requests.post("", data=params_json, headers=headers)
        print(res.text)
        # 保存文件
        data = json.loads(res.text)
        img_name = "{}_{}.{}".format(data['result'], str(time.time()).replace(".", ""), "jpg")
        path = os.path.join(origin_dir, img_name)
        with open(path, "wb") as f:
            f.write(response.content)
    print("============== end ==============")


if __name__ == '__main__':
    main()
