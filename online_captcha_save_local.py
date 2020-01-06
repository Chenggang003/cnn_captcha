"""
通过输入线上的获取第三方验证码图片地址，将其保存到本地image/origin
"""
import json
import requests
import os
import time


def main():
    with open("conf/app_config.json", "r") as f:
        app_conf = json.load(f)

    # 图片路径
    origin_dir = app_conf["origin_image_dir"]

    headers = {
        'user-agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/65.0.3325.146 Safari/537.36",
    }

    for index in range(200):
        # 请求
        while True:
            try:
                response = requests.request("GET", "http://www.sh.10086.cn/service/server/servlet/validateCodeServlet"
                                                   "?width=300&height=100&fontSize=60&rnd=1578292412588",
                                            headers=headers, timeout=6)
                if response.text:
                    break
                else:
                    print("retry, response.text is empty")
            except Exception as ee:
                print(ee)

        # 保存文件
        img_name = "_{}.{}".format(str(time.time()).replace(".", ""), "png")
        path = os.path.join(origin_dir, img_name)
        with open(path, "wb") as f:
            f.write(response.content)
        print("============== end ==============")


if __name__ == '__main__':
    main()
