import base64
import os

import requests


def test1():
    url = "http://192.168.106.12:10001/v1/etl4llm/predict"
    filename = "examples/docs/maoxuan_sample1.jpg"
    b64_data = base64.b64encode(open(filename, "rb").read()).decode()
    inp = dict(filename=os.path.basename(filename), b64_data=[b64_data], mode="text")
    resp = requests.post(url, json=inp).json()
    print(resp)


def test2():
    url = "http://192.168.106.12:10001/v1/etl4llm/predict"
    filename = "./examples/docs/毛泽东课件.pptx"
    b64_data = base64.b64encode(open(filename, "rb").read()).decode()
    inp = dict(filename=os.path.basename(filename), b64_data=[b64_data], mode="text")
    resp = requests.post(url, json=inp).json()
    print(resp)


def test3():
    url = "http://192.168.106.12:10001/v1/etl4llm/predict"
    filename = "./examples/docs/毛泽东课件.pptx"
    b64_data = base64.b64encode(open(filename, "rb").read()).decode()
    inp = dict(filename=os.path.basename(filename), b64_data=[b64_data], mode="partition")
    resp = requests.post(url, json=inp).json()
    print(resp)


def test4():
    url = "http://192.168.106.12:10001/v1/etl4llm/predict"
    filename = "./examples/docs/毛泽东课件.pptx"
    b64_data = base64.b64encode(open(filename, "rb").read()).decode()
    inp = dict(filename=os.path.basename(filename), b64_data=[b64_data], mode="vis")
    resp = requests.post(url, json=inp).json()
    print(resp)


def test5():
    url = "http://192.168.106.12:10001/v1/etl4llm/predict"
    filename = "./examples/docs/达梦数据库招股说明书.pdf"
    b64_data = base64.b64encode(open(filename, "rb").read()).decode()
    inp = dict(
        filename=os.path.basename(filename),
        b64_data=[b64_data],
        mode="partition",
        parameters={"start": 0, "n": 5},
    )
    resp = requests.post(url, json=inp).json()
    print(resp)


def test6():
    url = "http://192.168.106.12:10002/v1/etl4llm/predict"
    filename = "./examples/docs/maoxuan_sample1.jpg"
    output = "./temp/maoxuan_sample1_v1.pdf"
    b64_data = base64.b64encode(open(filename, "rb").read()).decode()
    inp = dict(filename=os.path.basename(filename), b64_data=[b64_data], mode="topdf")
    resp = requests.post(url, json=inp).json()
    b64_data = resp["b64_pdf"]
    with open(output, "wb") as fout:
        fout.write(base64.b64decode(b64_data))


# test1()
# test2()
# test3()
# test4()
# test5()
# test6()
