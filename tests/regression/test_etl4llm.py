import base64
import os

import requests


def test_part():
    uns_ep = os.environ.get("UNS_EP", "127.0.0.1:10001")

    url = f"http://{uns_ep}/v1/etl4llm/predict"
    filename = "../../examples/docs/达梦数据库招股说明书.pdf"
    b64_data = base64.b64encode(open(filename, "rb").read()).decode()
    inp = dict(
        filename=os.path.basename(filename),
        b64_data=[b64_data],
        mode="partition",
        parameters={"start": 0, "n": 5},
    )
    resp = requests.post(url, json=inp).json()
    print(resp)


test_part()
