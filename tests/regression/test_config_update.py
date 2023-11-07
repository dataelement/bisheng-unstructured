import os

import requests


def test_config_update():
    uns_ep = os.environ.get("UNS_EP", "127.0.0.1:10001")
    uns_ep = f"http://{uns_ep}"
    ep = uns_ep + "/v1/config/update"
    rt_ep = "192.168.106.x:y"
    inp = {"rt_ep": rt_ep}
    resp = requests.post(ep, json=inp).json()
    assert resp["status"] == "OK"

    ep2 = uns_ep + "/v1/config"
    resp2 = requests.get(ep2).json()
    assert rt_ep in resp2["config"]["pdf_model_params"]["layout_ep"]
    assert rt_ep in resp2["config"]["pdf_model_params"]["ocr_model_ep"]
    print("test config update: passed")


test_config_update()
