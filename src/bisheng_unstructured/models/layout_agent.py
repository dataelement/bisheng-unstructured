import base64
import copy

import requests


# Layout Agent Version 0.1, update at 2023.08.18
class LayoutAgent(object):
    def __init__(self, *args, **kwargs):
        self.ep = kwargs.get("layout_ep")
        self.client = requests.Session()
        self.timeout = kwargs.get("timeout", 10000)
        self.params = {
            "longer_edge_size": 0,
        }

    def predict(self, inp):
        params = copy.deepcopy(self.params)
        params.update(inp)
        # print('params', params, self.ep)
        try:
            r = self.client.post(url=self.ep, json=params, timeout=self.timeout)
            return r.json()
        except Exception as e:
            return {"status_code": 400, "status_message": str(e)}
