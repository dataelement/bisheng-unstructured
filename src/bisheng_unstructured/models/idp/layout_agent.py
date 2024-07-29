import base64
import copy

import requests


# Layout Agent Version 0.1, update at 2023.08.18
class LayoutAgent(object):
    def __init__(self, *args, **kwargs):
        self.ep = kwargs.get("layout_ep")
        self.client = requests.Session()
        self.timeout = kwargs.get("timeout", 60)
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
        except requests.exceptions.Timeout:
            raise Exception(f"timeout in layout predict")
        except Exception as e:
            raise Exception(f"exception in layout predict: [{e}]")
