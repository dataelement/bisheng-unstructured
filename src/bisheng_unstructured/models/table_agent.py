import base64
import copy

import requests


# Table Agent Version 0.1, update at 2023.08.18
class TableAgent(object):
    def __init__(self, **kwargs):
        cell_model_ep = kwargs.get("cell_model_ep")
        rowcol_model_ep = kwargs.get("rowcol_model_ep")

        self.ep_map = {
            "cell": cell_model_ep,
            "rowcol": rowcol_model_ep,
        }
        self.params = {
            "sep_char": " ",
            "longer_edge_size": None,
            "padding": False,
        }

        self.client = requests.Session()
        self.timeout = kwargs.get("timeout", 60)

    def predict(self, inp):
        scene = inp.pop("scene", "rowcol")
        ep = self.ep_map.get(scene)
        params = copy.deepcopy(self.params)
        params.update(inp)

        try:
            r = self.client.post(url=ep, json=params, timeout=self.timeout)
            return r.json()
        except requests.exceptions.Timeout:
            raise Exception(f"timeout in table structure predict")
        except Exception as e:
            raise Exception(f"exception in table structure predict: [{e}]")


# TableDet Agent Version 0.1, update at 2023.08.31
class TableDetAgent(object):
    def __init__(self, **kwargs):
        self.ep = kwargs.get("table_model_ep")
        self.client = requests.Session()
        self.timeout = kwargs.get("timeout", 60)
        self.params = {}

    def predict(self, inp):
        params = copy.deepcopy(self.params)
        params.update(inp)

        try:
            r = self.client.post(url=self.ep, json=params, timeout=self.timeout)
            return r.json()
        except requests.exceptions.Timeout:
            raise Exception(f"timeout in table det predict")
        except Exception as e:
            raise Exception(f"exception in table det predict: [{e}]")
