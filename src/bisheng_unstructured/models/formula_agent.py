import base64
import copy

import requests


class FormulaDetectAgent(object):
    def __init__(self, **kwargs):
        self.ep = kwargs.get("formula_det_model_ep")
        self.client = requests.Session()
        self.timeout = kwargs.get("timeout", 60)

    def predict(self, inp):
        try:
            r = self.client.post(url=self.ep, json=inp, timeout=self.timeout)
            return r.json()
        except requests.exceptions.Timeout:
            raise Exception(f"timeout in formula det predict")
        except Exception as e:
            raise Exception(f"exception in formula det predict: [{e}]")


class FormulaRecogAgent(object):
    def __init__(self, **kwargs):
        self.ep = kwargs.get("formula_recog_model_ep")
        self.client = requests.Session()
        self.timeout = kwargs.get("timeout", 60)
        self.params = {}

    def predict(self, inp):
        try:
            r = self.client.post(url=self.ep, json=inp, timeout=self.timeout)
            return r.json()
        except requests.exceptions.Timeout:
            raise Exception(f"timeout in formula recog predict")
        except Exception as e:
            raise Exception(f"exception in formula recog predict: [{e}]")
