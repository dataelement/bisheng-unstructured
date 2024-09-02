import base64
import copy
import json

import numpy as np
import requests
import tritonclient.http as httpclient


# Table Agent Version 0.1, update at 2023.08.18
class TableAgentV0(object):
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
class TableDetAgentV0(object):
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


class TableAgent(object):
    def __init__(self, **kwargs):
        ep_parts = kwargs.get("cell_model_ep").split("/")
        server_url = ep_parts[2]
        self.cell_model = ep_parts[-2]
        self.cell_client = httpclient.InferenceServerClient(url=server_url, verbose=False)

        ep_parts = kwargs.get("rowcol_model_ep").split("/")
        server_url = ep_parts[2]
        self.rowcol_model = ep_parts[-2]
        self.rowcol_client = httpclient.InferenceServerClient(url=server_url, verbose=False)

        self.timeout = kwargs.get("timeout", 60)
        self.params = {
            "sep_char": " ",
            "longer_edge_size": None,
            "padding": False,
        }

    def predict(self, inp):
        scene = inp.pop("scene", "rowcol")
        if scene == "rowcol":
            client, model = self.rowcol_client, self.rowcol_model
        else:
            client, model = self.cell_client, self.cell_model

        payload = copy.deepcopy(self.params)
        payload.update(inp)

        # ocr_result = json.dumps(ocr_result)
        # table_bbox = table_result["bboxes"]
        # b64_image = base64.b64encode(open(image_file, 'rb').read()).decode('utf-8')
        # payload = {'b64_image': b64_image, 'table_bboxes': table_bbox, 'ocr_result': ocr_result}

        input0_data = np.asarray([json.dumps(payload)], dtype=np.object_)
        # print(input0_data)
        inputs = [httpclient.InferInput("INPUT", [1], "BYTES")]
        inputs[0].set_data_from_numpy(input0_data)
        outputs = [httpclient.InferRequestedOutput("OUTPUT")]
        try:
            response = client.infer(model, inputs, request_id=str(1), outputs=outputs)
            print("response", response)
            output_data = json.loads(response.as_numpy("OUTPUT")[0].decode("utf-8"))
        except Exception as e:
            raise Exception(f"exception in table structure predict: [{e}]")

        return output_data


class TableDetAgent(object):
    def __init__(self, **kwargs):
        ep_parts = kwargs.get("table_model_ep").split("/")
        server_url = ep_parts[2]
        self.model = ep_parts[-2]
        self.client = httpclient.InferenceServerClient(url=server_url, verbose=False)
        self.timeout = kwargs.get("timeout", 60)

    def predict(self, inp):
        # b64data = base64.b64encode(open(image_file, 'rb').read()).decode('utf-8')
        input0_data = np.asarray([json.dumps(inp)], dtype=np.object_)

        inputs = [httpclient.InferInput("INPUT", [1], "BYTES")]
        inputs[0].set_data_from_numpy(input0_data)
        outputs = [httpclient.InferRequestedOutput("OUTPUT")]
        try:
            response = self.client.infer(self.model, inputs, request_id=str(1), outputs=outputs)
            output_data = json.loads(response.as_numpy("OUTPUT")[0].decode("utf-8"))
        except Exception as e:
            raise Exception(f"exception in table det predict: [{e}]")

        return output_data
