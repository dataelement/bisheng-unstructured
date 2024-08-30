import base64
import copy
import json

import numpy as np
import requests
import tritonclient.http as httpclient


# Layout Agent Version 0.1, update at 2023.08.18
class LayoutAgentV0(object):
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


class LayoutAgent:
    def __init__(self, *args, **kwargs):
        ep_parts = kwargs.get("layout_ep").split("/")
        self.model = ep_parts[-2]
        server_url = ep_parts[2]
        self.client = httpclient.InferenceServerClient(url=server_url, verbose=False)

    def predict(self, inp):
        # b64_image = base64.b64encode(open(image_file, 'rb').read()).decode('utf-8')
        input0_data = np.asarray([json.dumps(inp)], dtype=np.object_)
        inputs = [httpclient.InferInput("INPUT", [1], "BYTES")]
        inputs[0].set_data_from_numpy(input0_data)
        outputs = [httpclient.InferRequestedOutput("OUTPUT")]
        try:
            response = self.client.infer(self.model, inputs, request_id=str(1), outputs=outputs)
            output_data = json.loads(response.as_numpy("OUTPUT")[0].decode("utf-8"))
        except Exception as e:
            raise Exception(f"exception in layout predict: [{e}]")
        return output_data
