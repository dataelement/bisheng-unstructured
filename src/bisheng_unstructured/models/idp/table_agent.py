import copy
import json

import numpy as np
import tritonclient.http as httpclient
from loguru import logger


# Table Agent Version 0.1, update at 2023.08.31
class TableAgent(object):
    def __init__(self, **kwargs):
        ep_parts = kwargs.get("cell_model_ep").split("/")
        self.cell_server_url = ep_parts[2]
        self.cell_model = ep_parts[-2]

        ep_parts = kwargs.get("rowcol_model_ep").split("/")
        self.rowcol_server_url = ep_parts[2]
        self.rowcol_model = ep_parts[-2]

        self.timeout = kwargs.get("timeout", 60)
        self.params = {
            "sep_char": " ",
            "longer_edge_size": None,
            "padding": False,
        }

    def predict(self, inp):
        scene = inp.pop("scene", "rowcol")
        if scene == "rowcol":
            client, model = (
                httpclient.InferenceServerClient(url=self.rowcol_server_url, verbose=False),
                self.rowcol_model,
            )
        else:
            client, model = (
                httpclient.InferenceServerClient(url=self.cell_server_url, verbose=False),
                self.cell_model,
            )

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
            logger.info("table predict request, model: {}", model)
            response = client.infer(model, inputs, request_id=str(1), outputs=outputs)
            print("response", response)
            output_data = json.loads(response.as_numpy("OUTPUT")[0].decode("utf-8"))
        except Exception as e:
            raise Exception(f"exception in table structure predict: [{e}]")

        return output_data


class TableDetAgent(object):
    def __init__(self, **kwargs):
        ep_parts = kwargs.get("table_model_ep").split("/")
        self.server_url = ep_parts[2]
        self.model = ep_parts[-2]
        self.timeout = kwargs.get("timeout", 60)

    def predict(self, inp):
        # b64data = base64.b64encode(open(image_file, 'rb').read()).decode('utf-8')
        input0_data = np.asarray([json.dumps(inp)], dtype=np.object_)

        inputs = [httpclient.InferInput("INPUT", [1], "BYTES")]
        inputs[0].set_data_from_numpy(input0_data)
        outputs = [httpclient.InferRequestedOutput("OUTPUT")]
        client = httpclient.InferenceServerClient(url=self.server_url, verbose=False)
        try:
            response = client.infer(self.model, inputs, request_id=str(1), outputs=outputs)
            output_data = json.loads(response.as_numpy("OUTPUT")[0].decode("utf-8"))
        except Exception as e:
            raise Exception(f"exception in table det predict: [{e}]")

        return output_data
