import json
import numpy as np
import tritonclient.http as httpclient


# Layout Agent Version 0.1, update at 2023.08.18
class LayoutAgent(object):

    def __init__(self, *args, **kwargs):
        self.ep = kwargs.get("layout_ep")
        ep_parts = self.ep.split("/")
        self.model = ep_parts[-2]
        server_url = ep_parts[2]
        self.client = httpclient.InferenceServerClient(url=server_url, verbose=False)
        self.timeout = kwargs.get("timeout", 60)
        self.params = {
            "longer_edge_size": 0,
        }

    def predict(self, inp):
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
