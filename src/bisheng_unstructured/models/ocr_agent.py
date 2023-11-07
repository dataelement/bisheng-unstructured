import base64
import copy

import requests


# OCR Agent Version 0.1, update at 2023.08.18
class OCRAgent(object):
    def __init__(self, **kwargs):
        self.ep = kwargs.get("ocr_model_ep")
        self.client = requests.Session()
        self.timeout = kwargs.get("timeout", 10000)
        self.params = {
            "sort_filter_boxes": True,
            "enable_huarong_box_adjust": True,
            "rotateupright": False,
            "support_long_image_segment": True,
        }

        self.scene_mapping = {
            "print": {
                "det": "general_text_det_mrcnn_v2.0",
                "recog": "transformer-blank-v0.2-faster",
            },
            "hand": {
                "det": "general_text_det_mrcnn_v2.0",
                "recog": "transformer-hand-v1.16-faster",
            },
        }

    def predict(self, inp):
        scene = inp.pop("scene", "print")
        b64_image = inp.pop("b64_image")
        params = copy.deepcopy(self.params)
        params.update(self.scene_mapping[scene])
        params.update(inp)

        req_data = {"param": params, "data": [b64_image]}

        try:
            r = self.client.post(url=self.ep, json=req_data, timeout=self.timeout)
            return r.json()
        except Exception as e:
            return {"status_code": 400, "status_message": str(e)}
