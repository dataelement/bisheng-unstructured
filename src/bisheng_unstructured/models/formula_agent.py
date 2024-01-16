import base64
import copy
import io

import requests


def save_pillow_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str


class FormulaAgent(object):
    def __init__(self, **kwargs):
        self.det_ep = kwargs.get("formula_det_model_ep")
        self.recog_ep = kwargs.get("formula_recog_model_ep")
        self.client = requests.Session()
        self.timeout = kwargs.get("timeout", 60)

        self.embed_sep = kwargs.get("embed_sep", (" $", "$ "))
        self.isolated_sep = kwargs.get("isolated_sep", ("$$\n", "\n$$"))

    def _get_ep_result(self, ep, inp):
        try:
            r = self.client.post(url=ep, json=inp, timeout=self.timeout)
            return r.json()
        except requests.exceptions.Timeout:
            raise Exception(f"timeout in formula agent predict")
        except Exception as e:
            raise Exception(f"exception in formula agent predict: [{e}]")

    def predict(self, inp, image):
        print("inp", inp)
        formula_det_out = self._get_ep_result(self.det_ep, inp)
        print("formula_det_out", formula_det_out)
        mf_out = []
        for box_info in formula_det_out:
            box = box_info["box"]
            xmin, ymin, xmax, ymax = (
                int(box[0][0]),
                int(box[0][1]),
                int(box[2][0]),
                int(box[2][1]),
            )
            crop_patch = image.crop((xmin, ymin, xmax, ymax))
            patch_b64_image = save_pillow_to_base64(crop_patch)
            inp = {"b64_image": patch_b64_image}

            patch_out = self._get_ep_result(self.recog_ep, inp)
            sep = self.embed_sep
            if box_info["type"] == "isolated":
                sep = self.isolated_sep

            text = sep[0] + patch_out + sep[1]
            mf_out.append({"type": box_info["type"], "text": text, "position": box})

        return mf_out
