import base64
import copy
import io

import numpy as np
import requests
from PIL import Image
from shapely import Polygon


def save_pillow_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str


def get_hori_rect_v2(rot_rect):
    arr = np.asarray(rot_rect, dtype=np.float32).reshape((4, 2))
    x0 = np.min(arr[:, 0])
    x1 = np.max(arr[:, 0])
    y0 = np.min(arr[:, 1])
    y1 = np.max(arr[:, 1])
    return np.asarray[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def bbox_overlap(bbox0, bbox1):
    poly0 = Polygon(*bbox0.tolist())
    poly1 = Polygon(*bbox1.tolist())
    iou = poly0.intersection(poly1).area * 1.0 / poly1.area
    return iou


def is_valid_box(box, min_height=8, min_width=2) -> bool:
    # follow code from open project pix2text
    return (
        box[0, 0] + min_width <= box[1, 0]
        and box[1, 1] + min_height <= box[2, 1]
        and box[2, 0] >= box[3, 0] + min_width
        and box[3, 1] >= box[0, 1] + min_height
    )


def split_line_image(line_box, embed_mfs):
    # split line bbox by embedding formula bbox
    # code from open project pix2text
    line_box = line_box[0]
    if not embed_mfs:
        return [{"position": line_box.int().tolist(), "type": "text"}]
    embed_mfs.sort(key=lambda x: x["position"][0])

    outs = []
    start = int(line_box[0])
    xmax, ymin, ymax = int(line_box[2]), int(line_box[1]), int(line_box[-1])
    for mf in embed_mfs:
        _xmax = min(xmax, int(mf["position"][0]) + 1)
        if start + 8 < _xmax:
            outs.append({"position": [start, ymin, _xmax, ymax], "type": "text"})
        start = int(mf["position"][2])
    if start < xmax:
        outs.append({"position": [start, ymin, xmax, ymax], "type": "text"})
    return outs


# OCR Agent Version 0.1, update at 2023.08.18
#  - add predict_with_mask support recog with embedding formula, 2024.01.16
class OCRAgent(object):
    def __init__(self, **kwargs):
        self.ep = kwargs.get("ocr_model_ep")
        self.client = requests.Session()
        self.timeout = kwargs.get("timeout", 60)
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
            "print_recog": {
                "recog": "transformer-blank-v0.2-faster",
            },
            "hand_recog": {
                "recog": "transformer-hand-v1.16-faster",
            },
            "det": {
                "recog": "general_text_det_mrcnn_v2.0",
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
        except requests.exceptions.Timeout:
            raise Exception(f"timeout in ocr predict")
        except Exception as e:
            raise Exception(f"exception in ocr predict: [{e}]")

    def _get_ep_result(self, ep, inp):
        try:
            r = self.client.post(url=ep, json=inp, timeout=self.timeout)
            return r.json()
        except requests.exceptions.Timeout:
            raise Exception(f"timeout in formula agent predict")
        except Exception as e:
            raise Exception(f"exception in formula agent predict: [{e}]")

    def predict_with_mask(self, img0, mf_out, scene="print"):
        img = np.array(img0.copy())
        for box_info in mf_out:
            if box_info["type"] in ("isolated", "embedding"):
                box = box_info["box"]
                xmin, ymin = max(0, int(box[0][0]) - 1), max(0, int(box[0][1]) - 1)
                xmax, ymax = (
                    min(img0.size[0], int(box[2][0]) + 1),
                    min(img0.size[1], int(box[2][1]) + 1),
                )
                img[ymin:ymax, xmin:xmax, :] = 255

        b64_image = save_pillow_to_base64(Image.fromarray(img))
        params = copy.deepcopy(self.scene_mapping["det"])
        req_data = {"param": params, "data": [b64_image]}
        det_result = self._get_ep_result(self.ep, req_data)
        bboxes = det_result["result"]["ocr_result"]["bboxes"]

        EMB_BBOX_THREHOLD = 0.7
        text_bboxes = []
        for bbox in bboxes:
            hori_bbox = get_hori_rect_v2(bbox)
            if not is_valid_box(hori_bbox, min_height=8, min_width=2):
                continue

            embed_mfs = []
            for box_info in mf_out:
                if box_info["type"] == "embedding":
                    emb_bbox = box_info["position"]
                    bbox_iou = bbox_overlap(hori_bbox, emb_bbox)
                    if bbox_iou > EMB_BBOX_THREHOLD:
                        embed_mfs.append(
                            {
                                "position": box_info[0].int().tolist(),
                                "text": box_info["text"],
                                "type": box_info["type"],
                            }
                        )

            ocr_boxes = split_line_image(hori_bbox, embed_mfs)
            text_bboxes.extend(ocr_boxes)

        outs = copy(mf_out)
        # recog text for extracted text boxes
        params = copy.deepcopy(self.scene_mapping["print_recog"])
        req_data = {"param": params, "data": [b64_image], "bbox": text_bboxes}
        recog_result = self._get_ep_result(self.ep, req_data)
