import base64
import copy
import io
from functools import cmp_to_key
from typing import Any, Dict, List, Union

import cv2
import os
import numpy as np
import requests
from PIL import Image

from .common import (
    bbox_overlap,
    draw_polygon,
    get_hori_rect_v2,
    is_valid_box,
    join_line_outs,
    list2box,
    pil2opencv,
    save_pillow_to_base64,
    sort_boxes,
    split_line_image,
    load_json
)

DEFAULT_CONFIG = {
    "params": {
            "sort_filter_boxes": True,
            "enable_huarong_box_adjust": True,
            "rotateupright": False,
            "support_long_image_segment": True,
            "split_long_sentence_blank": True,
        },
    "scene_mapping": {
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
                "det": "general_text_det_mrcnn_v2.0",
            },
        }
}


# OCR Agent Version 0.1, update at 2023.08.18
#  - add predict_with_mask support recog with embedding formula, 2024.01.16
class OCRAgent(object):
    def __init__(self, **kwargs):
        self.ep = kwargs.get("ocr_model_ep")
        self.client = requests.Session()
        self.timeout = kwargs.get("timeout", 60)
        mdoel_config_path = "/opt/bisheng-unstructured/model_config.json"
        if os.path.exists(mdoel_config_path):
            jsoncontent = load_json(mdoel_config_path)
        else:
            jsoncontent = None
        if jsoncontent is not None and "params" in jsoncontent and \
            "scene_mapping" in jsoncontent:
            self.params = jsoncontent["params"]
            self.scene_mapping = jsoncontent["scene_mapping"]
        else:
            self.params = DEFAULT_CONFIG["params"]
            self.scene_mapping = DEFAULT_CONFIG["scene_mapping"]

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

    def _visualize(self, img0, bboxes, mf_out):
        # draw bbox

        img0.save("/public/bisheng/latex_data/xx0.png", format="PNG")

        cv_img = pil2opencv(img0)
        for bbox in bboxes:
            bbox = np.asarray(get_hori_rect_v2(bbox))
            # bbox = np.asarray(bbox).reshape((4, 2))
            cv_img = draw_polygon(cv_img, bbox, is_rect=True)

        for info in mf_out:
            text = "e" if info["type"] == "embedding" else "f"
            bbox = np.asarray(info["box"]).reshape((4, 2))
            cv_img = draw_polygon(cv_img, bbox, text, color=(0, 0, 255))

        cv2.imwrite("/public/bisheng/latex_data/xx.png", cv_img)

    def predict_with_mask(self, img0, mf_out, scene="print", **kwargs):
        img = np.array(img0.copy())
        for box_info in mf_out:
            if box_info["type"] in ("isolated", "embedding"):
                box = np.asarray(box_info["box"]).reshape((4, 2))

                xmin, ymin = max(0, int(box[0][0]) - 1), max(0, int(box[0][1]) - 1)
                xmax, ymax = (
                    min(img0.size[0], int(box[2][0]) + 1),
                    min(img0.size[1], int(box[2][1]) + 1),
                )
                img[ymin:ymax, xmin:xmax, :] = 255

        masked_image = Image.fromarray(img)
        b64_image = save_pillow_to_base64(masked_image)
        # b64_image = save_pillow_to_base64(img0)

        params = copy.deepcopy(self.params)
        params.update(self.scene_mapping["det"])
        req_data = {"param": params, "data": [b64_image]}
        det_result = self._get_ep_result(self.ep, req_data)
        bboxes = det_result["result"]["boxes"]

        # self._visualize(masked_image, bboxes, mf_out)

        EMB_BBOX_THREHOLD = 0.7
        text_bboxes = []
        for bbox in bboxes:
            hori_bbox = get_hori_rect_v2(bbox)
            if not is_valid_box(hori_bbox, min_height=8, min_width=2):
                continue

            embed_mfs = []
            for box_info in mf_out:
                if box_info["type"] == "embedding":
                    bb = box_info["box"]
                    emb_bbox = [bb[0], bb[1], bb[4], bb[5]]
                    bbox_iou = bbox_overlap(hori_bbox, emb_bbox)
                    if bbox_iou > EMB_BBOX_THREHOLD:
                        embed_mfs.append(
                            {
                                "position": emb_bbox,
                                "text": box_info["text"],
                                "type": box_info["type"],
                            }
                        )

            ocr_boxes = split_line_image(hori_bbox, embed_mfs)
            text_bboxes.extend(ocr_boxes)

        # recog the patches
        recog_data = []
        for bbox in text_bboxes:
            b64_data = save_pillow_to_base64(masked_image.crop(bbox["position"]))
            recog_data.append(b64_data)

        params = copy.deepcopy(self.params)
        params.update(self.scene_mapping["print_recog"])
        req_data = {"param": params, "data": recog_data}
        recog_result = self._get_ep_result(self.ep, req_data)

        outs = []
        for bbox, text in zip(text_bboxes, recog_result["result"]["texts"]):
            bbs = list2box(*bbox["position"])
            outs.append({"text": text, "position": bbs, "type": "text"})

        for info in mf_out:
            bbs = np.asarray(info["box"]).reshape((4, 2))
            outs.append({"text": info["text"], "position": bbs, "type": info["type"]})

        outs = sort_boxes(outs, key="position")
        texts, bboxes, words_info = join_line_outs(outs)

        # self._visualize(masked_image, bboxes, [])
        return texts, bboxes, words_info

    def predict_with_patches(self, pil_images, scene="print_recog", **kwargs):
        recog_data = []
        for pil_img in pil_images:
            b64_data = save_pillow_to_base64(pil_img)
            recog_data.append(b64_data)

        params = copy.deepcopy(self.params)
        params.update(self.scene_mapping[scene])
        req_data = {"param": params, "data": recog_data}
        recog_result = self._get_ep_result(self.ep, req_data)
        return recog_result["result"]["texts"]
