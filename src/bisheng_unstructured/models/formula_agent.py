import base64
import copy
import io

import cv2
import numpy as np
import requests

from bisheng_unstructured.common import Timer

from .common import (
    bbox_overlap,
    draw_polygon,
    pil2opencv,
    save_pillow_to_base64,
    smart_join,
    split_line_bbox_by_overlap_bbox,
)


class FormulaAgent(object):
    def __init__(self, **kwargs):
        self.det_ep = kwargs.get("formula_det_model_ep")
        self.recog_ep = kwargs.get("formula_recog_model_ep")
        self.client = requests.Session()
        self.timeout = kwargs.get("timeout", 60)

        self.embed_sep = kwargs.get("embed_sep", (" $", "$ "))
        self.isolated_sep = kwargs.get("isolated_sep", ("$$\n", "\n$$"))

    def _get_ep_result(self, ep, inp):
        headers = {"Content-type": "application/json"}
        try:
            r = self.client.post(url=ep, json=inp, timeout=self.timeout, headers=headers)
            return r.json()
        except requests.exceptions.Timeout:
            raise Exception(f"timeout in formula agent predict")
        except Exception as e:
            raise Exception(f"exception in formula agent predict: [{e}]")

    def predict(self, inp, image, **kwargs):
        enable_isolated_formula = kwargs.get("enable_isolated_formula", False)
        det_out = self._get_ep_result(self.det_ep, inp)
        formula_det_out = det_out.get("result", [])
        mf_out = []
        for box_info in formula_det_out:
            box = box_info["box"]
            if enable_isolated_formula:
                if box_info["type"] != "isolated":
                    continue

            xmin, ymin, xmax, ymax = box[0], box[1], box[4], box[5]
            # xmin, ymin, xmax, ymax = (
            #     int(box[0][0]),
            #     int(box[0][1]),
            #     int(box[2][0]),
            #     int(box[2][1]),
            # )
            crop_patch = image.crop((xmin, ymin, xmax, ymax))
            patch_b64_image = save_pillow_to_base64(crop_patch)
            inp = {"b64_image": patch_b64_image}
            patch_out = self._get_ep_result(self.recog_ep, inp)["result"]
            sep = self.embed_sep
            if box_info["type"] == "isolated":
                sep = self.isolated_sep

            text = sep[0] + patch_out + sep[1]
            mf_out.append({"type": box_info["type"], "text": text, "box": box})
        print("mf_out", mf_out)
        return mf_out

    def _visualize(self, img0, bboxes, mf_out):
        cv_img = pil2opencv(img0)
        for bbox in bboxes:
            bbox = np.asarray(bbox)
            # bbox = np.asarray(bbox).reshape((4, 2))
            cv_img = draw_polygon(cv_img, bbox, is_rect=True)

        for info in mf_out:
            text = "e" if info["type"] == "embedding" else "f"
            bbox = np.asarray(info["box"])
            cv_img = draw_polygon(cv_img, bbox, text, color=(0, 0, 255), is_rect=True)

        cv2.imwrite("/public/bisheng/latex_data/yy.png", cv_img)

    def predict_with_text_block(self, b64_image, image, textpage_info, **kwargs):
        # 1) get formula det result
        # 2) calculate overlap between formula det result and text block
        # 3) mask the text for isolated formula area
        # 4) split the text line by embedding formula area
        timer = Timer()

        enable_isolated_formula = kwargs.get("enable_isolated_formula", False)
        inp = {"b64_image": b64_image}
        det_out = self._get_ep_result(self.det_ep, inp)
        formula_det_out = det_out.get("result", [])
        mf_out = []
        timer.toc()

        for box_info in formula_det_out:
            box = box_info["box"]
            if enable_isolated_formula:
                if box_info["type"] != "isolated":
                    continue

            xmin, ymin, xmax, ymax = box[0], box[1], box[4], box[5]
            crop_patch = image.crop((xmin, ymin, xmax, ymax))
            patch_b64_image = save_pillow_to_base64(crop_patch)
            inp = {"b64_image": patch_b64_image}
            patch_out = self._get_ep_result(self.recog_ep, inp)["result"]
            sep = self.embed_sep
            if box_info["type"] == "isolated":
                sep = self.isolated_sep

            text = sep[0] + patch_out + sep[1]
            bb = [xmin, ymin, xmax, ymax]
            mf_out.append({"type": box_info["type"], "text": text, "box": bb})

        timer.toc()

        # text_bbs = [b.bbox for b in textpage_info[0]]
        # self._visualize(image, text_bbs, mf_out)

        # process the isolated formula
        # calculate overlap matrix
        blocks, words_info = textpage_info
        mf_cnt = len(mf_out)
        texts_cnt = len(blocks)
        overlap_matrix = np.zeros((mf_cnt, texts_cnt))
        OVERLAP_THRESHOLD = 0.7
        for i in range(mf_cnt):
            for j in range(texts_cnt):
                bbox0 = mf_out[i]["box"]
                bbox1 = blocks[j].bbox
                overlap_matrix[i, j] = bbox_overlap(bbox0, bbox1)

        isolated_ind = []
        mask_ind = []
        replace_info = []
        for i in range(mf_cnt):
            if mf_out[i]["type"] == "isolated":
                ind = np.argwhere(overlap_matrix[i, :] > OVERLAP_THRESHOLD)[:, 0]
                isolated_ind.extend(ind)
                min_ind = np.min(ind)
                mask_ind.extend([j for j in ind if j != min_ind])
                replace_info.append((min_ind, mf_out[i]["text"]))

        # process for the embedding formula
        overlap_matrix2 = np.zeros((texts_cnt, mf_cnt))
        OVERLAP_THRESHOLD = 0.7
        for i in range(texts_cnt):
            if i in isolated_ind:
                continue

            for j in range(mf_cnt):
                bbox0 = blocks[i].bbox
                bbox1 = mf_out[j]["box"]
                overlap_matrix2[i, j] = bbox_overlap(bbox0, bbox1)

        # embedding formula will split the normal text line
        ocr_agent = kwargs.get("ocr_agent")
        for i in range(texts_cnt):
            if i in isolated_ind:
                continue

            ind = np.argwhere(overlap_matrix2[i, :] > OVERLAP_THRESHOLD)[:, 0]
            ind = [k for k in ind if mf_out[k]["type"] == "embedding"]
            if len(ind) == 0:
                continue

            mf_boxes = [mf_out[k]["box"] for k in ind]
            line_mf_out = [mf_out[k] for k in ind]
            line_bbox = blocks[i].bbox
            split_outs = split_line_bbox_by_overlap_bbox(line_bbox, mf_boxes)

            text_bboxes = []
            text_index_map = {}
            line_texts = [""] * len(split_outs)
            text_ind = 0
            for k in range(len(split_outs)):
                bbox, text_type, mf_ind = split_outs[k]
                if text_type == "text":
                    text_bboxes.append(bbox)
                    text_index_map[text_ind] = k
                    text_ind += 1
                else:
                    line_texts[k] = line_mf_out[mf_ind]["text"]

            # recog the patches
            if len(text_bboxes) > 0:
                patches = [image.crop(bb) for bb in text_bboxes]
                texts = ocr_agent.predict_with_patches(patches)
                for k, text in enumerate(texts):
                    line_texts[text_index_map[k]] = text

            line_texts = [t for t in line_texts if t != ""]
            line_full_text = smart_join(line_texts)
            blocks[i].block_text = line_full_text

        for ind, text in replace_info:
            blocks[ind].block_text = text
            blocks[ind].layout_type = 1000

        new_blocks = [blocks[i] for i in range(texts_cnt) if i not in mask_ind]
        new_words_info = [words_info[i] for i in range(texts_cnt) if i not in mask_ind]

        return new_blocks, new_words_info
