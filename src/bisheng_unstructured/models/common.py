import base64
import io
import re
from functools import cmp_to_key
from typing import Any, Dict, List, Union
import os
import json
import cv2
import numpy as np
from PIL import Image, ImageOps

# from shapely import Polygon
def load_json(file_path, object_hook=dict):
    if os.path.isfile(file_path):
        with open(file_path, 'r') as fid:
            data = json.load(fid, object_pairs_hook=object_hook)
    elif isinstance(file_path, str):
        data = json.loads(file_path, object_pairs_hook=object_hook)
    elif isinstance(file_path, bytes):
        # file_path = file_path.decode('utf-8').replace("'", '"')
        file_path = file_path.decode('utf-8')
        data = json.loads(file_path, object_pairs_hook=object_hook)
    else:
        exit('can not open this file')

    return data


def read_pil_image(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    return img


def draw_polygon(image, bbox, text=None, color=(255, 0, 0), thickness=1, is_rect=False):
    bbox = bbox.astype(np.int32)
    # is_rect = bbox.shape[0] == 4
    if is_rect:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
    else:
        polys = [bbox.astype(np.int32).reshape((-1, 1, 2))]
        cv2.polylines(image, polys, True, color=color, thickness=thickness)
        start_point = (polys[0][0, 0, 0], polys[0][0, 0, 1])

    if text:
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (0, 0, 255)
        image = cv2.putText(image, text, start_point, fontFace, fontScale, color, 1)

    return image


def save_pillow_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def pil2opencv(image):
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def opencv2pil(ndarray):
    return Image.fromarray(cv2.cvtColor(ndarray, cv2.COLOR_BGR2RGB))


def get_hori_rect_v2(rot_rect):
    arr = np.asarray(rot_rect, dtype=np.float32).reshape((4, 2))
    x0 = np.min(arr[:, 0])
    x1 = np.max(arr[:, 0])
    y0 = np.min(arr[:, 1])
    y1 = np.max(arr[:, 1])
    return [x0, y0, x1, y1]


def bbox_overlap(bbox0, bbox1):
    # bbox0: [x0, y0, x1, y1]
    # bbox1: [x0, y0, x1, y1]
    def bbox_area(bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    xmin, ymin = max(bbox0[0], bbox1[0]), max(bbox0[1], bbox1[1])
    xmax, ymax = min(bbox0[2], bbox1[2]), min(bbox0[3], bbox1[3])

    if xmin >= xmax or ymin >= ymax:
        return 0

    iou_area = (xmax - xmin) * (ymax - ymin)
    return iou_area / bbox_area(bbox1)


def bbox_iou(bbox0, bbox1):
    # bbox0: [x0, y0, x1, y1]
    # bbox1: [x0, y0, x1, y1]
    def bbox_area(bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    xmin, ymin = max(bbox0[0], bbox1[0]), max(bbox0[1], bbox1[1])
    xmax, ymax = min(bbox0[2], bbox1[2]), min(bbox0[3], bbox1[3])

    if xmin >= xmax or ymin >= ymax:
        return 0

    iou_area = (xmax - xmin) * (ymax - ymin)
    return iou_area / (bbox_area(bbox1) + bbox_area(bbox0) - iou_area)


def rotated_box_to_horizontal(box):
    """将旋转框转换为水平矩形。

    :param box: [4, 2]，左上角、右上角、右下角、左下角的坐标
    """
    xmin = min(box[:, 0])
    xmax = max(box[:, 0])
    ymin = min(box[:, 1])
    ymax = max(box[:, 1])
    return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])


def is_valid_box_v1(box, min_height=8, min_width=2) -> bool:
    """judge a valid box
    Args:
        box: [4, 2]
    """
    return (
        box[0, 0] + min_width <= box[1, 0]
        and box[1, 1] + min_height <= box[2, 1]
        and box[2, 0] >= box[3, 0] + min_width
        and box[3, 1] >= box[0, 1] + min_height
    )


def is_valid_box(box, min_height=8, min_width=2) -> bool:
    x0, y0, x1, y1 = box
    return x0 + min_width <= x1 and y0 + min_height <= y1


def overlap(box1, box2, key="position"):
    # 计算它们在y轴上的IOU: Interaction / min(height1, height2)
    # 判断是否有交集
    box1 = [box1[key][0][0], box1[key][0][1], box1[key][2][0], box1[key][2][1]]
    box2 = [box2[key][0][0], box2[key][0][1], box2[key][2][0], box2[key][2][1]]
    if box1[3] <= box2[1] or box2[3] <= box1[1]:
        return 0
    # 计算交集的高度
    y_min = max(box1[1], box2[1])
    y_max = min(box1[3], box2[3])
    return (y_max - y_min) / max(1, min(box1[3] - box1[1], box2[3] - box2[1]))


# following functions are from pix2text project with tiny modification
def get_same_line_boxes(anchor, total_boxes):
    line_boxes = [anchor]
    for box in total_boxes:
        if box["line_number"] >= 0:
            continue
        if max([overlap(box, l_box) for l_box in line_boxes]) > 0.1:
            line_boxes.append(box)
    return line_boxes


def _compare_box(box1, box2, anchor, key, left_best: bool = True):
    over1 = overlap(box1, anchor, key)
    over2 = overlap(box2, anchor, key)
    if box1[key][2, 0] < box2[key][0, 0] - 3:
        return -1
    elif box2[key][2, 0] < box1[key][0, 0] - 3:
        return 1
    else:
        if max(over1, over2) >= 3 * min(over1, over2):
            return over2 - over1 if left_best else over1 - over2
        return box1[key][0, 0] - box2[key][0, 0]


def sort_and_filter_line_boxes(line_boxes, key):
    if len(line_boxes) <= 1:
        return line_boxes

    allowed_max_overlay_x = 20

    def find_right_box(anchor):
        anchor_width = anchor[key][2, 0] - anchor[key][0, 0]
        allowed_max = min(max(allowed_max_overlay_x, anchor_width * 0.5), anchor_width * 0.95)
        right_boxes = [
            l_box
            for l_box in line_boxes[1:]
            if l_box["line_number"] < 0 and l_box[key][0, 0] >= anchor[key][2, 0] - allowed_max
        ]
        if not right_boxes:
            return None
        right_boxes = sorted(
            right_boxes,
            key=cmp_to_key(lambda x, y: _compare_box(x, y, anchor, key, left_best=True)),
        )
        return right_boxes[0]

    def find_left_box(anchor):
        anchor_width = anchor[key][2, 0] - anchor[key][0, 0]
        allowed_max = min(max(allowed_max_overlay_x, anchor_width * 0.5), anchor_width * 0.95)
        left_boxes = [
            l_box
            for l_box in line_boxes[1:]
            if l_box["line_number"] < 0 and l_box[key][2, 0] <= anchor[key][0, 0] + allowed_max
        ]
        if not left_boxes:
            return None
        left_boxes = sorted(
            left_boxes,
            key=cmp_to_key(lambda x, y: _compare_box(x, y, anchor, key, left_best=False)),
        )
        return left_boxes[-1]

    res_boxes = [line_boxes[0]]
    anchor = res_boxes[0]
    line_number = anchor["line_number"]

    while True:
        right_box = find_right_box(anchor)
        if right_box is None:
            break
        right_box["line_number"] = line_number
        res_boxes.append(right_box)
        anchor = right_box

    anchor = res_boxes[0]
    while True:
        left_box = find_left_box(anchor)
        if left_box is None:
            break
        left_box["line_number"] = line_number
        res_boxes.insert(0, left_box)
        anchor = left_box

    return res_boxes


def sort_boxes(boxes: List[dict], key="position") -> List[List[dict]]:
    # 按y坐标排序所有的框
    boxes.sort(key=lambda box: box[key][0, 1])
    for box in boxes:
        box["line_number"] = -1  # 所在行号，-1表示未分配

    def get_anchor():
        anchor = None
        for box in boxes:
            if box["line_number"] == -1:
                anchor = box
                break
        return anchor

    lines = []
    while True:
        anchor = get_anchor()
        if anchor is None:
            break
        anchor["line_number"] = len(lines)
        line_boxes = get_same_line_boxes(anchor, boxes)
        line_boxes = sort_and_filter_line_boxes(line_boxes, key)
        lines.append(line_boxes)

    return lines


def split_line_bbox_by_overlap_bbox(line_box, embed_box):
    # keep the order for split box and embed box
    if not embed_box:
        return [(line_box, "text", None)]

    ind = list(range(len(embed_box)))
    ind.sort(key=lambda x: embed_box[x][0])
    x0, y0, x1, y1 = line_box
    outs = []
    new_x0 = int(x0)
    MIN_WIDTH = 3
    for i in ind:
        _x1 = min(x1, int(embed_box[i][0]) + 1)
        if new_x0 + 8 < _x1:
            bb = [new_x0, y0, _x1, y1]
            outs.append((bb, "text", None))
            outs.append((None, "embed", i))
        else:
            outs.append((None, "embed", i))

        new_x0 = int(embed_box[i][2])

    if new_x0 < x1:
        bb = [new_x0, y0, x1, y1]
        if x1 - new_x0 > MIN_WIDTH:
            outs.append((bb, "text", None))

    return outs


def split_line_image(line_box, embed_mfs):
    # split line bbox by embedding formula bbox
    # code from open project pix2text
    # line_box = line_box[0]
    if not embed_mfs:
        return [{"position": line_box, "type": "text"}]
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


def list2box(xmin, ymin, xmax, ymax):
    return np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]], dtype=float)


def is_chinese(ch):
    return "\u4e00" <= ch <= "\u9fff"


def smart_join(str_list):
    def contain_whitespace(s):
        if re.search(r"\s", s):
            return True
        else:
            return False

    res = str_list[0]
    for i in range(1, len(str_list)):
        if (is_chinese(res[-1]) and is_chinese(str_list[i][0])) or (
            contain_whitespace(res[-1] + str_list[i][0])
        ):
            res += str_list[i]
        else:
            res += " " + str_list[i]
    return res


def merge_bbox(bboxes):
    bboxes = bboxes.reshape((-1, 8))
    x0 = np.min(bboxes[:, 0])
    y0 = np.min(bboxes[:, 1])
    x1 = np.max(bboxes[:, 4])
    y1 = np.max(bboxes[:, 5])
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def rect4p_to_rect2p(rect4p):
    return [rect4p[0][0], rect4p[0][1], rect4p[2][0], rect4p[2][1]]


def join_line_outs(outs):
    line_texts = []
    line_bboxes = []
    line_words_info = []
    for line_elems in outs:
        texts = [elem["text"] for elem in line_elems if elem["text"]]
        line_texts.append(smart_join(texts))
        bboxes = [elem["position"] for elem in line_elems]
        line_bboxes.append(merge_bbox(np.asarray(bboxes)))

        types = [elem["type"] for elem in line_elems]
        rects = [rect4p_to_rect2p(bb) for bb in bboxes]
        words_info = [texts, rects, types]
        line_words_info.append(words_info)

    return line_texts, line_bboxes, line_words_info
