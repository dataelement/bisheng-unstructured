# flake8: noqa
import copy
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import numpy as np
import copy
import requests


# 表示1个文本区域
@dataclass
class BlockInfo:
    bbox: List[Union[float, int]]  # block bbox, [x0, y0, x1, y1]
    block_text: str
    block_no: int
    block_type: int = 0
    ts: Any = None  # line texts, ['line1'， 'line2']
    rs: Any = None  # line bboxes, [[x0, y0, x1, y1]]
    ind: List[int] = None
    ord_ind: int = None
    layout_type: int = None  # 3: title 4: pragraph, 5: table
    html_text: str = None

def find_xy(box):
    xmin = box[0][0]
    ymin = box[0][1]
    xmax = box[0][0]
    ymax = box[0][1]

    for point in box:
        x, y = point
        xmin = min(xmin, x)
        ymin = min(ymin, y)
        xmax = max(xmax, x)
        ymax = max(ymax, y)
    return [xmin,ymin,xmax,ymax]

def random_merge_text(a, n):
    size = len(a) // n
    remainder = len(a) % n
    result = [''.join(a[i * size + min(i, remainder):(i + 1) * size + min(i + 1, remainder)]) for i in range(n)]
    return result

def recalculate_xy(arr):
    arr = np.array(arr)
    min_col1 = np.min(arr[:, 0])
    min_col2 = np.min(arr[:, 1])
    max_col3 = np.max(arr[:, 2])
    max_col4 = np.max(arr[:, 3])
    return [min_col1,min_col2,max_col3,max_col4]

def process_table(table_result):

    max_row = max([max(item['rows']) for item in table_result])
    max_col = max([max(item['cols']) for item in table_result])
    layout = [["" for _ in range(max_col+1)] for _ in range(max_row+1)]
    rect_box = []
    for item in table_result:
        if not len(item["text_box"]): continue
        bboxes = np.array(item["text_box"]).reshape((-1,4,2)).tolist()
        new_bboxes = np.array([find_xy(box) for box in bboxes])
        sub_rect_box = recalculate_xy(new_bboxes)
        rect_box.append(sub_rect_box)
        
        cols = item['cols']
        rows = item['rows'] if len(cols) == len(item['rows']) else item['rows']*len(cols)
        if len(cols) == len(item["text"]):
            text = item["text"]
        else:
            if len(cols) == 1:
                text = ["".join(item["text"])]
            else:
                text = random_merge_text(item["text"], len(cols))

        for row, col, txt in zip(rows,cols, text):
            layout[row][col] = txt

    md_str = ""
    for row in layout:
        md_str += " | ".join(row) + " \n\n"

    rect_box = recalculate_xy(rect_box)
    return md_str, rect_box


def process_paragraph(bboxes, texts, rect_box):
    up_table = {'boxs': [], 'texts':[]}
    down_table = {'boxs': [], 'texts':[]}
    if rect_box:
        bboxes = np.array(bboxes).reshape((-1,4,2))
        for idx, box in enumerate(bboxes):
            a,ymi,b,ymx = find_xy(box)
            if ymi < rect_box[1]:
                up_table['boxs'].append([a,ymi,b,ymx])
                up_table['texts'].append(texts[idx])
            elif ymx > rect_box[3]:
                down_table['boxs'].append([a,ymi,b,ymx])
                down_table['texts'].append(texts[idx])
    else:
        bboxes = np.array(bboxes).reshape((-1,4,2))
        for idx, box in enumerate(bboxes):
            a,ymi,b,ymx = find_xy(box)
            up_table['boxs'].append([a,ymi,b,ymx])
            up_table['texts'].append(texts[idx])

    return up_table, down_table

def process_whole_paragraph(general_ocr_res):
    boxes = general_ocr_res["bboxes"]
    texts = general_ocr_res["texts"]
    rowcol = general_ocr_res["row_col_info"]
    
    max_row = max(row[0] for row in rowcol) + 1
    max_col = max(row[1] for row in rowcol) + 1

    layout_text = [["" for _ in range(max_col)] for _ in range(max_row)]
    layout_boxs = [[[] for _ in range(max_col)] for _ in range(max_row)]

    # 填充layout
    for (row, col), txt, box in zip(rowcol, texts, boxes):
        layout_text[row][col] = txt
        layout_boxs[row][col] = box
    return layout_text, layout_boxs


class OCRAgent(object):
    def __init__(self, **kwargs):
        self.ep = kwargs.get("ocr_model_ep")
        self.client = requests.Session()
        self.timeout = kwargs.get("timeout", 60)

    def predict(self, inp) -> List[BlockInfo]:
        scene = inp.pop("scene", "print")
        b64_image = inp.pop("b64_image")

        params = copy.deepcopy(self.params)
        # params.update(self.scene_mapping[scene])
        params.update(inp)
        req_data = {"param": params, "data": [b64_image]}
        try:
            r = self.client.post(url=self.ep, json=req_data, timeout=self.timeout)
            
        except requests.exceptions.Timeout:
            raise Exception(f"timeout in formula agent predict")
        except Exception as e:
            raise Exception(f"exception in formula agent predict: [{e}]")
        
        layout_text, layout_boxs = process_whole_paragraph(r["data"]["json"]["general_ocr_res"]) 
        b0 = BlockInfo(
            block=[],
            block_text=''.join([''.join(text) for text in layout_text]),
            block_no=0,
            ts=[''.join(text) for text in layout_text],
            rs=[''.join(text) for text in layout_boxs],
            layout_type=0,
        )
        return [b0]
