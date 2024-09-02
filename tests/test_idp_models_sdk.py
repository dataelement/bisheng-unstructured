# flake8: noqa
import base64
import hashlib
import json
import os

import pytest

from bisheng_unstructured.models.layout_agent import LayoutAgent
from bisheng_unstructured.models.ocr_agent import OCRAgent
from bisheng_unstructured.models.table_agent import TableAgent, TableDetAgent

configs = dict(
    layout_ep="http://192.168.106.20:10502/v2/models/elem_layout_v1/infer",
    cell_model_ep="http://192.168.106.20:10502/v2/models/elem_table_cell_detect_v1/infer",
    rowcol_model_ep="http://192.168.106.20:10502/v2/models/elem_table_rowcol_detect_v1/infer",
    table_model_ep="http://192.168.106.20:10502/v2/models/elem_table_detect_v1/infer",
    ocr_model_ep="http://192.168.106.20:10502/v2/idp/idp_app/infer",
)


# @pytest.mark.skip
def test_layout():
    layout_agent = LayoutAgent(**configs)

    image_file = "data/001.png"
    b64_image = base64.b64encode(open(image_file, "rb").read()).decode("utf-8")
    inp = {"b64_image": b64_image}
    result = layout_agent.predict(inp)
    print("result", result)


# @pytest.mark.skip
def test_ocr():
    ocr_agent = OCRAgent(**configs)

    image_file = "data/001.png"
    b64_image = base64.b64encode(open(image_file, "rb").read()).decode("utf-8")
    inp = {"b64_image": b64_image}
    result = ocr_agent.predict(inp)
    print("result", result)


def test_table_det():
    table_det_agent = TableDetAgent(**configs)
    table_agent = TableAgent(**configs)
    ocr_agent = OCRAgent(**configs)

    image_file = "data/001.png"
    b64_image = base64.b64encode(open(image_file, "rb").read()).decode("utf-8")
    inp = {"b64_image": b64_image}
    table_bboxes = table_det_agent.predict(inp)["bboxes"]
    ocr_result = json.dumps(ocr_agent.predict(inp)["result"]["ocr_result"])
    inp = {"b64_image": b64_image, "table_bboxes": table_bboxes, "ocr_result": ocr_result}
    table_result = table_agent.predict(inp)
    print("table_result", table_result)
