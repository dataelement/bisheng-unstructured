# flake8: noqa
from dataclasses import dataclass
from typing import Any, List, Optional, Union


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
    layout_type: int = None  # 0: pragraph, 1: table
    html_text: str = None


class OCRAgent(object):
    def __init__(self, **kwargs):
        self.ep = kwargs.get("ocr_model_ep")
        self.client = requests.Session()
        self.timeout = kwargs.get("timeout", 60)

    def predict(self, inp) -> List[BlockInfo]:
        scene = inp.pop("scene", "print")
        b64_image = inp.pop("b64_image")
        params = copy.deepcopy(self.params)
        # todo:

        b0 = BlockInfo(
            block=[],
            block_text="abcdef",
            block_no=0,
            ts=["abc", "def"],
            rs=[[0, 0, 100, 30], [0, 50, 100, 80]],
            layout_type=0,
        )

        b1 = BlockInfo(
            block=[],
            block_text="| h1 | h2 | h3 |\n|-|-|-|\n| data1 | data2 | data3 |",
            block_no=0,
            ts=[""],
            rs=[[0, 50, 100, 80]],
            layout_type=1,
        )

        return [b0, b1]
