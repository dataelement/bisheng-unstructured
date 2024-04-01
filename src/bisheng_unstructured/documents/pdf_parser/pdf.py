"""Loads PDF with semantic partition."""
import base64
import io
import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List, Optional, Union

import fitz as pymupdf
import numpy as np
import pypdfium2
from PIL import Image, ImageOps
from shapely import Polygon
from shapely import box as Rect

from bisheng_unstructured.common import Timer
from bisheng_unstructured.documents.base import Document, Page
from bisheng_unstructured.documents.elements import (
    ElementMetadata,
    NarrativeText,
    Table,
    Text,
    Title,
)
from bisheng_unstructured.documents.markdown import (
    clean_html_table,
    merge_html_tables,
    merge_md_tables,
    transform_html_table_to_md,
    transform_list_to_table,
)
from bisheng_unstructured.models import (
    FormulaAgent,
    LayoutAgent,
    OCRAgent,
    TableAgent,
    TableDetAgent,
)

from .blob import Blob

ZH_CHAR = re.compile("[\u4e00-\u9fa5]")
ENG_WORD = re.compile(pattern=r"^[a-zA-Z0-9?><;,{}[\]\-_+=!@#$%\^&*|']*$", flags=re.DOTALL)
RE_MULTISPACE_INCLUDING_NEWLINES = re.compile(pattern=r"\s+", flags=re.DOTALL)


def read_image(path):
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    return img


def merge_rects(bboxes):
    x0 = np.min(bboxes[:, 0])
    y0 = np.min(bboxes[:, 1])
    x1 = np.max(bboxes[:, 2])
    y1 = np.max(bboxes[:, 3])
    return [x0, y0, x1, y1]


def norm_rect(bbox):
    x0 = np.min([bbox[0], bbox[2]])
    x1 = np.max([bbox[0], bbox[2]])
    y0 = np.min([bbox[1], bbox[3]])
    y1 = np.max([bbox[1], bbox[3]])
    return np.asarray([x0, y0, x1, y1])


def get_hori_rect(rot_rect):
    arr = np.asarray(rot_rect, dtype=np.float32).reshape((4, 2))
    x0 = np.min(arr[:, 0])
    x1 = np.max(arr[:, 0])
    y0 = np.min(arr[:, 1])
    y1 = np.max(arr[:, 1])
    return [float(e) for e in (x0, y0, x1, y1)]


def find_max_continuous_seq(arr):
    n = len(arr)
    max_info = (0, 1)
    for i in range(n):
        m = 1
        for j in range(i + 1, n):
            if arr[j] - arr[j - 1] == 1:
                m += 1
            else:
                break

        if m > max_info[1]:
            max_info = (i, m)

    max_info = (max_info[0] + arr[0], max_info[1])
    return max_info


def order_by_tbyx(block_info, th=10):
    """
    block_info: [(b0, b1, b2, b3, text, x, y)+]
    th: threshold of the position threshold
    """
    # sort using y1 first and then x1
    res = sorted(block_info, key=lambda b: (b.bbox[1], b.bbox[0]))
    for i in range(len(res) - 1):
        for j in range(i, 0, -1):
            # restore the order using the
            bbox_jplus1 = res[j + 1].bbox
            bbox_j = res[j].bbox
            if abs(bbox_jplus1[1] - bbox_j[1]) < th and (bbox_jplus1[0] < bbox_j[0]):
                tmp = deepcopy(res[j])
                res[j] = deepcopy(res[j + 1])
                res[j + 1] = deepcopy(tmp)
            else:
                break
    return res


def is_eng_word(word):
    return bool(ENG_WORD.search(word))


def rect2polygon(bboxes):
    polys = []
    for x0, y0, x1, y1 in bboxes:
        polys.append([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
    return polys


def join_lines(texts, is_table=False, lang="eng"):
    if is_table:
        return "\n".join(texts)

    PUNC_SET = set([".", ",", ";", "?", "!"])
    if lang == "eng":
        t0 = texts[0]
        for t in texts[1:]:
            if t0[-1] == "-":
                t0 = t0[:-1] + t
            elif t0[-1].isalnum() and t[0].isalnum():
                t0 += " " + t
            elif t0[-1] in PUNC_SET or t[0] in PUNC_SET:
                t0 += " " + t
            else:
                t0 += t
        return t0
    else:
        return "".join(texts)


@dataclass
class BlockInfo:
    bbox: List[Union[float, int]]
    block_text: str
    block_no: int
    block_type: int
    ts: Any = None
    rs: Any = None
    ind: List[int] = None
    ord_ind: int = None
    layout_type: int = None
    html_text: str = None


class Segment:
    def __init__(self, seg):
        self.whole = seg
        self.segs = []

    @staticmethod
    def is_align(seg0, seg1, delta=5, mode=0):
        # mode=0 edge align
        # mode=1, edge align or center align
        res = Segment.contain(seg0, seg1)
        if not res:
            return False
        else:
            if mode == 1:
                r1 = seg1[0] - seg0[0] <= delta or seg0[1] - seg1[1] <= delta
                c0 = (seg0[0] + seg0[1]) / 2
                c1 = (seg1[0] + seg1[1]) / 2
                r2 = abs(c1 - c0) <= delta
                return r1 or r2
            else:
                return seg1[0] - seg0[0] <= delta or seg0[1] - seg1[1] <= delta

    @staticmethod
    def contain(seg0, seg1):
        return seg0[0] <= seg1[0] and seg0[1] >= seg1[0]

    @staticmethod
    def overlap(seg0, seg1):
        max_x0 = max(seg0[0], seg1[0])
        min_x1 = min(seg0[1], seg1[1])
        return max_x0 < min_x1

    def _merge(self, segs):
        x0s = [s[0] for s in segs]
        x1s = [s[1] for s in segs]
        return (np.min(x0s), np.max(x1s))

    def add(self, seg):
        if not self.segs:
            self.segs.append(seg)
        else:
            overlaps = []
            non_overlaps = []
            for seg0 in self.segs:
                if Segment.overlap(seg0, seg):
                    overlaps.append(seg0)
                else:
                    non_overlaps.append(seg0)

            if not overlaps:
                self.segs.append(seg)
            else:
                overlaps.append(seg)
                new_seg = self._merge(overlaps)
                non_overlaps.append(new_seg)
                self.segs = non_overlaps

    def get_free_segment(self, incr_margin=True, margin_threshold=10):
        sorted_segs = sorted(self.segs, key=lambda x: x[0])
        n = len(sorted_segs)
        free_segs = []
        if incr_margin:
            if n > 0:
                seg_1st = sorted_segs[0]
                if (seg_1st[0] - self.whole[0]) > margin_threshold:
                    free_segs.append((self.whole[0], seg_1st[0]))

                seg_last = sorted_segs[-1]
                if (self.whole[1] - seg_last[1]) > margin_threshold:
                    free_segs.append((seg_last[1], self.whole[1]))

        for i in range(n - 1):
            x0 = sorted_segs[i][1]
            x1 = sorted_segs[i + 1][0]
            free_segs.append((x0, x1))

        return free_segs


class PDFDocument(Document):
    """Loads a PDF with pdf and chunks at character level.

    Loader also stores page numbers in metadata.
    """

    def __init__(
        self,
        file: str,
        model_params: dict,
        password: Optional[Union[str, bytes]] = None,
        is_join_table: bool = True,
        with_columns: bool = False,
        support_rotate: bool = False,
        text_elem_sep: str = "\n",
        start: int = 0,
        n: int = None,
        verbose: bool = False,
        enhance_table: bool = True,
        keep_text_in_image: bool = True,
        support_formula: bool = False,
        enable_isolated_formula: bool = False,
        n_parallel: int = 10,
        **kwargs,
    ) -> None:
        """Initialize with a file path."""
        self.layout_agent = LayoutAgent(**model_params)
        self.table_agent = TableAgent(**model_params)
        self.ocr_agent = OCRAgent(**model_params)
        self.table_det_agent = TableDetAgent(**model_params)
        self.formula_agent = FormulaAgent(**model_params)

        self.with_columns = with_columns
        self.is_join_table = is_join_table
        self.support_rotate = support_rotate
        self.start = start
        self.n = n
        self.verbose = verbose
        self.text_elem_sep = text_elem_sep
        self.file = file
        self.enhance_table = enhance_table
        self.keep_text_in_image = keep_text_in_image
        self.support_formula = support_formula
        self.enable_isolated_formula = enable_isolated_formula
        self.n_parallel = n_parallel
        super().__init__()

    def _get_image_blobs(self, fitz_doc, pdf_reader, n=None, start=0):
        blobs = []
        pages = []
        if not n:
            n = fitz_doc.page_count
        for pg in range(start, start + n):
            bytes_img = None
            page = fitz_doc.load_page(pg)
            pages.append(page)
            mat = pymupdf.Matrix(1, 1)
            try:
                pm = page.get_pixmap(matrix=mat, alpha=False)
                bytes_img = pm.getPNGData()
            except Exception:
                # some pdf input cannot get render image from fitz
                page = pdf_reader.get_page(pg)
                pil_image = page.render().to_pil()
                img_byte_arr = io.BytesIO()
                pil_image.save(img_byte_arr, format="PNG")
                bytes_img = img_byte_arr.getvalue()

            blobs.append(Blob(data=bytes_img))
        return blobs, pages

    def _extract_lines_v2(self, textpage):
        line_blocks = []
        line_words_info = []
        page_dict = textpage.extractRAWDICT()
        for block in page_dict["blocks"]:
            block_type = block["type"]
            block_no = block["number"]
            if block_type != 0:
                bbox = block["bbox"]
                block_text = ""
                block_info = BlockInfo(
                    [bbox[0], bbox[1], bbox[2], bbox[3]], block_text, block_no, block_type
                )
                line_blocks.append(block_info)
                line_words_info.append((None, None))

            lines = block["lines"]

            for line in lines:
                bbox = line["bbox"]
                words = []
                words_bboxes = []
                for span in line["spans"]:
                    cont_bboxes = []
                    cont_text = []
                    for char in span["chars"]:
                        c = char["c"]
                        if c == " ":
                            if cont_bboxes:
                                word_bbox = merge_rects(np.asarray(cont_bboxes))
                                word = "".join(cont_text)
                                words.append(word)
                                words_bboxes.append(word_bbox)
                                cont_bboxes = []
                                cont_text = []
                        else:
                            cont_bboxes.append(char["bbox"])
                            cont_text.append(c)

                    if cont_bboxes:
                        word_bbox = merge_rects(np.asarray(cont_bboxes))
                        word = "".join(cont_text)
                        words.append(word)
                        words_bboxes.append(word_bbox)

                if not words_bboxes:
                    continue

                line_words_info.append((words, words_bboxes))
                line_text = "".join([char["c"] for span in line["spans"] for char in span["chars"]])
                bb0, bb1, bb2, bb3 = merge_rects(np.asarray(words_bboxes))

                block_info = BlockInfo([bb0, bb1, bb2, bb3], line_text, block_no, block_type)
                line_blocks.append(block_info)

        return line_blocks, line_words_info

    def _extract_lines(self, textpage):
        line_blocks = []
        line_words_info = []
        page_dict = textpage.extractDICT()
        for block in page_dict["blocks"]:
            block_type = block["type"]
            block_no = block["number"]
            if block_type != 0:
                bbox = block["bbox"]
                block_text = ""
                block_info = BlockInfo(
                    [bbox[0], bbox[1], bbox[2], bbox[3]], block_text, block_no, block_type
                )
                line_blocks.append(block_info)
                line_words_info.append((None, None))

            lines = block["lines"]

            for line in lines:
                bbox = line["bbox"]
                line_text = []

                words = [span["text"] for span in line["spans"]]
                words_bbox = [span["bbox"] for span in line["spans"]]
                line_words_info.append((words, words_bbox))

                line_text = "".join([span["text"] for span in line["spans"]])
                block_info = BlockInfo(
                    [bbox[0], bbox[1], bbox[2], bbox[3]], line_text, block_no, block_type
                )
                line_blocks.append(block_info)

        return line_blocks, line_words_info

    def _extract_blocks(self, textpage, lang):
        blocks = []
        blocks_words_info = []
        page_dict = textpage.extractDICT()
        for block in page_dict["blocks"]:
            block_type = block["type"]
            block_no = block["number"]
            block_bbox = block["bbox"]
            if block_type != 0:
                block_text = ""
                block_info = BlockInfo(
                    [block_bbox[0], block_bbox[1], block_bbox[2], block_bbox[3]],
                    block_text,
                    block_no,
                    block_type,
                )
                blocks.append(block_info)
                blocks_words_info.append((None, None))

            lines = block["lines"]
            block_words = []
            block_words_bbox = []
            block_lines = []
            for line in lines:
                block_words.extend([span["text"] for span in line["spans"]])
                block_words_bbox.extend([span["bbox"] for span in line["spans"]])
                line_text = "".join([span["text"] for span in line["spans"]])
                block_lines.append(line_text)

            block_text = join_lines(block_lines, False, lang)
            block_info = BlockInfo(
                [block_bbox[0], block_bbox[1], block_bbox[2], block_bbox[3]],
                block_text,
                block_no,
                block_type,
            )
            blocks.append(block_info)
            blocks_words_info.append((block_words, block_words_bbox))

        return blocks, blocks_words_info

    def _extract_blocks_from_image(self, b64_image, img):
        inp = {"b64_image": b64_image}
        if self.support_formula:
            # step 1. formula detection
            # step 2. crop the formula region and call formula recognition
            # step 3. mask the formula region and call general ocr
            # step 4. split the text lines by embedding formula region
            # step 5. recog the segmented line patches
            # step 6. merge the segmented line patches and sort by tlbr
            inp = {"b64_image": b64_image}
            mf_outs = self.formula_agent.predict(
                inp, img, enable_isolated_formula=self.enable_isolated_formula
            )
            texts, bboxes, words_info = self.ocr_agent.predict_with_mask(img, mf_outs)
        else:
            # get general ocr result
            ocr_result = self.ocr_agent.predict(inp)
            texts = ocr_result["result"]["ocr_result"]["texts"]
            bboxes = ocr_result["result"]["ocr_result"]["bboxes"]
            words_info = []

        blocks = []
        blocks_words_info = []
        block_type = 0
        for i in range(len(texts)):
            block_no = i
            block_text = texts[i]
            b0, b1, b2, b3 = get_hori_rect(bboxes[i])
            block_type = 0
            if words_info:
                if "isolated" in words_info[i][2]:
                    block_type = 1000

            block_info = BlockInfo([b0, b1, b2, b3], block_text, block_no, block_type)

            blocks.append(block_info)
            if not words_info:
                blocks_words_info.append(([block_text], [[b0, b1, b2, b3]]))
            else:
                blocks_words_info.append((words_info[i][0], words_info[i][1]))

        return blocks, blocks_words_info

    def _enhance_table_layout(self, b64_image, layout_blocks):
        TABLE_ID = 5
        inp = {"b64_image": b64_image}
        result = self.table_det_agent.predict(inp)
        table_layout = []
        for bb in result["bboxes"]:
            coords = ((bb[0], bb[1]), (bb[2], bb[3]), (bb[4], bb[5]), (bb[6], bb[7]))
            poly = Polygon(coords)
            table_layout.append((poly, TABLE_ID))

        general_table_layout = []
        result_layout = []
        for e in layout_blocks["result"]:
            bb = e["bbox"]
            coords = ((bb[0], bb[1]), (bb[2], bb[3]), (bb[4], bb[5]), (bb[6], bb[7]))
            poly = Polygon(coords)
            label = e["category_id"]
            if label == TABLE_ID:
                general_table_layout.append((poly, label))
            else:
                result_layout.append((poly, label))

        # merge general table layout with specific table layout
        OVERLAP_THRESHOLD = 0.7
        mask = np.zeros(len(general_table_layout))
        for i, (poly0, cate0) in enumerate(general_table_layout):
            for poly1, _ in table_layout:
                biou = poly0.intersection(poly1).area * 1.0 / poly1.area
                if biou >= OVERLAP_THRESHOLD:
                    mask[i] = 1
                    break

        for e in table_layout:
            result_layout.append(e)
        for i, e in enumerate(general_table_layout):
            if mask[i] == 0:
                result_layout.append(e)

        semantic_polys = [e[0] for e in result_layout]
        semantic_labels = [e[1] for e in result_layout]

        return semantic_polys, semantic_labels

    def _enhance_texts_info_with_formula(self, b64_image, img, textpage_info):
        if self.support_formula:
            blocks, words = self.formula_agent.predict_with_text_block(
                b64_image,
                img,
                textpage_info,
                enable_isolated_formula=self.enable_isolated_formula,
                ocr_agent=self.ocr_agent,
            )
            return blocks, words
        else:
            return textpage_info

    def _allocate_semantic(
        self, textpage_info, layout, b64_image, img, is_scan=True, lang="zh", rot_matrix=None
    ):
        class_name = ["印章", "图片", "标题", "段落", "表格", "页眉", "页码", "页脚"]
        effective_class_inds = [3, 4, 5, 999, 1000]
        non_conti_class_ids = [6, 7, 8]
        TEXT_ID = 4
        TABLE_ID = 5
        IMAGE_ID = 2
        FORMULA_ID = 1000

        timer = Timer()
        if not is_scan:
            # textpage = page.get_textpage()
            # blocks = textpage.extractBLOCKS()
            # blocks, words = self._extract_blocks(textpage)
            # blocks, words = self._extract_lines(textpage)
            # blocks, words = self._extract_lines_v2(textpage)
            # blocks, words = textpage_info
            blocks, words = self._enhance_texts_info_with_formula(b64_image, img, textpage_info)
        else:
            blocks, words = self._extract_blocks_from_image(b64_image, img)

        timer.toc()
        # print('---line blocks---')
        # for b in blocks:
        #     print(b)

        # Phrase0. Support rotated pdf page, some pdf page are rotated
        # and the rotation matrix is stored in the layout, rotate it first
        if self.support_rotate and is_scan:
            rotation_matrix = np.asarray(rot_matrix).reshape((3, 2))
            c1 = (rotation_matrix[0, 0] - 1) <= 1e-6
            c2 = (rotation_matrix[1, 1] - 1) <= 1e-6
            is_rotated = c1 and c2
            # print('c1/c2', c1, c2)
            if is_rotated:
                # new_blocks = []
                new_words = []
                for b, w in zip(blocks, words):
                    bbox = np.asarray(b.bbox)
                    aug_bbox = bbox.reshape((-1, 2))
                    padding = np.ones((len(aug_bbox), 1))
                    aug_bbox = np.hstack([aug_bbox, padding])
                    bb = np.dot(aug_bbox, rotation_matrix).reshape(-1)
                    bb = norm_rect(bb)
                    b.bbox = [bb[0], bb[1], bb[2], bb[3]]

                    # process for words
                    words_text, words_bb = w
                    if words_bb is None:
                        new_words.append(w)
                        continue

                    new_words_bb = []
                    for w_b in words_bb:
                        bbox = np.asarray(w_b)
                        aug_bbox = bbox.reshape((-1, 2))
                        padding = np.ones((len(aug_bbox), 1))
                        aug_bbox = np.hstack([aug_bbox, padding])
                        bb = np.dot(aug_bbox, rotation_matrix).reshape(-1)
                        bb = norm_rect(bb).tolist()
                        new_words_bb.append(bb)

                    new_words.append((words_text, new_words_bb))

                # blocks = new_blocks
                words = new_words

        timer.toc()
        # if not self.with_columns:
        #     blocks = order_by_tbyx(blocks)

        # print('---ori blocks---')
        # for b in blocks:
        #     print(b)

        IMG_BLOCK_TYPE = 1
        text_ploys = []
        text_rects = []
        texts = []
        for b in blocks:
            texts.append(b.block_text)
            text_ploys.append(Rect(*b.bbox))
            text_rects.append(b.bbox)

        text_rects = np.asarray(text_rects)
        texts = np.asarray(texts)

        semantic_polys = []
        semantic_labels = []

        # Enhance the table layout, the table layout is not very accurate in
        # a layout model, use the independent table detection model to enhance
        layout_info = layout
        # print('layout_info', layout_info)
        if self.enhance_table:
            semantic_polys, semantic_labels = self._enhance_table_layout(b64_image, layout)
        else:
            for info in layout_info["result"]:
                bbs = info["bbox"]
                coords = ((bbs[0], bbs[1]), (bbs[2], bbs[3]), (bbs[4], bbs[5]), (bbs[6], bbs[7]))
                semantic_polys.append(Polygon(coords))
                semantic_labels.append(info["category_id"])

        timer.toc()

        # phrase 1. merge continuous text block by the containing matrix
        # 1) calculate the overlap between semantic and text bboxes
        # 2) find max continuous text blocks with threshold keep.
        # 3) merge the continuous text blocks
        # 4) get the new blocks
        semantic_bboxes = []
        for poly in semantic_polys:
            x, y = poly.exterior.coords.xy
            semantic_bboxes.append([x[0], y[0], x[1], y[1], x[2], y[2], x[3], y[3]])

        # calculate containing overlap
        sem_cnt = len(semantic_polys)
        texts_cnt = len(text_ploys)
        contain_matrix = np.zeros((sem_cnt, texts_cnt))
        for i in range(sem_cnt):
            for j in range(texts_cnt):
                inter = semantic_polys[i].intersection(text_ploys[j]).area
                contain_matrix[i, j] = inter * 1.0 / text_ploys[j].area

        # print('----------------containing matrix--------')
        # for r in contain_matrix.tolist():
        #     print([round(r_, 2) for r_ in r])

        # print('---text---')
        # for t in texts:
        #     print(t)

        CONTRAIN_THRESHOLD = 0.70
        contain_info = []
        for i in range(sem_cnt):
            ind = np.argwhere(contain_matrix[i, :] > CONTRAIN_THRESHOLD)[:, 0]
            if len(ind) == 0:
                continue
            label = semantic_labels[i]
            if label in non_conti_class_ids:
                n = len(ind)
                contain_info.append((None, None, n, label, ind))
            else:
                start, n = find_max_continuous_seq(ind)
                if n >= 1:
                    contain_info.append((start, start + n, n, label, None))

        contain_info = sorted(contain_info, key=lambda x: x[2], reverse=True)
        mask = np.zeros(texts_cnt)
        new_block_info = []
        for info in contain_info:
            start, end, n, label, ind = info
            if label in non_conti_class_ids and np.all(mask[ind] == 0):
                rect = merge_rects(text_rects[ind])
                ori_orders = [blocks[i].block_no for i in ind]
                ts = texts[ind]
                rs = text_rects[ind]
                ord_ind = np.min(ori_orders)
                mask[ind] = 1
                new_block_info.append(
                    BlockInfo(
                        [rect[0], rect[1], rect[2], rect[3]], "", -1, -1, ts, rs, ind, ord_ind
                    )
                )
                max_block_type = np.max([blocks[i].block_type for i in ind])
                if max_block_type == FORMULA_ID:
                    new_block_info[-1].layout_type = FORMULA_ID

            elif np.all(mask[start:end] == 0):
                rect = merge_rects(text_rects[start:end])
                ori_orders = [blocks[i].block_no for i in range(start, end)]
                arg_ind = np.argsort(ori_orders)
                # print('ori_orders', ori_orders, arg_ind)
                ord_ind = np.min(ori_orders)

                ts = texts[start:end]
                rs = text_rects[start:end]
                if label == TABLE_ID:
                    ts = ts[arg_ind]
                    rs = rs[arg_ind]

                pos = np.arange(start, end)
                mask[start:end] = 1

                new_block_info.append(
                    BlockInfo(
                        [rect[0], rect[1], rect[2], rect[3]], "", -1, -1, ts, rs, pos, ord_ind
                    )
                )

                max_block_type = np.max([blocks[i].block_type for i in pos])
                if max_block_type == FORMULA_ID:
                    new_block_info[-1].layout_type = FORMULA_ID

        for i in range(texts_cnt):
            if mask[i] == 0:
                b = blocks[i]
                r = np.asarray(b.bbox)
                ord_ind = b.block_no

                new_block_info.append(BlockInfo(b.bbox, "", -1, -1, [texts[i]], [r], [i], ord_ind))
                if b.block_type == FORMULA_ID:
                    new_block_info[-1].layout_type = FORMULA_ID

        timer.toc()

        if self.with_columns:
            new_blocks = sorted(new_block_info, key=lambda x: x.ord_ind)
        else:
            new_blocks = order_by_tbyx(new_block_info)

        # print('\n\n---new blocks---')
        # for idx, b in enumerate(new_blocks):
        #     print(idx, b)

        text_ploys = []
        # texts = []
        for b in new_blocks:
            # texts.append(b.ts)
            text_ploys.append(Rect(*b.bbox))

        # calculate overlap
        sem_cnt = len(semantic_polys)
        texts_cnt = len(text_ploys)
        overlap_matrix = np.zeros((sem_cnt, texts_cnt))
        for i in range(sem_cnt):
            for j in range(texts_cnt):
                inter = semantic_polys[i].intersection(text_ploys[j]).area
                union = semantic_polys[i].union(text_ploys[j]).area
                overlap_matrix[i, j] = (inter * 1.0) / union

        # print('---overlap_matrix---')
        # for r in overlap_matrix:
        #     print([round(r_, 3) for r_ in r])
        # print('---semantic_labels---', semantic_labels)

        # phrase 2. allocate label by the layout information
        # 1) calculate the overlap between semantic and merge block bboxes
        # 2) find the most overlap semantic label, allocate it
        # 3) for table, parse the table layout
        # 4) for formula, keep it.
        OVERLAP_THRESHOLD = 0.2
        # texts_labels = []
        DEF_SEM_LABEL = 999
        FORMULA_LABEL = 1000
        table_infos = []
        for j in range(texts_cnt):
            ind = np.argwhere(overlap_matrix[:, j] > OVERLAP_THRESHOLD)[:, 0]
            if len(ind) == 0:
                sem_label = DEF_SEM_LABEL
            else:
                c = Counter([semantic_labels[i] for i in ind])
                items = c.most_common()
                sem_label = items[0][0]
                if len(items) > 1 and TEXT_ID in dict(items):
                    sem_label = TEXT_ID

            # Don't allocate the isolated formula block again
            if new_blocks[j].layout_type == FORMULA_ID:
                continue

            if sem_label == TABLE_ID:
                b = new_blocks[j]
                # b_inds = b[-2]
                b_inds = b.ind
                texts = []
                bboxes = []
                for k in b_inds:
                    for t, b_ in zip(words[k][0], words[k][1]):
                        if not t.strip():
                            continue
                        texts.append(t)
                        bboxes.append(b_)

                table_bbox = semantic_bboxes[ind[0]]
                table_infos.append((j, texts, bboxes, table_bbox))

            new_blocks[j].layout_type = sem_label

            # texts_labels.append(sem_label)

        timer.toc()

        # Parse the table layout
        table_layout = []
        for table_info in table_infos:
            block_ind, texts, bboxes, table_bbox = table_info
            if not texts:
                continue
            ocr_result = {"texts": texts, "bboxes": rect2polygon(bboxes)}

            inp = {
                "b64_image": b64_image,
                "ocr_result": json.dumps(ocr_result),
                "table_bboxes": [table_bbox],
                "scene": "cell",
            }
            table_result = self.table_agent.predict(inp)
            # print('---table--', ocr_result, table_bbox, table_result)
            h_bbox = get_hori_rect(table_bbox)

            if not table_result["htmls"]:
                # table layout parse failed, manually construce table
                b = new_blocks[block_ind]
                html = transform_list_to_table(b.ts)
                table_layout.append((block_ind, html, h_bbox))
                # print('---missing table---', block_ind, html)
            else:
                table_layout.append((block_ind, table_result["htmls"][0], h_bbox))

        for i, table_html, h_bbox in table_layout:
            table_md = transform_html_table_to_md(table_html)
            text = table_md["text"]
            html = table_md["html"]
            b = new_blocks[i]
            b.bbox = [h_bbox[0], h_bbox[1], h_bbox[2], h_bbox[3]]
            b.layout_type = TABLE_ID
            b.html_text = html
            b.block_text = text
        timer.toc()
        # print(texts_labels)
        # filter the unused element

        filtered_blocks = []
        for b in new_blocks:
            # ori_i = b[6][0]
            # ori_b = blocks[ori_i]
            # block_type = ori_b[-1]

            # filter the image block in real pdf
            if b.block_type == IMG_BLOCK_TYPE:
                continue

            # filter the empty text block
            if np.all([len(t) == 0 for t in b.ts]):
                continue

            label = b.layout_type
            if label == TABLE_ID:
                filtered_blocks.append(b)

            elif label == IMAGE_ID:
                if self.keep_text_in_image:
                    b.block_text = join_lines(b.ts, False, lang)
                    filtered_blocks.append(b)

            elif label in effective_class_inds:
                b.block_text = join_lines(b.ts, False, lang)
                filtered_blocks.append(b)

        # print('---filtered_blocks---')
        # for b in filtered_blocks:
        #     print(b)

        timer.toc()
        # print('_allocate_semantic', timer.get())
        return filtered_blocks

    def _divide_blocks_into_groups(self, blocks):
        # support only pure two columns layout, each has same width
        rects = np.asarray([b.bbox for b in blocks])
        min_x0 = np.min(rects[:, 0])
        max_x1 = np.max(rects[:, 2])
        root_seg = (min_x0, max_x1)
        root_pc = (min_x0 + max_x1) / 2
        root_offset = 20
        center_seg = (root_pc - root_offset, root_pc + root_offset)

        segment = Segment(root_seg)
        for r in rects:
            segment.add((r[0], r[2]))

        COLUMN_THRESHOLD = 0.90
        CENTER_GAP_THRESHOLD = 0.90
        free_segs = segment.get_free_segment()
        columns = []
        if len(free_segs) == 1 and len(segment.segs) == 2:
            free_seg = free_segs[0]
            seg0 = segment.segs[0]
            seg1 = segment.segs[1]
            cover = seg0[1] - seg0[0] + seg1[1] - seg1[0]
            c0 = cover / (root_seg[1] - root_seg[0])
            c1 = Segment.contain(center_seg, free_seg)
            if c0 > COLUMN_THRESHOLD and c1:
                # two columns
                columns.extend([seg0, seg1])

        groups = [blocks]
        if columns:
            groups = [[] for _ in columns]
            for b, r in zip(blocks, rects):
                column_ind = 0
                cand_seg = (r[0], r[2])
                for i, seg in enumerate(columns):
                    if Segment.contain(seg, cand_seg):
                        column_ind = i
                        break
                groups[i].append(b)

        return groups

    def _allocate_continuous(self, groups, lang):
        g_bound = []
        groups = [g for g in groups if g]
        for blocks in groups:
            arr = [b.bbox for b in blocks]
            bboxes = np.asarray(arr)
            g_bound.append(np.asarray(merge_rects(bboxes)))

        LINE_FULL_THRESHOLD = 0.80
        START_THRESHOLD = 0.8
        SIMI_HEIGHT_THRESHOLD = 0.3
        SIMI_WIDTH_THRESHOLD = 0.05

        TEXT_ID = 4
        TABLE_ID = 5

        def _get_elem(blocks, is_first=True):
            if not blocks:
                return (None, None, None, None, None)
            if is_first:
                b1 = blocks[0]
                b1_label = b1.layout_type
                if b1_label == TABLE_ID:
                    r1 = b1.bbox
                else:
                    r1 = b1.rs[0]

                r1_w = r1[2] - r1[0]
                r1_h = r1[3] - r1[1]
                return (b1, b1_label, r1, r1_w, r1_h)
            else:
                b0 = blocks[-1]
                b0_label = b0.layout_type
                if b0_label == TABLE_ID:
                    r0 = b0.bbox
                else:
                    r0 = b0.rs[-1]

                r0_w = r0[2] - r0[0]
                r0_h = r0[3] - r0[1]
                return (b0, b0_label, r0, r0_w, r0_h)

        b0, b0_label, r0, r0_w, r0_h = _get_elem(groups[0], False)
        g0 = g_bound[0]

        for i in range(1, len(groups)):
            b1, b1_label, r1, r1_w, r1_h = _get_elem(groups[i], True)
            g1 = g_bound[i]

            # print('\n_allocate_continuous:')
            # print(b0, b0_label, b1, b1_label)

            if b0_label and b0_label == b1_label and b0_label == TEXT_ID:
                c0 = r0_w / (g0[2] - g0[0])
                c1 = (r1[0] - g1[0]) / r1_h
                c2 = np.abs(r0_h - r1_h) / r1_h

                # print('\n\n---conti texts---')
                # print(b0_label, c0, c1, c2,
                #       b0, b0_label, r0, r0_w, r0_h,
                #       b1, b1_label, r1, r1_w, r1_h)

                if c0 > LINE_FULL_THRESHOLD and c1 < START_THRESHOLD and c2 < SIMI_HEIGHT_THRESHOLD:
                    new_text = join_lines([b0.block_text, b1.block_text], lang)
                    joined_lines = np.hstack([b0.ts, b1.ts])
                    joined_bboxes = np.vstack([b0.rs, b1.rs])
                    new_block = b1
                    new_block.block_text = new_text
                    new_block.ts = joined_lines
                    new_block.rs = joined_bboxes
                    groups[i][0] = new_block
                    groups[i - 1].pop(-1)
            elif self.is_join_table and b0_label and b0_label == b1_label and b0_label == TABLE_ID:
                row0 = b0.block_text.split("\n", 1)[0].split(" | ")
                row1 = b1.block_text.split("\n", 1)[0].split(" | ")

                c0 = (r1_w - r0_w) / r0_w
                c1 = len(row0) == len(row1)
                # print('---table join---', c0, c1, row0, row1, r1_w, r0_w)

                if c0 < SIMI_WIDTH_THRESHOLD and c1:
                    has_header = np.all([e0 == e1 for e0, e1 in zip(row0, row1)])
                    new_text = merge_md_tables([b0.block_text, b1.block_text], has_header)
                    new_html_text = merge_html_tables([b0.html_text, b1.html_text], has_header)
                    new_block = b1
                    new_block.block_text = new_text
                    new_block.html_text = new_html_text
                    groups[i][0] = new_block
                    groups[i - 1].pop(-1)

            b0, b0_label, r0, r0_w, r0_h = _get_elem(groups[i], False)

        return groups

    def _save_to_pages(self, groups, page_inds, lang):
        TITLE_ID = 3
        TEXT_ID = 4
        TABLE_ID = 5
        pages = []
        for idx, blocks in zip(page_inds, groups):
            page = Page(number=idx)
            for b in blocks:
                # bbox = [b[0], b[1], b[2], b[3]]
                # label, text = b[6], b[4]
                bbox = b.bbox
                label = b.layout_type
                text = b.block_text

                element = None
                extra_data = {"bboxes": [bbox]}

                if label == TABLE_ID:
                    # html = b[-1]
                    html = b.html_text
                    clean_html = clean_html_table(html)
                    extra_data.update({"types": ["table"], "pages": [idx]})
                    prev_ind = 0
                    s = prev_ind
                    e = prev_ind + len(text) - 1
                    indexes = [[s, e]]
                    extra_data.update({"indexes": indexes})
                    metadata = ElementMetadata(text_as_html=clean_html, extra_data=extra_data)
                    element = Table(text=text, metadata=metadata)
                else:
                    prev_ind = 0
                    line_bboxes = [b.tolist() for b in b.rs]
                    # lines = b[-1]
                    lines = b.ts
                    line_cnt = len(lines)
                    extra_data.update({"bboxes": line_bboxes})
                    if True or lang == "zh":  # for join test only
                        extra_data.update({"pages": [idx] * line_cnt})
                        line_chars_cnt = [len(line) for line in lines]
                        indexes = []
                        for cnt in line_chars_cnt:
                            s = prev_ind
                            e = prev_ind + cnt - 1
                            indexes.append([s, e])
                            prev_ind = e + 1
                        extra_data.update({"indexes": indexes})

                    if label == TITLE_ID:
                        extra_data.update({"types": ["title"] * line_cnt})
                        metadata = ElementMetadata(extra_data=extra_data)
                        element = Title(text=text, metadata=metadata)
                    elif label == TEXT_ID:
                        extra_data.update({"types": ["paragraph"] * line_cnt})
                        metadata = ElementMetadata(extra_data=extra_data)
                        element = Text(text=text, metadata=metadata)
                    else:
                        extra_data.update({"types": ["paragraph"] * line_cnt})
                        metadata = ElementMetadata(extra_data=extra_data)
                        element = NarrativeText(text=text, metadata=metadata)

                page.elements.append(element)
            pages.append(page)

        return pages

    def load(self) -> List[Page]:
        """Load given path as pages."""
        blob = Blob.from_path(self.file)
        start = self.start
        groups = []
        page_inds = []
        lang = None

        def _task(textpage_info, bytes_img, img, is_scan, lang, rot_matirx):
            b64_data = base64.b64encode(bytes_img).decode()
            layout_inp = {"b64_image": b64_data}
            layout = self.layout_agent.predict(layout_inp)
            blocks = self._allocate_semantic(
                textpage_info, layout, b64_data, img, is_scan, lang, rot_matrix
            )
            return blocks

        with blob.as_bytes_io() as file_path:
            fitz_doc = pymupdf.open(file_path)
            pdf_doc = pypdfium2.PdfDocument(file_path, autoclose=True)
            max_page = fitz_doc.page_count - start
            n = self.n if self.n else max_page
            n = min(n, max_page)

            sample_n = min(5, fitz_doc.page_count)
            type_texts = [page.get_text() for page in fitz_doc.pages(0, sample_n)]
            type_texts = "".join(type_texts)
            zh_n = len(re.findall(ZH_CHAR, type_texts))
            total_n = len(type_texts)

            is_scan = total_n < 10
            if not is_scan:
                lang = "zh" if zh_n > 200 or zh_n / total_n > 0.5 else "eng"
            else:
                lang = "zh"

            # is_scan = True

            timer = Timer()
            if self.verbose:
                print(f"{n} pages need be processed...")

            bytes_imgs = []
            page_imgs = []
            for idx in range(start, start + n):
                page = pdf_doc.get_page(idx)
                pil_image = page.render().to_pil()
                page_imgs.append(pil_image)
                img_byte_arr = io.BytesIO()
                pil_image.save(img_byte_arr, format="PNG")
                bytes_img = img_byte_arr.getvalue()
                bytes_imgs.append(bytes_img)

            timer.toc()
            print("pdfium render image", timer.get())

            results = []
            with ThreadPoolExecutor(max_workers=self.n_parallel) as executor:
                futures = []
                for idx in range(start, start + n):
                    # timer = Timer()
                    # Becareful: pymupdf doc load page in parallel will cause
                    # corrupted double-linked list, do not keep page object
                    textpage = fitz_doc.load_page(idx).get_textpage()
                    rot_matrix = None

                    # page = pdf_doc.get_page(idx)
                    # pil_image = page.render().to_pil()
                    # img_byte_arr = io.BytesIO()
                    # pil_image.save(img_byte_arr, format="PNG")
                    # bytes_img = img_byte_arr.getvalue()
                    # timer.toc()

                    bytes_img = bytes_imgs[idx - start]
                    img = page_imgs[idx - start]
                    # print('pil image png convert', timer.get())

                    if not is_scan:
                        textpage_info = self._extract_lines_v2(textpage)
                    else:
                        textpage_info = (None, None)

                    # blocks = _task(textpage_info, bytes_img, img, is_scan, lang, rot_matrix)

                    futures.append(
                        executor.submit(
                            _task, textpage_info, bytes_img, img, is_scan, lang, rot_matrix
                        )
                    )

                idx = start
                for future in futures:
                    blocks = future.result()
                    if not blocks:
                        continue

                    if self.with_columns:
                        sub_groups = self._divide_blocks_into_groups(blocks)
                        groups.extend(sub_groups)
                        for _ in sub_groups:
                            page_inds.append(idx + 1)
                    else:
                        groups.append(blocks)
                        page_inds.append(idx + 1)

                    idx += 1

        groups = self._allocate_continuous(groups, lang)
        pages = self._save_to_pages(groups, page_inds, lang)
        return pages

    @property
    def pages(self) -> List[Page]:
        """Gets all elements from pages in sequential order."""
        if self._pages is None:
            self._pages = self.load()

        return super().pages
