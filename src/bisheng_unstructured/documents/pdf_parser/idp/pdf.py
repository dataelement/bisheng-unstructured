# flake8: noqa
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
from bisheng_unstructured.models.idp.dummy_ocr_agent import OCRAgent

from ..blob import Blob

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
        self.ocr_agent = OCRAgent(**model_params)

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

    def _save_to_pages(self, groups, page_inds, lang):
        TITLE_ID = 3
        TEXT_ID = 4
        TABLE_ID = 5
        pages = []
        for idx, blocks in zip(page_inds, groups):
            page = Page(number=idx)
            for b in blocks:
                bbox = b.bbox
                label = b.layout_type
                text = b.block_text

                element = None
                extra_data = {"bboxes": [bbox]}

                if label == TABLE_ID:
                    # todo: update table html later
                    html = ""
                    clean_html = ""
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
                    line_bboxes = [b for b in b.rs]
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

        def _task(bytes_img, img, is_scan, lang, rot_matirx):
            b64_data = base64.b64encode(bytes_img).decode()
            payload = {"b64_image": b64_data}
            result = self.ocr_agent.predict(payload)
            return result

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
                    bytes_img = bytes_imgs[idx - start]
                    img = page_imgs[idx - start]
                    futures.append(
                        executor.submit(_task, bytes_img, img, is_scan, lang, rot_matrix)
                    )

                idx = start
                for future in futures:
                    blocks = future.result()
                    if not blocks:
                        continue

                    groups.append(blocks)
                    page_inds.append(idx + 1)
                    idx += 1

        pages = self._save_to_pages(groups, page_inds, lang)
        return pages

    @property
    def pages(self) -> List[Page]:
        """Gets all elements from pages in sequential order."""
        if self._pages is None:
            self._pages = self.load()

        return super().pages
