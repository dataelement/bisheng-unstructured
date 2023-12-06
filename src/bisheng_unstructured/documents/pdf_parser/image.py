import base64
from typing import Any, Iterator, List, Mapping, Optional, Union

from bisheng_unstructured.documents.base import Page
from bisheng_unstructured.models import LayoutAgent, OCRAgent, TableAgent, TableDetAgent

from .blob import Blob
from .pdf import PDFDocument

# from bisheng_unstructured.common import Timer


class ImageDocument(PDFDocument):
    def __init__(
        self,
        file: str,
        model_params: dict,
        with_columns: bool = False,
        text_elem_sep: str = "\n",
        enhance_table: bool = True,
        keep_text_in_image: bool = True,
        lang: str = "zh",
        verbose: bool = False,
        n_parallel: int = 10,
        **kwargs
    ) -> None:
        self.layout_agent = LayoutAgent(**model_params)
        self.table_agent = TableAgent(**model_params)
        self.ocr_agent = OCRAgent(**model_params)
        self.table_det_agent = TableDetAgent(**model_params)

        self.with_columns = with_columns
        self.verbose = verbose
        self.text_elem_sep = text_elem_sep
        self.file = file
        self.enhance_table = enhance_table
        self.lang = lang

        self.is_scan = True
        self.support_rotate = False
        self.is_join_table = False
        self.keep_text_in_image = keep_text_in_image
        self.n_parallel = n_parallel

        super(PDFDocument, self).__init__()

    def load(self) -> List[Page]:
        """Load given path as pages."""
        # timer = Timer()
        blob = Blob.from_path(self.file)
        groups = []
        b64_data = base64.b64encode(blob.as_bytes()).decode()
        layout_inp = {"b64_image": b64_data}
        # timer.toc()

        layout = self.layout_agent.predict(layout_inp)
        # timer.toc()

        page_inds = []
        blocks = self._allocate_semantic(None, layout, b64_data, self.is_scan, self.lang)
        # timer.toc()

        if blocks:
            if self.with_columns:
                sub_groups = self._divide_blocks_into_groups(blocks)
                groups.extend(sub_groups)
                for _ in sub_groups:
                    page_inds.append(1)
            else:
                groups.append(blocks)
                page_inds.append(1)

        # timer.toc()
        groups = self._allocate_continuous(groups, self.lang)

        # timer.toc()
        pages = self._save_to_pages(groups, page_inds, self.lang)
        # timer.toc()
        # print('timers', timer.get())
        return pages
