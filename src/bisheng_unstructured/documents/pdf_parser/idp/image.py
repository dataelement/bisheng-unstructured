import base64
from typing import Any, Iterator, List, Mapping, Optional, Union

from bisheng_unstructured.documents.base import Page
from bisheng_unstructured.models.idp.dummy_ocr_agent import OCRAgent

from ..blob import Blob
from ..idp.pdf import PDFDocument

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
        self.ocr_agent = OCRAgent(**model_params)
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
        blob = Blob.from_path(self.file)
        groups = []
        b64_data = base64.b64encode(blob.as_bytes()).decode()
        payload = {"b64_image": b64_data}

        page_inds = []
        blocks = self.ocr_agent.predict(payload)
        if blocks:
            groups.append(blocks)
            page_inds.append(1)

        pages = self._save_to_pages(groups, page_inds, self.lang)
        return pages
