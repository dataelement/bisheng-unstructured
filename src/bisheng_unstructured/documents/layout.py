from __future__ import annotations

import os
import tempfile
from pathlib import PurePath
from typing import BinaryIO, Collection, List, Optional, Tuple, Union, cast

import numpy as np
from PIL import Image


class DocumentLayout:
    """Class for handling documents that are saved as .pdf files. For .pdf files, a
    document image analysis (DIA) model detects the layout of the page prior to extracting
    element."""

    def __init__(self, pages=None):
        self._pages = pages

    def __str__(self) -> str:
        return "\n\n".join([str(page) for page in self.pages])

    @property
    def pages(self) -> List[PageLayout]:
        """Gets all elements from pages in sequential order."""
        return self._pages

    @classmethod
    def from_pages(cls, pages: List[PageLayout]) -> DocumentLayout:
        """Generates a new instance of the class from a list of `PageLayouts`s"""
        doc_layout = cls()
        doc_layout._pages = pages
        return doc_layout

    @classmethod
    def from_file(
        cls,
        filename: str,
        detection_model: Optional[Any] = None,
        element_extraction_model: Optional[Any] = None,
        fixed_layouts: Optional[List[Optional[List[Any]]]] = None,
        ocr_strategy: str = "auto",
        ocr_languages: str = "eng",
        extract_tables: bool = False,
        pdf_image_dpi: int = 200,
    ) -> DocumentLayout:
        """Creates a DocumentLayout from a pdf file."""
        logger.info(f"Reading PDF for file: {filename} ...")
        pages: List[PageLayout] = []
        return cls.from_pages(pages)

    @classmethod
    def from_image_file(
        cls,
        filename: str,
        detection_model: Optional[Any] = None,
        element_extraction_model: Optional[Any] = None,
        ocr_strategy: str = "auto",
        ocr_languages: str = "eng",
        fixed_layout: Optional[List[Any]] = None,
        extract_tables: bool = False,
    ) -> DocumentLayout:
        """Creates a DocumentLayout from an image file."""
        logger.info(f"Reading image file: {filename} ...")
        return cls.from_pages([])


class PageLayout:
    """Class for an individual PDF page."""

    def __init__(
        self,
        number: int,
        image: Image.Image,
        layout: Optional[List[Any]],
        image_metadata: Optional[dict] = None,
        image_path: Optional[Union[str, PurePath]] = None,
        detection_model: Optional[Any] = None,
        element_extraction_model: Optional[Any] = None,
        ocr_strategy: str = "auto",
        ocr_languages: str = "eng",
        extract_tables: bool = False,
    ):
        self.elements: Collection[Any] = []

    def __str__(self) -> str:
        return "\n\n".join([str(element) for element in self.elements])

    def get_elements_using_image_extraction(
        self,
        inplace=True,
    ) -> Optional[List[Any]]:
        """Uses end-to-end text element extraction model to extract the elements on the page."""
        return []

    def get_elements_with_detection_model(self, inplace=True) -> Optional[List[Any]]:
        """Uses specified model to detect the elements on the page."""
        elements = []
        if inplace:
            self.elements = elements
            return None
        return elements

    def get_elements_from_layout(self, layout: List[Any]) -> List[Any]:
        """Uses the given Layout to separate the page text into elements, either extracting the
        text from the discovered layout blocks or from the image using OCR."""
        return []

    def _get_image_array(self) -> Union[np.ndarray, None]:
        """Converts the raw image into a numpy array."""
        if self.image_array is None:
            if self.image:
                self.image_array = np.array(self.image)
            else:
                image = Image.open(self.image_path)
                self.image_array = np.array(image)
        return self.image_array

    @classmethod
    def from_image(
        cls,
        image: Image.Image,
        image_path: Optional[Union[str, PurePath]],
        number: int = 1,
        detection_model: Optional[Any] = None,
        element_extraction_model: Optional[Any] = None,
        layout: Optional[List[Any]] = None,
        ocr_strategy: str = "auto",
        ocr_languages: str = "eng",
        extract_tables: bool = False,
        fixed_layout: Optional[List[Any]] = None,
    ):
        """Creates a PageLayout from an already-loaded PIL Image."""
        page = cls(
            number=number,
            image=image,
            layout=layout,
            detection_model=detection_model,
            element_extraction_model=element_extraction_model,
            ocr_strategy=ocr_strategy,
            ocr_languages=ocr_languages,
            extract_tables=extract_tables,
        )
        return page
