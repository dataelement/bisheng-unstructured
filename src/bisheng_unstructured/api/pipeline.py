import json
from typing import Dict

from bisheng_unstructured.documents.html_utils import save_to_txt, visualize_html
from bisheng_unstructured.documents.pdf_parser.image import ImageDocument
from bisheng_unstructured.documents.pdf_parser.pdf import PDFDocument
from bisheng_unstructured.partition.csv import partition_csv
from bisheng_unstructured.partition.doc import partition_doc
from bisheng_unstructured.partition.docx import partition_docx
from bisheng_unstructured.partition.html import partition_html
from bisheng_unstructured.partition.md import partition_md
from bisheng_unstructured.partition.ppt import partition_ppt
from bisheng_unstructured.partition.pptx import partition_pptx
from bisheng_unstructured.partition.text import partition_text
from bisheng_unstructured.partition.tsv import partition_tsv
from bisheng_unstructured.partition.xlsx import partition_xlsx
from bisheng_unstructured.staging.base import convert_to_isd

from .any2pdf import Any2PdfCreator
from .types import UnstructuredInput, UnstructuredOutput


def partition_pdf(filename, model_params, **kwargs):
    doc = PDFDocument(file=filename, model_params=model_params, **kwargs)
    _ = doc.pages
    return doc.elements


def partition_image(filename, model_params, **kwargs):
    doc = ImageDocument(file=filename, model_params=model_params, **kwargs)
    _ = doc.pages
    return doc.elements


PARTITION_MAP = {
    "pdf": partition_pdf,
    "png": partition_image,
    "jpeg": partition_image,
    "jpg": partition_image,
    "tif": partition_image,
    "tiff": partition_image,
    "bmp": partition_image,
    "doc": partition_doc,
    "docx": partition_docx,
    "ppt": partition_ppt,
    "pptx": partition_pptx,
    "xlsx": partition_xlsx,
    "md": partition_md,
    "html": partition_html,
    "txt": partition_text,
    "csv": partition_csv,
    "tsv": partition_tsv,
}


class Pipeline(object):
    def __init__(self, config_file: str):
        self.config = json.load(open(config_file))
        self.pdf_model_params = self.config.get("pdf_model_params")
        topdf_model_params = self.config.get("topdf_model_params", {})
        self.pdf_creator = Any2PdfCreator(topdf_model_params)

    def update_config(self, config_dict):
        self.config = config_dict
        self.pdf_model_params = self.config.get("pdf_model_params")
        topdf_model_params = self.config.get("topdf_model_params", {})
        self.pdf_creator = Any2PdfCreator(topdf_model_params)

    def to_pdf(self, inp: UnstructuredInput) -> UnstructuredOutput:
        try:
            output = self.pdf_creator.run(inp.file_path, inp.file_type)
            result = UnstructuredOutput(b64_pdf=output)
            return result
        except Exception as e:
            return UnstructuredOutput(status_code=400, status_message=str(e))

    def predict(self, inp: UnstructuredInput) -> UnstructuredOutput:
        if inp.mode == "topdf":
            return self.to_pdf(inp)

        if inp.file_type not in PARTITION_MAP:
            raise Exception(f"file type[{inp.file_type}] not supported")
        filename = inp.file_path
        file_type = inp.file_type

        # part_params = inp.parameters
        part_inp = {"filename": filename, **inp.parameters}
        part_func = PARTITION_MAP.get(file_type)
        if part_func == partition_pdf or part_func == partition_image:
            part_inp.update({"model_params": self.pdf_model_params})
        try:
            elements = part_func(**part_inp)
            # for e in elements:
            #     print("e", e.to_dict())
            mode = inp.mode
            if mode == "partition":
                isd = convert_to_isd(elements)
                result = UnstructuredOutput(partitions=isd)
            elif mode == "text":
                text = save_to_txt(elements)
                result = UnstructuredOutput(text=text)
            elif mode == "vis":
                html_text = visualize_html(elements)
                result = UnstructuredOutput(html_text=html_text)

            return result
        except Exception as e:
            return UnstructuredOutput(status_code=400, status_message=str(e))
