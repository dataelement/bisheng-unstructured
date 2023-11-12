import base64

from bisheng_unstructured.documents.pdf_parser.pdf_creator import PdfCreator
from bisheng_unstructured.topdf import DocxToPDFV1, ExcelToPDF, PptxToPDF, Text2PDF


class Any2PdfCreator(object):
    def __init__(self, kwargs):
        self.type_to_creator_map = {
            "png": PdfCreator,
            "jpeg": PdfCreator,
            "jpg": PdfCreator,
            "tiff": PdfCreator,
            "tif": PdfCreator,
            "bmp": PdfCreator,
            "doc": DocxToPDFV1,
            "docx": DocxToPDFV1,
            "ppt": PptxToPDF,
            "pptx": PptxToPDF,
            "xlsx": ExcelToPDF,
            "xls": ExcelToPDF,
            "txt": Text2PDF,
            "md": Text2PDF,
            "html": Text2PDF,
        }
        self.model_params = kwargs

    def run(self, file_path, file_type) -> str:
        if file_type not in self.type_to_creator_map:
            raise Exception(f"{file_type} is not supported")
        CreatorCls = self.type_to_creator_map.get(file_type)
        creator = CreatorCls(self.model_params)
        content_bytes = creator.render(file_path, to_bytes=True)
        return base64.b64encode(content_bytes).decode()
