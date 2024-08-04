# flake8: noqa
import hashlib
import os

from bisheng_unstructured.documents.html_utils import save_to_txt, visualize_html
from bisheng_unstructured.documents.pdf_parser.idp.image import ImageDocument
from bisheng_unstructured.documents.pdf_parser.idp.pdf import PDFDocument

RT_EP = os.environ.get("RT_EP", "192.168.106.12:9001")
TEST_RT_URL = f"http://{RT_EP}/v2.1/models/"


def test_pdf1():
    url = TEST_RT_URL
    layout_ep = url + "elem_layout_v1/infer"
    cell_model_ep = url + "elem_table_cell_detect_v1/infer"
    rowcol_model_ep = url + "elem_table_rowcol_detect_v1/infer"
    table_model_ep = url + "elem_table_detect_v1/infer"

    model_params = {
        "layout_ep": layout_ep,
        "cell_model_ep": cell_model_ep,
        "rowcol_model_ep": rowcol_model_ep,
        "table_model_ep": table_model_ep,
    }

    filename = "examples/docs/layout-parser-paper-fast.pdf"
    pdf_doc = PDFDocument(file=filename, model_params=model_params, n=2)
    pages = pdf_doc.pages
    elements = pdf_doc.elements
    for e in elements:
        print("e", e.to_dict())

    # visualize_html(elements, "data/layout-parser-paper-fast-v2.html")
    # save_to_txt(elements, "data/layout-parser-paper-fast-v2.txt")


def test_image1():
    url = TEST_RT_URL
    layout_ep = url + "elem_layout_v1/infer"
    cell_model_ep = url + "elem_table_cell_detect_v1/infer"
    rowcol_model_ep = url + "elem_table_rowcol_detect_v1/infer"
    table_model_ep = url + "elem_table_detect_v1/infer"

    model_params = {
        "layout_ep": layout_ep,
        "cell_model_ep": cell_model_ep,
        "rowcol_model_ep": rowcol_model_ep,
        "table_model_ep": table_model_ep,
    }

    filename = "examples/docs/maoxuan_intro_with_table.jpg"
    doc = ImageDocument(file=filename, model_params=model_params)
    pages = doc.pages
    elements = doc.elements
    for e in elements:
        print("e", e.to_dict())


# test_pdf1()
test_image1()
