import hashlib
import os

from bisheng_unstructured.documents.html_utils import save_to_txt, visualize_html
from bisheng_unstructured.documents.pdf_parser.pdf import PDFDocument

RT_EP = os.environ.get("RT_EP", "192.168.106.12:9001")
TEST_RT_URL = f"http://{RT_EP}/v2.1/models/"


def test_formula1():
    url = TEST_RT_URL
    layout_ep = url + "elem_layout_v1/infer"
    cell_model_ep = url + "elem_table_cell_detect_v1/infer"
    rowcol_model_ep = url + "elem_table_rowcol_detect_v1/infer"
    table_model_ep = url + "elem_table_detect_v1/infer"
    formula_det_model_ep = url + "formula_det_v1/infer"
    formula_recog_model_ep = url + "formula_recog_v1/infer"

    model_params = {
        "layout_ep": layout_ep,
        "cell_model_ep": cell_model_ep,
        "rowcol_model_ep": rowcol_model_ep,
        "table_model_ep": table_model_ep,
        "ocr_model_ep": f"{TEST_RT_URL}elem_ocr_collection_v3/infer",
        "formula_recog_model_ep": formula_recog_model_ep,
        "formula_det_model_ep": formula_det_model_ep,
    }
    print("model_params", model_params)

    filename = "examples/docs/baby_rmt.pdf"
    pdf_doc = PDFDocument(
        file=filename,
        model_params=model_params,
        start=6,
        n=1,
        verbose=True,
        n_parallel=1,
        support_formula=True,
    )
    pages = pdf_doc.pages
    elements = pdf_doc.elements
    # for e in elements:
    #     print("---", e)

    visualize_html(elements, "data/baby_rmt.html")
    save_to_txt(elements, "data/baby_rmt.txt")


test_formula1()
