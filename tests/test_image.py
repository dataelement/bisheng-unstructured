import hashlib

from bisheng_unstructured.documents.html_utils import save_to_txt, visualize_html
from bisheng_unstructured.documents.pdf_parser.image import ImageDocument

TEST_RT_URL = "http://192.168.106.12:9001/v2.1/models/"


def test_image():
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
        "ocr_model_ep": f"{TEST_RT_URL}elem_ocr_collection_v3/infer",
    }

    filename = "examples/docs/maoxuan_intro_with_table.jpg"
    doc = ImageDocument(file=filename, model_params=model_params)
    pages = doc.pages
    elements = doc.elements

    visualize_html(elements, "data/maoxuan_intro_with_table2.html")
    save_to_txt(elements, "data/maoxuan_intro_with_table2.txt")


def test_image2():
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
        "ocr_model_ep": f"{TEST_RT_URL}elem_ocr_collection_v3/infer",
    }

    filename = "examples/docs/maoxuan_sample1.jpg"
    doc = ImageDocument(file=filename, model_params=model_params)
    pages = doc.pages
    elements = doc.elements

    visualize_html(elements, "data/maoxuan_sample2.html")
    save_to_txt(elements, "data/maoxuan_sample2.txt")


def test_image3():
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
        "ocr_model_ep": f"{TEST_RT_URL}elem_ocr_collection_v3/infer",
    }

    filename = "examples/docs/bmp图片.bmp"
    doc = ImageDocument(file=filename, model_params=model_params)
    pages = doc.pages
    elements = doc.elements

    visualize_html(elements, "data/bmp图片2.html")
    save_to_txt(elements, "data/bmp图片2.txt")


def test_regress():
    files = [
        ("data/bmp图片2.txt", "data/bmp图片.txt"),
        ("data/maoxuan_sample2.txt", "data/maoxuan_sample2.txt"),
        ("data/maoxuan_intro_with_table2.txt", "data/maoxuan_intro_with_table.txt"),
    ]

    for new_f, ori_f in files:
        s1 = hashlib.md5(open(new_f, "rb").read()).hexdigest()
        s2 = hashlib.md5(open(ori_f, "rb").read()).hexdigest()
        assert s1 == s2


test_image3()
# test_image2()
# test_image()
# test_regress()
