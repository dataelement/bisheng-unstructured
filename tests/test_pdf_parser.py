import hashlib
import os

from bisheng_unstructured.documents.html_utils import save_to_txt, visualize_html
from bisheng_unstructured.documents.pdf_parser.pdf import PDFDocument

RT_EP = os.environ.get("RT_EP", "192.168.106.12:9001")
TEST_RT_URL = f"http://{RT_EP}/v2.1/models/"


def test_pdf_doc():
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

    visualize_html(elements, "data/layout-parser-paper-fast-v2.html")
    save_to_txt(elements, "data/layout-parser-paper-fast-v2.txt")


def test_pdf_doc2():
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

    filename = "examples/docs/layout-parser-paper.pdf"
    pdf_doc = PDFDocument(file=filename, model_params=model_params, start=0)
    pages = pdf_doc.pages
    elements = pdf_doc.elements
    visualize_html(elements, "data/layout-parser-paper-v2.html")
    save_to_txt(elements, "data/layout-parser-paper-v2.txt")


def test_pdf_doc3():
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

    filename = "examples/docs/sw-flp-1965-v1.pdf"
    pdf_doc = PDFDocument(file=filename, model_params=model_params, start=0, n=5)
    pages = pdf_doc.pages
    elements = pdf_doc.elements
    # visualize_html(elements, 'data/sw-flp-1965-v1.html')
    # save_to_txt(elements, 'data/sw-flp-1965-v1.txt')


def test_pdf_doc4():
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

    filename = "examples/docs/sw-flp-1965-v1.pdf"
    pdf_doc = PDFDocument(file=filename, model_params=model_params, enhance_table=False, start=0)
    pages = pdf_doc.pages
    elements = pdf_doc.elements
    visualize_html(elements, "data/sw-flp-1965-v1.1.html")
    save_to_txt(elements, "data/sw-flp-1965-v1.1.txt")


def test_pdf_doc5():
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

    filename = "examples/docs/layout-parser-paper.pdf"
    pdf_doc = PDFDocument(file=filename, model_params=model_params, start=0)
    pages = pdf_doc.pages
    elements = pdf_doc.elements
    visualize_html(elements, "data/layout-parser-paper-v1.1.html")
    save_to_txt(elements, "data/layout-parser-paper-v1.1.txt")


def test_pdf_doc6():
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

    filename = "examples/docs/达梦数据库招股说明书.pdf"
    pdf_doc = PDFDocument(file=filename, model_params=model_params, start=0, n=10)
    pages = pdf_doc.pages
    elements = pdf_doc.elements
    visualize_html(elements, "data/达梦数据库招股说明书-v1_1.html")
    save_to_txt(elements, "data/达梦数据库招股说明书-v1_1.txt")


def test_pdf_doc7():
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

    filename = "examples/docs/maoxuan_scan.pdf"
    pdf_doc = PDFDocument(
        file=filename, model_params=model_params, enhance_table=False, start=0, n=100
    )
    pages = pdf_doc.pages
    elements = pdf_doc.elements
    visualize_html(elements, "data/maoxuan_scan-v1_2.html")
    save_to_txt(elements, "data/maoxuan_scan-v1_2.txt")


def test_pdf_doc8():
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
    print("model_params", model_params)

    filename = "examples/docs/达梦数据库招股说明书.pdf"
    pdf_doc = PDFDocument(file=filename, model_params=model_params, start=0, n=30, verbose=True)
    pages = pdf_doc.pages
    elements = pdf_doc.elements
    # for e in elements:
    #     print("e", e.to_dict())

    visualize_html(elements, "data/达梦数据库招股说明书-v1_2.html")
    save_to_txt(elements, "data/达梦数据库招股说明书-v1_2.txt")


def test_pdf_doc9():
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
    # print("model_params", model_params)

    for _ in range(1):
        filename = "examples/docs/流动资金借款合同1.pdf"
        pdf_doc = PDFDocument(
            file=filename, model_params=model_params, start=0, n=10, verbose=True, n_parallel=5
        )
        pages = pdf_doc.pages
        elements = pdf_doc.elements
        # for e in elements:
        #     print("e", e.to_dict())
        visualize_html(elements, "data/流动资金借款合同1-1.html")
        save_to_txt(elements, "data/流动资金借款合同1-1.txt")


def test_regress():
    files = [
        ("data/流动资金借款合同1-2.txt", "data/流动资金借款合同1-1.txt"),
        ("data/达梦数据库招股说明书-v1_2.txt", "data/达梦数据库招股说明书-v1_1.txt"),
        ("data/maoxuan_scan-v1_2.txt", "data/maoxuan_scan-v1_1.txt"),
    ]

    for new_f, ori_f in files:
        s1 = hashlib.md5(open(new_f, "rb").read()).hexdigest()
        s2 = hashlib.md5(open(ori_f, "rb").read()).hexdigest()
        assert s1 == s2


def test_pdf_doc10():
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
    print("model_params", model_params)

    filename = "examples/docs/南陵电子2022.pdf"
    pdf_doc = PDFDocument(
        file=filename, model_params=model_params, start=0, verbose=True, n_parallel=10
    )
    pages = pdf_doc.pages
    elements = pdf_doc.elements
    visualize_html(elements, "data/南陵电子2022-2.html")
    save_to_txt(elements, "data/南陵电子2022-2.txt")


# test_pdf_doc()
# test_pdf_doc2()
# test_pdf_doc3()
# test_pdf_doc4()
# test_pdf_doc5()
# test_pdf_doc6()

# test_pdf_doc7()
# test_pdf_doc8()
# test_pdf_doc9()
# test_regress()

test_pdf_doc10()
