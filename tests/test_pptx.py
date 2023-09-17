from bisheng_unstructured.documents.html_utils import save_to_txt, visualize_html
from bisheng_unstructured.partition.pptx import partition_pptx


def test1():
    filename = "./examples/docs/毛泽东课件.pptx"
    elements = partition_pptx(filename=filename)

    output_file = "./data/maozedong_kejian_v1.0.html"
    output_file2 = "./data/maozedong_kejian_v1.0.txt"
    visualize_html(elements, output_file)
    save_to_txt(elements, output_file2)


test1()
