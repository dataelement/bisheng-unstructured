from bisheng_unstructured.partition.pdf import partition_pdf
from bisheng_unstructured.documents.html_utils import visualize_html, save_to_txt


def test1():
  elements = partition_pdf(
    filename="examples/docs/layout-parser-paper-fast.pdf",
    infer_table_structure=True)

  # for el in elements:
  #   if el.category == 'Title':
  #     print('Title', el.text)

  visualize_html(elements, 'data/layout-parser-paper-fast.html')
  save_to_txt(elements, 'data/layout-parser-paper-fast.txt')


def test2():
  elements = partition_pdf(
    filename="examples/docs/layout-parser-paper.pdf",
    infer_table_structure=True,
    strategy="hi_res")

  visualize_html(elements, 'data/layout-parser-paper.html')
  save_to_txt(elements, 'data/layout-parser-paper.txt')


def test3():
  elements = partition_pdf(
    filename="examples/docs/maoxuan_full.pdf",
    infer_table_structure=True,
    strategy="hi_res")

  visualize_html(elements, 'data/maoxuan_full.html')


def test4():
  filename = "examples/docs/sw-flp-1965-v1.pdf"
  elements = partition_pdf(
    filename=filename,
    infer_table_structure=True,
    strategy="hi_res")

  visualize_html(elements, 'data/sw-flp-1965-v1_ori.html')


# test1()
# test2()
# test3()
test4()