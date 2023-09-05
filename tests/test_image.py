from bisheng_unstructured.documents.pdf_parser.image import ImageDocument
from bisheng_unstructured.documents.html_utils import visualize_html, save_to_txt

TEST_RT_URL = 'http://192.168.106.12:9001/v2.1/models/'


def test_image():
  url = TEST_RT_URL
  layout_ep = url + 'elem_layout_v1/infer'
  cell_model_ep = url + 'elem_table_cell_detect_v1/infer'
  rowcol_model_ep = url + 'elem_table_rowcol_detect_v1/infer'
  table_model_ep = url + 'elem_table_detect_v1/infer'

  model_params = {
    'layout_ep': layout_ep,
    'cell_model_ep': cell_model_ep,
    'rowcol_model_ep': rowcol_model_ep,
    'table_model_ep': table_model_ep,
  }

  filename = "examples/docs/maoxuan_intro_with_table.jpg"
  doc = ImageDocument(
    file=filename, 
    model_params=model_params)
  pages = doc.pages
  elements = doc.elements

  visualize_html(elements, 'data/maoxuan_intro_with_table.html')
  save_to_txt(elements, 'data/maoxuan_intro_with_table.txt')


def test_image2():
  url = TEST_RT_URL
  layout_ep = url + 'elem_layout_v1/infer'
  cell_model_ep = url + 'elem_table_cell_detect_v1/infer'
  rowcol_model_ep = url + 'elem_table_rowcol_detect_v1/infer'
  table_model_ep = url + 'elem_table_detect_v1/infer'

  model_params = {
    'layout_ep': layout_ep,
    'cell_model_ep': cell_model_ep,
    'rowcol_model_ep': rowcol_model_ep,
    'table_model_ep': table_model_ep,
  }

  filename = "examples/docs/maoxuan_sample1.jpg"
  doc = ImageDocument(
    file=filename, 
    model_params=model_params)
  pages = doc.pages
  elements = doc.elements

  visualize_html(elements, 'data/maoxuan_sample1.html')
  save_to_txt(elements, 'data/maoxuan_sample1.txt')


test_image2()
test_image()