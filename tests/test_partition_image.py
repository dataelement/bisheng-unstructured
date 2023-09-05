from bisheng_unstructured.partition.image import partition_image
from bisheng_unstructured.documents.markdown import (
  transform_html_table_to_md,
  merge_html_tables)


def test1():
  filename = './examples/docs/maoxuan_sample1.jpg'
  elements = partition_image(filename=filename)

  texts = "\n\n".join([str(el) for el in elements])
  print(texts)

  # with open('test_output/out1.txt', 'w') as fout:
  #    fout.write(texts)


def test2():
  filename = './examples/docs/maoxuan_intro_with_table.jpg'
  elements = partition_image(filename=filename, infer_table_structure=True)
  print('\n\n---RESULT---')
  texts = "\n\n".join([str(el) for el in elements])
  print(texts)


def test3():
  filename = './examples/docs/layout-parser-paper-with-table.jpg'
  elements = partition_image(filename=filename, infer_table_structure=True)
  

  result = []
  for el in elements:
    if el.category == 'Table':
      r = transform_html_table_to_md(el.metadata.text_as_html)
      result.append(r['text'])
    else:
      result.append(el.text)

  print('\n\n'.join(result))


def test4():
  html_text = """
<table>
  <thead><th>Dataset</th><th></th><th>Base Model|Large Model|Notes</th>
  <th></th></thead>
  <tr><td>PubLayNet[38]</td><td>F/M</td><td>M</td>
      <td>Layouts of modern scientific documents</td></tr>
  <tr><td>PRImA [3]</td><td>M</td><td>:</td>
      <td>Layouts of scanned modern magaxines and sciertific reports</td></tr>
  <tr><td></td><td>AA</td><td></td>
      <td>Layouts of scanned US newspapers from the 20th century</td></tr>
  <tr><td>TableBank[18]</td><td></td><td></td>
      <td>Table region on modern scientific and business document</td></tr>
  <tr><td>HJDataset [31]</td><td>F/M</td><td></td>
      <td>Layouts of history Japanese documents</td></tr>
</table>
  """

  out = transform_html_table_to_md(html_text)
  print(out['text'], out['html'])


def test5():
  html_text1 = """
<table>
  <thead><th>Dataset</th><th></th><th>Base Model|Large Model|Notes</th>
  <th></th></thead>
  <tr><td>PubLayNet[38]</td><td>F/M</td><td>M</td>
      <td>Layouts of modern scientific documents</td></tr>
  <tr><td>PRImA [3]</td><td>M</td><td>:</td>
      <td>Layouts of scanned modern magaxines and sciertific reports</td></tr>
  <tr><td></td><td>AA</td><td></td>
      <td>Layouts of scanned US newspapers from the 20th century</td></tr>
  <tr><td>TableBank[18]</td><td></td><td></td>
      <td>Table region on modern scientific and business document</td></tr>
  <tr><td>HJDataset [31]</td><td>F/M</td><td></td>
      <td>Layouts of history Japanese documents</td></tr>
</table>
  """

  html_text2 = """
<table>
  <thead><th>Dataset</th><th></th><th>Base Model|Large Model|Notes</th>
  <th></th></thead>
  <tr><td>PubLayNet[38]</td><td>F/M</td><td>M</td>
      <td>Layouts of modern scientific documents</td></tr>
  <tr><td>PRImA [3]</td><td>M</td><td>:</td>
      <td>Layouts of scanned modern magaxines and sciertific reports</td></tr>
  <tr><td></td><td>AA</td><td></td>
      <td>Layouts of scanned US newspapers from the 20th century</td></tr>
  <tr><td>TableBank[18]</td><td></td><td></td>
      <td>Table region on modern scientific and business document</td></tr>
  <tr><td>HJDataset [31]</td><td>F/M</td><td></td>
      <td>Layouts of history Japanese documents</td></tr>
</table>
  """

  result = merge_html_tables([html_text1, html_text2], True)
  print(result)


# test1()
# test2()
# test3()
# test4()
test5()