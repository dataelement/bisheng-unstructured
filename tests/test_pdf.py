from unstructured.partition.pdf import partition_pdf


def test1():
  elements = partition_pdf(filename="example-docs/layout-parser-paper-fast.pdf")
  texts = "\n\n".join([str(el) for el in elements])
  with open('test_output/out1.txt', 'w') as fout:
     fout.write(texts)


def test2():
  elements = partition_pdf(
    filename="example-docs/layout-parser-paper.pdf",
    infer_table_structure=True,
    strategy="hi_res")


  cates = [el.category for el in elements]
  print(cates)


def test3():
  elements = partition_pdf(
    filename="example-docs/达梦数据库招股说明书_test_v1.pdf",
    # infer_table_structure=True,
    strategy="fast")

  cates = [el.category for el in elements]
  print(cates)

  content = []
  for el in elements:
    if el.category == 'Title':
      content.append(f'<h1>{el.text}</h1>')
    else:
      content.append(f'<p>{el.text}</p>')

  # print(content)
  with open('test_output/dameng_v1.html', 'w') as fout:
    fout.write('\n'.join(content))


# test1()
# test2()
test3()