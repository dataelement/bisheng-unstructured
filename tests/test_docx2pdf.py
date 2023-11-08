from bisheng_unstructured.partition.common import convert_office_doc
from bisheng_unstructured.topdf.docx2pdf import DocxToPDF, DocxToPDFV1


def test1():
    input_file = "./examples/docs/maoxuan_sample.docx"
    output_file = "./data/maoxuan_sample.pdf"
    engine = DocxToPDF()
    engine.render(input_file, output_file)


def test2():
    input_file = "./examples/docs/maoxuan_sample.doc"
    output_file = "./data/maoxuan_sample.docx"
    convert_office_doc(input_file, "./data/")


def test3():
    input_file = "./examples/docs/maoxuan_sample.doc"
    output_file = "./data/maoxuan_sample-v1.pdf"
    engine = DocxToPDF()
    engine.render(input_file, output_file)


def test4():
    input_file = "./examples/docs/maoxuan_sample.doc"
    output_file = "./data/maoxuan_sample-v1.pdf"
    engine = DocxToPDFV1()
    engine.render(input_file, output_file)


def test5():
    input_file = "./examples/docs/maoxuan_sample.docx"
    output_file = "./data/maoxuan_sample-v2.pdf"
    engine = DocxToPDFV1()
    engine.render(input_file, output_file)


def test6():
    input_file = "./examples/docs/UI自动化测试说明文档V1.0.docx"
    output_file = "./data/UI自动化测试说明文档V1.0.pdf"
    engine = DocxToPDFV1()
    engine.render(input_file, output_file)


def test7():
    # not supported
    input_file = "./examples/docs/UI自动化测试说明文档V1.0.docx"
    output_file = "./data/UI自动化测试说明文档V1.0-v0.pdf"
    engine = DocxToPDF()
    engine.render(input_file, output_file)


# test1()
# test2()
# test3()
# test4()
# test6()
# test7()
