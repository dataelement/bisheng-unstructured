from bisheng_unstructured.topdf.excel2pdf import ExcelToPDF


def test1():
    input_file = "./examples/docs/tests-example.xlsx"
    output_file = "./data/tests-example-v1.0.pdf"
    engine = ExcelToPDF()
    engine.render(input_file, output_file)


def test2():
    input_file = "./examples/docs/tests-example.xls"
    output_file = "./data/tests-example-v1.1.pdf"
    engine = ExcelToPDF()
    engine.render(input_file, output_file)


test1()
test2()
