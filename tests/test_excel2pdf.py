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


def test3():
    input_file = "./examples/docs/DataelemPricingSheetAugust.2023.xlsx"
    output_file = "./data/DataelemPricingSheetAugust.2023.pdf"
    engine = ExcelToPDF()
    engine.render(input_file, output_file)


def test4():
    input_file = "./examples/docs/table_x1.tsv"
    output_file = "./data/table_x1.pdf"
    engine = ExcelToPDF()
    engine.render(input_file, output_file)


def test5():
    input_file = "./examples/docs/table_x2.csv"
    output_file = "./data/table_x2.pdf"
    engine = ExcelToPDF()
    engine.render(input_file, output_file)


# test1()
# test2()
test3()
test4()
test5()
