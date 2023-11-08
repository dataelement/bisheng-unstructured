from bisheng_unstructured.topdf.text2pdf import Text2PDF


def test1():
    input_file = "./examples/docs/maoxuan_volumn_v1.txt"
    output_file = "./data/maoxuan_volumn_v1.pdf"
    engine = Text2PDF()
    engine.render(input_file, output_file)


def test2():
    input_file = "./examples/docs/maoxuan_wikipedia.html"
    output_file = "./data/maoxuan_wikipedia.pdf"
    engine = Text2PDF()
    engine.render(input_file, output_file)


def test3():
    input_file = "./examples/docs/test.md"
    output_file = "./data/test_md.pdf"
    engine = Text2PDF()
    engine.render(input_file, output_file)


test1()
# test2()
# test3()
