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


def test4():
    input_file = "./examples/docs/4d9dce37d9c5d7542ca89bf8c1984e7f.md"
    output_file = "./data/4d9dce37d9c5d7542ca89bf8c1984e7f.pdf"
    engine = Text2PDF()
    engine.render(input_file, output_file)


test4()
# test1()
# test2()
# test3()
