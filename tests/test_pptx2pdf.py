from bisheng_unstructured.topdf.pptx2pdf import PptxToPDF


def test1():
    input_file = "./examples/docs/毛泽东课件.pptx"
    output_file = "./data/毛泽东课件_v0.0.pdf"
    engine = PptxToPDF()
    engine.render(input_file, output_file)


def test2():
    input_file = "./examples/docs/毛泽东课件.ppt"
    output_file = "./data/毛泽东课件_v0.0.pdf"
    engine = PptxToPDF()
    engine.render(input_file, output_file)


test1()
test2()
