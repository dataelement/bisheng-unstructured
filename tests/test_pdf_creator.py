from bisheng_unstructured.documents.pdf_parser.pdf_creator import PdfCreator


def test_creator():
    image_file = "./examples/docs/maoxuan_sample1.jpg"
    output_filename = "./data/maoxuan_sample1.pdf"

    model_params = {}
    pdf_creator = PdfCreator(model_params)
    pdf_creator.render(image_file, output_filename)


def test_creator2():
    image_file = "./examples/docs/autogen-sample1.jpg"
    output_filename = "./data/autogen-sample1.pdf"

    model_params = {}
    pdf_creator = PdfCreator(model_params)
    pdf_creator.render(image_file, output_filename)


test_creator()
test_creator2()
