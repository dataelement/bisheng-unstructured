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


def test_creator3():
    image_file = "./examples/docs/yyzz-sample.jpg"
    output_filename = "./data/yyzz-sample.pdf"

    model_params = {}
    pdf_creator = PdfCreator(model_params)
    pdf_creator.render(image_file, output_filename)


def test_creator4():
    image_file = "./examples/docs/危化品经营许可证_3.jpg"
    output_filename = "./data/危化品经营许可证_3.pdf"

    model_params = {}
    pdf_creator = PdfCreator(model_params)
    pdf_creator.render(image_file, output_filename)


def test_dpi():
    import fitz

    file_path = "./data/危化品经营许可证_3.pdf"
    fitz_doc = fitz.open(file_path)
    page = fitz_doc.load_page(0)
    mat = fitz.Matrix(1, 1)
    pm = page.get_pixmap(matrix=mat, alpha=False)
    bytes_img = pm.tobytes()
    with open("./data/危化品经营许可证_3.png", "wb") as fout:
        fout.write(bytes_img)

    print("---", pm.width, pm.height, pm.xres, pm.yres)
    # bytes_img = pm.getPNGData()


# test_creator()
# test_creator2()
# test_creator3()
test_creator4()
test_dpi()
