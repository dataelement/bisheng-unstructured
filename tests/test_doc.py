from bisheng_unstructured.documents.html_utils import save_to_txt, visualize_html
from bisheng_unstructured.partition.docx import partition_docx


def test_docx():
    filename = "./examples/docs/maoxuan_sample.docx"
    elements = partition_docx(filename=filename)

    output_file = "./data/maoxuan_sample_docx.html"
    output_file2 = "./data/maoxuan_sample_docx.txt"
    visualize_html(elements, output_file)
    save_to_txt(elements, output_file2)


def test_docx2():
    filename = "./examples/docs/handbook-1p.docx"
    elements = partition_docx(filename=filename)

    output_file = "./data/handbook-1p.html"
    visualize_html(elements, output_file)


def test_docx3():
    import docx

    filename = "./examples/docs/handbook-1p.docx"
    output = "./examples/docs/handbook-1p.pdf"

    # Open the .docs file
    doc = docx.Document(filename)
    # Save the file as pdf
    doc.save(output)


def test4():
    inp = "./examples/docs/handbook-1p.docx"
    outp = "./examples/docs/handbook-1p.pdf"

    import pypandoc

    pypandoc.convert_file(inp, "pdf", outputfile=outp)


def test5():
    inp = "./examples/docs/maoxuan_sample.docx"
    outp = "./data/maoxuan_sample.pdf"


test_docx()
# test_docx2()
# test_docx3()
# test4()
# test5()
