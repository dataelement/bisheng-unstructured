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


def test_part2():
    filename = "./examples/docs/" "XXXX-需求-预警共享平台信息接入统一门户消息中心需求分析V1.0.docx"

    elements = partition_docx(filename=filename)

    output_file = "./data/需求-预警共享平台信息接入统一门户消息中心需求分析V1.0.html"
    output_file2 = "./data/需求-预警共享平台信息接入统一门户消息中心需求分析V1.0.txt"
    visualize_html(elements, output_file)
    save_to_txt(elements, output_file2)


def test_part3():
    fn = "XXXX-需求-贷后定期检视业务需求(含处置调查报告需求)-V0.52-20211129"
    filename = "./examples/docs/" + fn + ".docx"

    elements = partition_docx(filename=filename)

    output_file = "./data/" + fn + ".html"
    output_file2 = "./data/" + fn + ".txt"
    visualize_html(elements, output_file)
    save_to_txt(elements, output_file2)


# test_part2()
# test_part3()

# test_docx()
# test_docx2()
# test_docx3()
# test4()
# test5()
