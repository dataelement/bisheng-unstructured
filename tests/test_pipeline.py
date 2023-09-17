from bisheng_unstructured.api.pipeline import Pipeline
from bisheng_unstructured.api.types import UnstructuredInput, UnstructuredOutput

CONFIG = "./config/config.json"


def test1():
    file_path = "examples/docs/maoxuan_intro_with_table.jpg"
    inp = UnstructuredInput(filename="", file_path=file_path, file_type="jpg", mode="text")
    pipeline = Pipeline(CONFIG)
    outp = pipeline.predict(inp)
    print(outp.dict())


def test2():
    file_path = "examples/docs/maoxuan_scan.pdf"
    inp = UnstructuredInput(
        filename="", file_path=file_path, file_type="pdf", mode="text", parameters={"n": 20}
    )

    pipeline = Pipeline(CONFIG)
    outp = pipeline.predict(inp)
    print(outp.dict())


def test3():
    file_path = "./examples/docs/毛泽东课件.pptx"
    inp = UnstructuredInput(filename="", file_path=file_path, file_type="pptx", mode="text")

    pipeline = Pipeline(CONFIG)
    outp = pipeline.predict(inp)
    print(outp.dict())


# test1()
# test2()
test3()
