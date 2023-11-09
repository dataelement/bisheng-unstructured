import base64
import os

import requests


def test_part():
    uns_ep = os.environ.get("UNS_EP", "127.0.0.1:10001")

    url = f"http://{uns_ep}/v1/etl4llm/predict"
    filename = "../../examples/docs/达梦数据库招股说明书.pdf"
    b64_data = base64.b64encode(open(filename, "rb").read()).decode()
    inp = dict(
        filename=os.path.basename(filename),
        b64_data=[b64_data],
        mode="partition",
        parameters={"start": 0, "n": 5},
    )
    resp = requests.post(url, json=inp).json()
    assert resp["status_code"] == 200, resp
    # print(resp)


def test_any2pdf():
    uns_ep = os.environ.get("UNS_EP", "127.0.0.1:10001")
    url = f"http://{uns_ep}/v1/etl4llm/predict"

    filenames = [
        "../../examples/docs/maoxuan_wikipedia.html",
        "../../examples/docs/maoxuan_volumn_v1.txt",
        "../../examples/docs/test.md",
        "../../examples/docs/毛泽东课件.ppt",
        "../../examples/docs/毛泽东课件.pptx",
        "../../examples/docs/tests-example.xls",
        "../../examples/docs/tests-example.xlsx",
        "../../examples/docs/maoxuan_sample.docx",
        "../../examples/docs/maoxuan_sample.doc",
    ]
    for filename in filenames:
        b64_data = base64.b64encode(open(filename, "rb").read()).decode()
        inp = dict(
            filename=os.path.basename(filename),
            b64_data=[b64_data],
            mode="topdf",
        )
        resp = requests.post(url, json=inp).json()

        assert resp["status_code"] == 200, resp

        # out_fn = os.path.basename(filename).rsplit('.', 1)[0] + '.pdf'
        # out_file = os.path.join('./data', out_fn)
        # with open(ouf_file, 'wb') as fout:
        #     fout.write(base64.b64decode(resp['b64_pdf']))


# test_part()
test_any2pdf()
