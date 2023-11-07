import base64
import json
import os
import tempfile

import requests
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from .pipeline import Pipeline
from .types import ConfigInput, UnstructuredInput, UnstructuredOutput

# Fastapi App


def handle_http_exception(req: Request, exc: HTTPException) -> ORJSONResponse:
    msg = {"status_code": exc.status_code, "status_message": exc.detail}
    return ORJSONResponse(content=msg)


_EXCEPTION_HANDLERS = {HTTPException: handle_http_exception}


def create_app():
    """Create the FastAPI app and include the router."""

    app = FastAPI(
        default_response_class=ORJSONResponse,
        exception_handlers=_EXCEPTION_HANDLERS,
    )

    origins = [
        "*",
    ]

    @app.get("/health")
    def get_health():
        return {"status": "OK"}

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = create_app()

config_file = "./config/config.json"
pipeline = Pipeline(config_file)


@app.post("/v1/config/update")
async def update_config(inp: ConfigInput):
    pdf_model_params_temp = {
        "layout_ep": "http://{0}/v2.1/models/elem_layout_v1/infer",
        "cell_model_ep": ("http://{0}/v2.1/models/elem_table_cell_detect_v1/infer"),
        "rowcol_model_ep": ("http://{0}/v2.1/models/elem_table_rowcol_detect_v1/infer"),
        "table_model_ep": "http://{0}/v2.1/models/elem_table_detect_v1/infer",
        "ocr_model_ep": "http://{0}/v2.1/models/elem_ocr_collection_v3/infer",
    }

    if inp.rt_ep is not None:
        pdf_model_params = {}
        for k, v in pdf_model_params_temp.items():
            pdf_model_params[k] = v.format(inp.rt_ep)

        config_dict = {"pdf_model_params": pdf_model_params}
    else:
        config_dict = inp.dict()

    pipeline.update_config(config_dict)
    return {"status": "OK"}


@app.get("/v1/config")
async def config():
    return {"status": "OK", "config": pipeline.config}


@app.post("/v1/etl4llm/predict", response_model=UnstructuredOutput)
async def etl4_llm(inp: UnstructuredInput):
    filename = inp.filename
    b64_data = inp.b64_data
    file_type = filename.rsplit(".", 1)[1].lower()

    if not inp.b64_data and not inp.url:
        raise Exception("url or b64_data at least one must be given")

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, filename)
        if b64_data:
            with open(file_path, "wb") as fout:
                fout.write(base64.b64decode(b64_data[0]))
        else:
            headers = inp.parameters.get("headers", {})
            ssl_verify = inp.parameters.get("ssl_verify", True)
            response = requests.get(inp.url, headers=headers, verify=ssl_verify)
            if not response.ok:
                raise Exception(f"URL return an error: {response.status_code}")
            with open(file_path, "wb") as fout:
                fout.write(response.text)

        inp.file_path = file_path
        inp.file_type = file_type

        return pipeline.predict(inp)
