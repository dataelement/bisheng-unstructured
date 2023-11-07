#!/bin/bash


unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

uvicorn --host 0.0.0.0 --port 10001 bisheng_unstructured.api.main:app