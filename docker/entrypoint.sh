#!/bin/bash


unset http_proxy
unset https_proxy
unset HTTP_PROXY
unset HTTPS_PROXY

export PATH=/opt/texlive/2023/bin/x86_64-linux:$PATH
export MANPATH=/opt/texlive/2023/texmf-dist/doc/man:$MANPATH
export INFOPATH=/opt/texlive/2023/texmf-dist/doc/info:$INFOPATH
export PATH=/opt/pandoc/pandoc-3.1.9/bin:$PATH

uvicorn --host 0.0.0.0 --port 10001 --workers 8 bisheng_unstructured.api.main:app