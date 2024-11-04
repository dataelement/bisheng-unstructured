#!/bin/bash


export PATH=/usr/local/texlive/2024/bin/aarch64-linux:$PATH
export MANPATH=/usr/local/texlive/2024/texmf-dist/doc/man:$MANPATH
export INFOPATH=/usr/local/texlive/2024/texmf-dist/doc/info:$INFOPATH

uvicorn --host 0.0.0.0 --port 10001 --workers 8 bisheng_unstructured.api.main:app
