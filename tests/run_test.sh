#!/bin/bash


function run_server() {
  APP_PATH="/home/hanfeng/projects/bisheng-unstructured/src"
  cp -fr config /app/
  pushd /app
  PYTHONPATH=$APP_PATH uvicorn --host 0.0.0.0 --port 10001 --workers 8 bisheng_unstructured.api.main:app
}


function test() {
  time PYTHONPATH=./src python3 ./tests/test_pdf_parser.py
  time PYTHONPATH=./src python3 ./tests/test_docx2pdf.py
  time PYTHONPATH=./src python3 ./tests/test_excel2pdf.py
  time PYTHONPATH=./src python3 ./tests/test_pptx2pdf.py
  time PYTHONPATH=./src python3 ./tests/test_text2pdf.py
}


run_server