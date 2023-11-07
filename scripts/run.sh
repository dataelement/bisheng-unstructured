#!/bin/bash

function run() {
  PYTHONPATH=./src/ uvicorn --host 0.0.0.0 --port 10002 bisheng_unstructured.api.main:app
}


run