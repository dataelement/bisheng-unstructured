#!/bin/bash

uvicorn --host 0.0.0.0 --port 10001 --workers 8 bisheng_unstructured.api.main:app
