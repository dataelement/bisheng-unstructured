#!/bin/bash
  
uvicorn --host 0.0.0.0 --port 10001 bisheng_unstructured.api.main:app