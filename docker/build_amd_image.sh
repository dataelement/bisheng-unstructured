#!/bin/bash

# build amd base image
docker build -t dataelement/bisheng-unstructured:base.v1 --build-arg PANDOC_ARCH=amd64 -f docker/base.Dockerfile .
# build uns image
docker build -t dataelement/bisheng-unstructured:v1.0.0 -f docker/Dockerfile .
