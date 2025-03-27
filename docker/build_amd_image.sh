#!/bin/bash
set -e  # set -o errexit  遇到非0状态码时退出shell, 管道命令只判断最后一个命令的状态码，如果需要管道命令里的所有命令都成功才继续，可以使用set -o pipefail
set -x  # set -o xtrace  打印执行的命令
set -u  # set -o nounset  使用未定义变量时退出shell
set -v # 打印shell接收的输入

# build amd base image
docker build -t dataelement/bisheng-unstructured:base.v1 --build-arg PANDOC_ARCH=amd64 -f docker/base.Dockerfile .
# build uns image
docker build -t dataelement/bisheng-unstructured:v1.0.0 -f docker/Dockerfile .
