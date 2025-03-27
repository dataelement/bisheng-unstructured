#!/bin/bash
set -e  # set -o errexit  ������0״̬��ʱ�˳�shell, �ܵ�����ֻ�ж����һ�������״̬�룬�����Ҫ�ܵ����������������ɹ��ż���������ʹ��set -o pipefail
set -x  # set -o xtrace  ��ӡִ�е�����
set -u  # set -o nounset  ʹ��δ�������ʱ�˳�shell
set -v # ��ӡshell���յ�����

# build amd base image
docker build -t dataelement/bisheng-unstructured:base.v1 --build-arg PANDOC_ARCH=amd64 -f docker/base.Dockerfile .
# build uns image
docker build -t dataelement/bisheng-unstructured:v1.0.0 -f docker/Dockerfile .
