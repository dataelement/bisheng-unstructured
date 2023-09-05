FROM ubuntu:20.04
MAINTAINER "dataelem inc."

# 安装系统库依赖
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y nasm zlib1g-dev libssl-dev libre2-dev libb64-dev locales libsm6 libxext6 libxrender-dev libgl1 python3-dev python3-pip git

# Configure language
RUN locale-gen en_US.UTF-8
ENV LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

# Configure timezone
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

