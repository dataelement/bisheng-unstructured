#!/bin/bash

function start_docker() {
  docker run --gpus=all --shm-size 2g --net=host -itd --name bisheng_unstr_dev1 \
   -v /home/hanfeng:/home/hanfeng -v /home/public:/home/public ubuntu:20.04 bash
}

function prepare_env() {
  # Install Basic Dependences
  export DEBIAN_FRONTEND=noninteractive
  apt update && apt install -y nasm zlib1g-dev libssl-dev libre2-dev libb64-dev locales libsm6 libxext6 libxrender-dev libgl1 python3-dev python3-pip git

  # Configure language
  locale-gen en_US.UTF-8
  export LC_ALL=en_US.UTF-8
  export LANG=en_US.UTF-8
  export LANGUAGE=en_US.UTF-8

  # Configure timezone
  export TZ=Asia/Shanghai
  ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
}

function install_deps() {
  # apt install -y python3-dev python3-pip git
  # pip3 install git+https://github.com/pypdfium2-team/ctypesgen@pypdfium2
  pip3 install -r requirements.txt -i https://mirrors.tencent.com/pypi/simple
  python3 -c "import nltk; nltk.download('punkt')" && \
    python3 -c "import nltk; nltk.download('averaged_perceptron_tagger')"

}

prepare_env
# install_deps


