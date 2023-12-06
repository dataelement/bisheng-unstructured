#!/bin/bash


function temp_build_image() {
    image="dataelement/bisheng-unstructured:0.0.1"
    docker rmi $image
    docker commit -a "author@dataelem.com" -m "commit bisheng-unstructured image" bisheng-unstr-v001-dev $image
    #docker push $image
}


function create_dev_image() {
    image="dataelement/bisheng-unstructured:0.0.1"
    docker run -itd --name bisheng-uns-v002-dev -p 50001:10001 \
        -v /home/hanfeng:/home/hanfeng -v /home/public:/home/public \
        $image bash
}


function temp_build_image_v002() {
    image="dataelement/bisheng-unstructured:0.0.2"
    docker rmi $image
    docker commit -a "author@dataelem.com" -m "commit bisheng-unstructured image" bisheng-uns-v002-dev $image
    #docker push $image
}


function upload_image() {
    IMAGE_DIR="/home/public/bisheng-images"
    IMAGE_FILE="${IMAGE_DIR}/dataelement-bisheng-unstructured-v0.0.2.tar.gz"
    docker save dataelement/bisheng-unstructured:0.0.2 | gzip > $IMAGE_FILE
    upload-data $IMAGE_FILE
}


function run_dev_docker() {
  image="dataelement/bisheng-unstructured:0.0.2"
  cnt_name="bisheng_uns_v002_rd_dev"
  MOUNT="-v $HOME:$HOME -v /public:/public"
  pushd $(cd $(dirname $0); pwd)
  docker run -p 22001:10001 -itd --name ${cnt_name} $MOUNT $image bash
}


# create_dev_i
# mage
# temp_build_image
# temp_build_image_v002
# upload_image

run_dev_docker
