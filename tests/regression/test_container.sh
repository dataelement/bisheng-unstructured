#!/bin/bash


function test_start() {
  image="dataelement/bisheng-unstructured:0.0.1"
  temp_ctn="bisheng_uns_v001_dev"
  docker run -p 10002:10001 -itd --workdir /opt/bisheng-unstructured --name ${temp_ctn} $image bash bin/entrypoint.sh
  sleep 2
  curl -X GET http://127.0.0.1:10002/health
  docker stop ${temp_ctn} && docker rm ${temp_ctn}
}


test_start