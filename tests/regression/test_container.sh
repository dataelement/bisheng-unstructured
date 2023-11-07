#!/bin/bash


function test_cases() {
  export UNS_EP="$1"
  curl -X GET ${UNS_EP}/health
  python3 test_config_update.py
}


function test_container() {
  image="dataelement/bisheng-unstructured:0.0.1"
  temp_ctn="bisheng_uns_v001_test"

  pushd $(cd $(dirname $0); pwd)
  docker run -p 10002:10001 -itd --workdir /opt/bisheng-unstructured --name ${temp_ctn} $image bash bin/entrypoint.sh
  UNS_EP="http://127.0.0.1:10002"

  sleep 5
  test_cases $UNS_EP

  docker stop ${temp_ctn} && docker rm ${temp_ctn}
}


test_container