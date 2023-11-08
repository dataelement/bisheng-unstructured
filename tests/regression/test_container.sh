#!/bin/bash


function test_cases() {
  export UNS_EP="$1"
  curl -X GET http://${UNS_EP}/health
  python3 test_config_update.py
}


function test_container() {
  image="dataelement/bisheng-unstructured:0.0.1"
  temp_ctn="bisheng_uns_v001_test"

  image="dataelement/bisheng-unstructured:0.0.2"
  temp_ctn="bisheng_uns_v002_test"

  pushd $(cd $(dirname $0); pwd)
  # docker run -p 10005:10001 -itd --workdir /opt/bisheng-unstructured --name ${temp_ctn} $image bash bin/entrypoint.sh
  UNS_EP="127.0.0.1:10005"

  sleep 5
  # test_cases $UNS_EP

  curl -X POST http://${UNS_EP}/v1/config/update -H 'Content-Type: application/json' -d '{"rt_ep": "192.168.106.12:9005"}'
  curl -X GET http://${UNS_EP}/v1/config
  UNS_EP=${UNS_EP} python3 test_etl4llm.py

  docker stop ${temp_ctn} && docker rm ${temp_ctn}
}


test_container