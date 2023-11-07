#!/bin/bash

pushd $(cd $(dirname $0); pwd)
export UNS_EP="192.168.106.12:10003"

curl -X POST http://${UNS_EP}/v1/config/update -H 'Content-Type: application/json' -d '{"rt_ep": "192.168.106.12:9005"}'
curl -X GET http://${UNS_EP}/v1/config

python3 test_etl4llm.py
popd
