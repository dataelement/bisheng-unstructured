#!/bin/bash

function deploy() {
  docker run -p 10001:10001 -itd --workdir /opt/bisheng-unstructured \
    --name bisheng_uns_v002_release dataelement/bisheng-unstructured:0.0.2 bash
}

function update() {
  pushd $(cd $(dirname $0); pwd)
  curl -v -X POST http://192.168.106.12:10001/v1/config/update \
        -H 'Content-Type: application/json' -d @deploy_model.json
  curl -X GET http://192.168.106.12:10001/v1/config
  popd
}

# deploy
update
