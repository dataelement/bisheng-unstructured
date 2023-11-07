#!/bin/bash


function temp_release() {
  python3 setup.py install
  cp -fr config /opt/bisheng-unstructured/
  cp docker/entrypoint.sh /opt/bisheng-unstructured/bin/
  
  pkg_path="/usr/local/lib/python3.8/dist-packages/bisheng_unstructured-0.0.2-py3.8.egg"
  ens_words_pkg_path="${pkg_path}/bisheng_unstructured/nlp/english-words.txt"
  cp src/bisheng_unstructured/nlp/english-words.txt ${ens_words_pkg_path}
}


temp_release