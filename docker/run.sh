#!/bin/bash


function temp_build_image() {
    image="dataelement/bisheng-unstructured:0.0.1"
    docker rmi $image
    docker commit -a "author@dataelem.com" -m "commit bisheng-unstructured image" bisheng-unstr-v001-dev $image
    #docker push $image
}


temp_build_image