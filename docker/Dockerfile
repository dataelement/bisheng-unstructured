FROM dataelement/bisheng-unstructured-runtime:0.0.1
MAINTAINER "Dataelem inc."

ARG BISHENG_UNS_VER=0.0.2

RUN pip install bisheng-unstructured==${BISHENG_UNS_VER}

# Copy bins and configs
RUN mkdir -p /opt/bisheng-unstructured/bin
COPY ./docker/entrypoint.sh /opt/bisheng-unstructured/bin/
COPY config /opt/bisheng-unstructured/

WORKDIR /opt/bisheng-unstructured

RUN apt-get clean &&  rm -rf /var/lib/apt/lists/* && rm -rf /root/.cache/pip

CMD ["bash", "bin/entrypoint.sh"]
