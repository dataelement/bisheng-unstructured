FROM dataelement/bisheng-unstructured-runtime:0.0.1
LABEL org.opencontainers.image.authors="Dataelem inc."

# unset proxy
ENV http_proxy=
ENV https_proxy=
ENV HTTP_PROXY=
ENV HTTPS_PROXY=

# Copy bins and configs
RUN mkdir -p /opt/bisheng-unstructured/bin
COPY ./docker/entrypoint.sh /opt/bisheng-unstructured/bin/
COPY config /opt/bisheng-unstructured/

# FIX the high risk scan
RUN rm -f /opt/texlive/2023/texmf-dist/scripts/tlcockpit/tlcockpit.jar

WORKDIR /opt/bisheng-unstructured

# Copy source code
COPY ./src/ /opt/bisheng-unstructured/
COPY ./requirements.txt /opt/bisheng-unstructured/

# install requirements
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt

RUN apt-get clean &&  rm -rf /var/lib/apt/lists/* && rm -rf /root/.cache/pip

CMD ["bash", "bin/entrypoint.sh"]
