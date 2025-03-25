FROM dataelement/bisheng-unstructured:base.v1
LABEL org.opencontainers.image.authors="Dataelem inc."


RUN sh -c 'echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse \n \
        deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse \n \
        deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse \n \
        deb http://security.ubuntu.com/ubuntu/ focal-security main restricted universe multiverse" > /etc/apt/sources.list'

WORKDIR /app
# Copy bins and configs
COPY ./docker/entrypoint.sh /app/
COPY config /opt/bisheng-unstructured/


# FIX the high risk scan
RUN rm -f /opt/texlive/2023/texmf-dist/scripts/tlcockpit/tlcockpit.jar

WORKDIR /opt/bisheng-unstructured

# Copy source code
COPY ./src/ /opt/bisheng-unstructured/
COPY ./requirements.txt /opt/bisheng-unstructured/
COPY ./docker/wkhtmltox_0.12.6-1.focal_amd64.deb /opt/bisheng-unstructured/
RUN apt update && apt install -y xfonts-utils xfonts-encodings xfonts-base xfonts-75dpi
RUN dpkg -i wkhtmltox_0.12.6-1.focal_amd64.deb

# install requirements
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt

RUN apt-get clean &&  rm -rf /var/lib/apt/lists/* && rm -rf /root/.cache/pip

CMD ["bash", "bin/entrypoint.sh"]
