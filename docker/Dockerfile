FROM dataelement/bisheng-unstructured-runtime:0.0.1
LABEL org.opencontainers.image.authors="Dataelem inc."

ARG BISHENG_UNS_VER=0.0.2

# unset proxy
ENV http_proxy=
ENV https_proxy=
ENV HTTP_PROXY=
ENV HTTPS_PROXY=


RUN sh -c 'echo "deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal main restricted universe multiverse \
        deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-updates main restricted universe multiverse \
        deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ focal-backports main restricted universe multiverse \
        deb http://security.ubuntu.com/ubuntu/ focal-security main restricted universe multiverse" > /etc/apt/sources.list'

# Install Poetry
RUN apt-get update && apt-get install gcc g++ curl build-essential postgresql-server-dev-all -y
RUN apt-get update && apt-get install procps -y
# opencv
RUN apt-get install -y libglib2.0-0 libsm6 libxrender1 libxext6 libgl1
RUN curl -sSL https://install.python-poetry.org | python3 -
# # Add Poetry to PATH
ENV PATH="${PATH}:/root/.local/bin"

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
COPY ./docker/wkhtmltox_0.12.6-1.focal_amd64.deb /opt/bisheng-unstructured/
RUN apt update && apt install -y xfonts-utils xfonts-encodings xfonts-base xfonts-75dpi
RUN dpkg -i wkhtmltox_0.12.6-1.focal_amd64.deb

# install requirements
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt

RUN apt-get clean &&  rm -rf /var/lib/apt/lists/* && rm -rf /root/.cache/pip

CMD ["bash", "bin/entrypoint.sh"]
