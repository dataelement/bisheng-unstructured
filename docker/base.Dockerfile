FROM ubuntu:20.04

ARG PANDOC_ARCH=amd64
# 设置环境变量以避免交互式安装时的提示
ENV DEBIAN_FRONTEND=noninteractive
ENV PANDOC_ARCH=$PANDOC_ARCH

RUN apt-get update && apt-get install -y \
    fontconfig \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    libreoffice \
    curl \
    wget \
    unzip \
    locales \
    tzdata \
    xfonts-utils \
    xfonts-encodings \
    xfonts-base \
    xfonts-75dpi \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update

WORKDIR /opt

# Install fonts
RUN wget -d --header="Referer: https://www.alibabafonts.com" https://fonts.alibabadesign.com/AlibabaPuHuiTi-3.zip \
    && unzip AlibabaPuHuiTi-3.zip \
    && mkdir -p /usr/share/fonts/truetype/alibaba \
    && find AlibabaPuHuiTi-3 -type f|grep ttf | xargs -I {} mv {} /usr/share/fonts/truetype/alibaba/ \
    && fc-cache -f -v

# Install pandoc
RUN mkdir -p /opt/pandoc \
    && cd /opt/pandoc \
    && wget https://github.com/jgm/pandoc/releases/download/3.1.9/pandoc-3.1.9-linux-${PANDOC_ARCH}.tar.gz \
    && tar xvf pandoc-3.1.9-linux-${PANDOC_ARCH}.tar.gz \
    && cd pandoc-3.1.9 \
    && cp bin/pandoc /usr/bin/ \
    && cd ..

COPY ./docker/unnested-table.lua /opt/pandoc/

# Install wkhtmltopdf
RUN wget https://github.com/wkhtmltopdf/packaging/releases/download/0.12.6-1/wkhtmltox_0.12.6-1.focal_${PANDOC_ARCH}.deb && \
    dpkg -i wkhtmltox_0.12.6-1.focal_${PANDOC_ARCH}.deb

# Install Python3.10
RUN apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 创建符号链接以使 python3 指向 python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# 更新 pip 并检查 Python 和 pip 版本
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 \
    && python3 --version \
    && pip3 --version \

CMD ["python3"]
