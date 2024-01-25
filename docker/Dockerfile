FROM dataelement/bisheng-unstructured-runtime:0.0.1
LABEL org.opencontainers.image.authors="Dataelem inc."

# unset proxy
ENV http_proxy=
ENV https_proxy=
ENV HTTP_PROXY=
ENV HTTPS_PROXY=

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
COPY ./src/ /opt/bisheng-unstructured/
COPY pyproject.toml /opt/bisheng-unstructured/

# FIX the high risk scan
RUN rm -f /opt/texlive/2023/texmf-dist/scripts/tlcockpit/tlcockpit.jar

WORKDIR /opt/bisheng-unstructured

# install requirements
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-ansi --without dev

RUN apt-get clean &&  rm -rf /var/lib/apt/lists/* && rm -rf /root/.cache/pip

CMD ["bash", "bin/entrypoint.sh"]
