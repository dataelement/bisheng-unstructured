FROM dataelement/bisheng-unstructured:base.v1
LABEL org.opencontainers.image.authors="Dataelem inc."


WORKDIR /app
# Copy bin script
COPY ./docker/entrypoint.sh /app/

# Copy source code
COPY ./src/ /app/
COPY ./requirements.txt /app/

# install requirements
RUN pip install -r requirements.txt

CMD ["bash", "entrypoint.sh"]
