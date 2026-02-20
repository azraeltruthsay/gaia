FROM python:3.11-slim

WORKDIR /docs

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    mkdocs==1.6.0 \
    mkdocs-material==9.5.0 \
    mkdocs-awesome-pages-plugin==2.9.2

COPY gaia-wiki/mkdocs.yml .
COPY gaia-wiki/docs/ docs/

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=15s \
    CMD curl -f http://localhost:8080 || exit 1

CMD ["mkdocs", "serve", "--dev-addr=0.0.0.0:8080"]
