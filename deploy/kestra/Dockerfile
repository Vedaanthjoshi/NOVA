FROM kestra/kestra:v1.3.49-slim

USER root

# 04_postgres_taxi.yaml uses `wget -qO- ... | gunzip` inside a Process taskRunner.
RUN apt-get update \
    && apt-get install -y --no-install-recommends wget \
    && rm -rf /var/lib/apt/lists/*

# Install only the two plugins required by 04_postgres_taxi.yaml.
RUN /app/kestra plugins install --locally \
    io.kestra.plugin:plugin-script-shell:1.3.1 \
    io.kestra.plugin:plugin-jdbc-postgres:1.3.3
