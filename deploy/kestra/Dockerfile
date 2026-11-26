FROM kestra/kestra:v1.3.49-slim

USER root

# 04_postgres_taxi.yaml uses `wget -qO- ... | gunzip` inside a Process taskRunner,
# and 03_getting_started_data_pipeline.yaml uses `python3` inside a Process taskRunner.
RUN apt-get update \
    && apt-get install -y --no-install-recommends wget python3 python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

# Install plugins required by 04_postgres_taxi.yaml and 03_getting_started_data_pipeline.yaml.
RUN /app/kestra plugins install --locally \
    io.kestra.plugin:plugin-script-shell:1.3.1 \
    io.kestra.plugin:plugin-jdbc-postgres:1.3.3 \
    io.kestra.plugin:plugin-script-python:1.3.1 \
    io.kestra.plugin:plugin-jdbc-duckdb:1.3.3
