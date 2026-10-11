# Build all three command packages as one installable course distribution.
FROM python:3.12.14-slim AS builder
WORKDIR /src
COPY requirements/build.txt requirements/build.txt
COPY vendor vendor
RUN pip install --no-cache-dir --require-hashes -r requirements/build.txt
COPY . .
RUN python -m build --wheel --no-isolation

FROM python:3.12.14-slim AS runtime
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends libexpat1 && rm -rf /var/lib/apt/lists/*
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}" \
    TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
WORKDIR /src
COPY requirements/runtime.txt requirements/runtime.txt
COPY vendor vendor
RUN python -m venv "$VIRTUAL_ENV" && pip install --no-cache-dir --require-hashes -r requirements/runtime.txt
COPY --from=builder /src/dist/*.whl /tmp/wheels/
RUN pip install --no-cache-dir --no-deps /tmp/wheels/*.whl && pip check && rm -rf /src /tmp/wheels
RUN groupadd -g 1001 eoap && useradd -u 1001 -g 1001 -m eoap && mkdir /work && chown eoap:eoap /work
USER 1001:1001
WORKDIR /work
ENTRYPOINT []
CMD ["tile-based-training", "--help"]
