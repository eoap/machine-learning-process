FROM python:3.12.14-slim
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends libexpat1 && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY vendor vendor
COPY requirements/training.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --require-hashes -r /tmp/requirements.txt
COPY training/make-ml-model /build/package
RUN pip install --no-cache-dir --no-deps /build/package && rm -rf /build
RUN groupadd -g 1001 eoap && useradd -u 1001 -g 1001 -m eoap && mkdir /work && chown eoap:eoap /work
USER 1001:1001
WORKDIR /work
ENV TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
ENTRYPOINT []
CMD ["tile-based-training", "--help"]
