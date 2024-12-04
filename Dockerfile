FROM rust:1.83.0 as router-builder
WORKDIR /

COPY backends backends
COPY core /core
COPY router router
COPY Cargo.toml ./
COPY Cargo.lock ./

ENV PATH="$PATH:~/.cargo/bin/"
ENV PATH="$PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/torch/bin:$PATH"

RUN apt update && apt install --no-install-recommends -y protobuf-compiler && \
    cargo install --path router -F python -F http --no-default-features

FROM ubuntu:22.04 as base

ARG RUST_VERSION=1.82.0
ARG PYTHON_VERSION=3.11.10
ARG PYTHON_BIG_VERSION=3.11
ARG ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest

ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/lib64/plugin/opskernel:${ASCEND_TOOLKIT_HOME}/lib64/plugin/nnengine:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/$(arch):${ASCEND_TOOLKIT_HOME}/tools/aml/lib64:${ASCEND_TOOLKIT_HOME}/tools/aml/lib64/plugin:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:/usr/local/python${PYTHON_VERSION}/lib \
    ASCEND_TOOLKIT_HOME=${ASCEND_TOOLKIT_HOME} \
    PYTHONPATH=${ASCEND_TOOLKIT_HOME}/python/site-packages:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe \
    PATH=${ASCEND_TOOLKIT_HOME}/bin:${ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin:${ASCEND_TOOLKIT_HOME}/tools/ccec_compiler/bin:/usr/local/python${PYTHON_VERSION}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    ASCEND_AICPU_PATH=${ASCEND_TOOLKIT_HOME} \
    ASCEND_OPP_PATH=${ASCEND_TOOLKIT_HOME}/opp \
    TOOLCHAIN_HOME=${ASCEND_TOOLKIT_HOME}/toolkit \
    ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME}

COPY backends /usr/local/text-embeddings-inference-npu-backends

RUN apt update && \
    apt install --no-install-recommends -y gcc make build-essential libssl-dev zlib1g-dev libbz2-dev libsqlite3-dev && \
    apt install --no-install-recommends -y wget curl xz-utils libffi-dev  pkg-config


RUN wget https://mirrors.aliyun.com/python-release/source/Python-${PYTHON_VERSION}.tar.xz --no-check-certificate && \
    tar -xf Python-${PYTHON_VERSION}.tar.xz && \
    cd Python-${PYTHON_VERSION} && ./configure --prefix=/usr/local/python${PYTHON_VERSION} --enable-optimizations --enable-shared && \
    make -j$(nproc) && make install && \
    ln -s /usr/local/python${PYTHON_VERSION}/bin/python3 /usr/local/python${PYTHON_VERSION}/bin/python && \
    ln -s /usr/local/python${PYTHON_VERSION}/bin/pip3 /usr/local/python${PYTHON_VERSION}/bin/pip && \
    pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ && \
    pip config set global.trusted-host mirrors.aliyun.com && \
    pip install attrs==23.2.0 numpy==1.26.4 decorator==5.1.1 sympy==1.12.1 psutil==6.0.0 torch==2.1.0 torch_npu==2.1.0.post8 && \
    pip install grpcio-tools==1.51.1 mypy-protobuf==3.4.0 'types-protobuf>=3.20.4'  && \
    cd /usr/local/text-embeddings-inference-npu-backends/python/server && make install && \
    apt clean && rm -rf /var/lib/apt/lists/* && \
    rm -rf /Python* && rm -rf ~/.cache/pip

COPY --from=router-builder /usr/local/cargo/bin/text-embeddings-router /usr/local/bin/text-embeddings-router

FROM base

ENV PORT=80 \
    TEI_NPU_DEVICE=0

EXPOSE 80

ENTRYPOINT ["text-embeddings-router"]
CMD ["--json-output"]