# Text Embeddings Inference for Ascend NPU

参考Ascend文档[《TEI框架接入MindIE Torch组件全量适配代码》](https://www.hiascend.com/document/detail/zh/mindie/10RC3/mindietorch/Torchdev/mindie_torch0171.html)在[huggingface/text-embeddings-inference v1.2.3](https://github.com/huggingface/text-embeddings-inference/tree/v1.2.3)版本上修改而来

**目前只在`Ascend 910B3`上进行过测试**

[![][github-release-shield]][github-release-link]
[![][docker-release-shield]][docker-release-link]

## Docker部署
[![][docker-size-shield]][docker-release-link]
[![][docker-pulls-shield]][docker-release-link]

```shell
model=BAAI/bge-large-en-v1.5

docker run -p 8080:80 \
--device=/dev/davinci0 \
--device=/dev/davinci_manager \
--device=/dev/devmm_svm \
--device=/dev/hisi_hdc \
-v /usr/local/Ascend:/usr/local/Ascend \
--env TEI_NPU_DEVICE=0 --pull always \
fjklqq/text-embeddings-inference:npu-1.2.3  --model-id $model
```

```bash
curl 127.0.0.1:8080/embed \
    -X POST \
    -d '{"inputs":"What is Deep Learning?"}' \
    -H 'Content-Type: application/json'
```


## 本地部署
### 1. 安装rust
```bash
# 对于ARM 64位CPU为aarch64，对于X86 64位CPU可将下面指令的aarch64替换为x86_64
wget https://static.rust-lang.org/dist/rust-1.82.0-aarch64-unknown-linux-gnu.tar.gz --no-check-certificate
tar -xvf rust-1.82.0-aarch64-unknown-linux-gnu.tar.gz
cd rust-1.82.0-aarch64-unknown-linux-gnu
bash install.sh

sudo apt update
apt install pkg-config
```
### 2. 设置环境变量
首先在命令行运行Python，通过torch.__file__的路径确认protoc所在目录，以Python 3.11.10为例
```
Python 3.11.10 (main, Nov  5 2024, 04:00:52) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.__file__
'/usr/local/python3.11.10/lib/python3.11/site-packages/torch/__init__.py'
```
控制台输出的__init__.py所在目录的子文件夹bin下即为protoc的放置路径。随后，将Cargo的可执行文件目录和protoc目录导出到$PATH（在进行下一步骤前该目录可能为空或不存在）：
```bash
# Cargo编译出的可执行文件目录
export PATH=$PATH:~/.cargo/bin/
# protoc所在目录 
export PATH=/usr/local/python3.11.10/lib/python3.11/site-packages/torch/bin:$PATH
```
### 3. 安装
安装text-embeddings-inference
```bash
cd ./text-embeddings-inference-npu
cargo install --path router -F python -F http --no-default-features
cp target/release/text-embeddings-router /usr/local/bin/text-embeddings-router
cd ./backends/python/server
make install
```

## 使用
### 1.设置TEI运行NPU编号 
```bash
export TEI_NPU_DEVICE=0
```

### 2. 启动服务
####  Embedding模型
```bash
text-embeddings-router \
--model-id /home/data/models/bge-large-zh-v1.5  \
--dtype float16 --pooling cls \
--max-concurrent-requests 2048 \
--max-batch-requests 2048 \
--max-batch-tokens 1100000 \
--max-client-batch-size 64 \
--port 8035
```
####  Reranker模型
```bash
text-embeddings-router \
--model-id /home/data/models/bge-reranker-large \
--dtype float16 \
--max-client-batch-size 64 \
--max-concurrent-requests 2048 \
--max-batch-tokens 163840 \
--max-batch-requests 128 \
--port 8036
```
### 3. 访问
#### Embed接口
```bash
curl 127.0.0.1:8035/embed \ 
-X POST \
-d '{"inputs":"What is Deep Learning?"}' \     
-H 'Content-Type: application/json'
```

#### Embed_all接口
```bash
curl 127.0.0.1:8035/embed_all \ 
-X POST \
-d '{"inputs":["What is Deep Learning?", "What is Deep Learning?"]}' \     
-H 'Content-Type: application/json'
```

#### Rerank接口
```bash
curl 127.0.0.1:8036/rerank \
-X POST \
-d '{"query":"What is Deep Learning?", "texts": ["Deep Learning is a sub-filed of Machin Learning.", "Deep learning is a country."]}' \
-H 'Content-Type: application/json'
```

[github-release-shield]: https://img.shields.io/github/v/release/fjklqq/text-embeddings-inference-npu?color=369eff&labelColor=black&logo=github&style=flat-square
[github-release-link]: https://github.com/fjklqq/text-embeddings-inference-npu/releases

[docker-release-link]: https://hub.docker.com/r/fjklqq/text-embeddings-inference
[docker-release-shield]: https://img.shields.io/docker/v/fjklqq/text-embeddings-inference?color=369eff&label=docker&labelColor=black&logo=docker&logoColor=white&style=flat-square
[docker-pulls-shield]: https://img.shields.io/docker/pulls/fjklqq/text-embeddings-inference?color=45cc11&labelColor=black&style=flat-square
[docker-size-shield]: https://img.shields.io/docker/image-size/fjklqq/text-embeddings-inference?color=369eff&labelColor=black&style=flat-square
