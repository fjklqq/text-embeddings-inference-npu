# Text Embeddings Inference for Ascend NPU

参考Ascend文档[《TEI框架接入MindIE Torch组件全量适配代码》](https://www.hiascend.com/document/detail/zh/mindie/10RC3/mindietorch/Torchdev/mindie_torch0171.html)在[huggingface/text-embeddings-inference v1.2.3](https://github.com/huggingface/text-embeddings-inference/tree/v1.2.3)版本上修改而来

## 安装
### 1. 安装rush
```bash
# 对于ARM 64位CPU为aarch64，对于X86 64位CPU可将下面指令的aarch64替换为x86_64
wget https://static.rust-lang.org/dist/rust-1.79.0-aarch64-unknown-linux-gnu.tar.gz --no-check-certificate
tar -xvf rust-1.79.0-aarch64-unknown-linux-gnu.tar.gz
cd rust-1.79.0-aarch64-unknown-linux-gnu
bash install.sh

sudo apt update
apt install pkg-config
```
### 2. 设置环境变量
首先在命令行运行Python，通过torch.__file__的路径确认protoc所在目录，以Python 3.10.2为例
```
Python 3.10.2 (main, Sep 23 2024, 10:52:24) [GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> torch.__file__
'/usr/local/python3.10.2/lib/python3.10/site-packages/torch/__init__.py'
```
控制台输出的__init__.py所在目录的子文件夹bin下即为protoc的放置路径。随后，将Cargo的可执行文件目录和protoc目录导出到$PATH（在进行下一步骤前该目录可能为空或不存在）：
```bash
# Cargo编译出的可执行文件目录
export PATH=$PATH:~/.cargo/bin/
# protoc所在目录 
export PATH=/usr/local/python3.10.2/lib/python3.10/site-packages/torch/bin:$PATH
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
安装 mind-cli 并创建touch 2.2.0虚拟环境
```bash
cd ./text-embeddings-inference-npu/mind-cli
make install-all
```
> 仅安装 mind-cli
> ```bash
> cd ./text-embeddings-inference-npu/mind-cli
> make install
> ```

## 使用
### 1. 使用mind-cli对文本嵌入模型进行追踪及编译优化
```bash
mind-cli trace-and-compile /home/data/models/bge-large-zh-v1.5
```
### 1. 使用mind-cli对重排序模型进行编译优化
```bash
source venv/bin/active
mind-cli trace /home/data/models/bge-reranker-large --rerank
deactivate
mind-cli compile /home/data/models/bge-reranker-large --rerank
```

### 2.设置TEI运行显卡编号 
```bash
export TEI_NPU_DEVICE=0
```

### 3. 启动服务
####  Embedding模型
```bash
text-embeddings-router \
--model-id /home/data/models/bge-large-zh-v1.5  \
--dtype float16 --pooling cls \
--max-concurrent-requests 2048 \
--max-batch-requests 2048 \
--max-batch-tokens 1100000 \
--max-client-batch-size 256 \
--port 8035
```
####  Reranker模型
```bash
text-embeddings-router \
--model-id /home/data/models/bge-reranker-large \
--dtype float16 \
--max-client-batch-size 192 \
--max-concurrent-requests 2048 \
--max-batch-tokens 163840 \
--max-batch-requests 128 \
--port 8036
```
### 4. 访问
#### Embed接口
```bash
curl 127.0.0.1:8035/embed \ 
-X POST \
-d '{"inputs":"What is Deep Learning?"}' \     
-H 'Content-Type: application/json'
```

#### Embed_all接口
```bash
curl 127.0.0.1:8035/embed \ 
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

