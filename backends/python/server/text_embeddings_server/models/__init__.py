import os
import torch
import mindietorch

from loguru import logger
from pathlib import Path
from typing import Optional
from transformers import AutoConfig
from transformers.models.bert import BertConfig

from text_embeddings_server.models.model import Model
from text_embeddings_server.models.default_model import DefaultModel
from text_embeddings_server.models.rerank_model import RerankModel

__all__ = ["Model"]

# Disable gradients
torch.set_grad_enabled(False)

FLASH_ATTENTION = True
try:
    from text_embeddings_server.models.flash_bert import FlashBert
except ImportError as e:
    logger.warning(f"Could not import Flash Attention enabled models: {e}")
    FLASH_ATTENTION = False

if FLASH_ATTENTION:
    __all__.append(FlashBert)


def get_model(model_path: Path, dtype: Optional[str]):
    if dtype == "float32":
        dtype = torch.float32
    elif dtype == "float16":
        dtype = torch.float16
    elif dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        raise RuntimeError(f"Unknown dtype {dtype}")

    deviceIdx = os.environ.get('TEI_NPU_DEVICE', '0')
    if deviceIdx != None and deviceIdx.isdigit() and int(deviceIdx) >= 0 and int(deviceIdx) <= 7:
        mindietorch.set_device(int(deviceIdx))
        device = torch.device(f"npu:{int(deviceIdx)}")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if config.architectures[0].endswith("Classification"):
        return RerankModel(model_path, device, dtype)
    else:
        if (
                config.model_type == "bert"
                and device.type == "cuda"
                and config.position_embedding_type == "absolute"
                and dtype in [torch.float16, torch.bfloat16]
                and FLASH_ATTENTION
        ):
            return FlashBert(model_path, device, dtype)
        else:
            return DefaultModel(model_path, device, dtype)
    raise NotImplementedError
