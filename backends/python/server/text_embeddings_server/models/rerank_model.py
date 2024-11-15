# @Time     : 2024/11/15 15:23
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
import torch
import mindietorch
from pathlib import Path
from typing import Type, List
from opentelemetry import trace
from loguru import logger

from text_embeddings_server.models import Model
from text_embeddings_server.models.types import PaddedBatch, Embedding, Prediction, TokenEmbedding

tracer = trace.get_tracer(__name__)


class RerankModel(Model):
    def __init__(self, model_path: Path, device: torch.device, dtype: torch.dtype):
        mindietorch.set_device(device.index)
        model = torch.jit.load(next(Path(model_path).rglob("*.pt"))).eval().to(device)
        super(RerankModel, self).__init__(model=model, dtype=dtype, device=device)

    @property
    def batch_type(self) -> Type[PaddedBatch]:
        return PaddedBatch

    @tracer.start_as_current_span("embed")
    def embed(self, batch: PaddedBatch) -> List[Embedding]:
        logger.warning("rerank model does not support embed function")
        return []

    @tracer.start_as_current_span("embed_all")
    def embed_all(self, batch: PaddedBatch) -> List[TokenEmbedding]:
        logger.warning("rerank model does not support embed_all function")
        return []

    @tracer.start_as_current_span("predict")
    def predict(self, batch: PaddedBatch) -> List[Prediction]:
        kwargs = {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}

        scores = self.model(kwargs["input_ids"].to(self.device), kwargs["attention_mask"].to(self.device))[0].tolist()
        return [
            Prediction(
                values=scores[i]
            )
            for i in range(len(batch))
        ]