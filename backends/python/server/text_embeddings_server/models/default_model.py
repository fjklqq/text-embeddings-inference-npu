import inspect
import torch
import mindietorch

from pathlib import Path
from typing import Type, List
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel, AutoConfig
from loguru import logger
from opentelemetry import trace

from text_embeddings_server.models import Model
from text_embeddings_server.models.types import PaddedBatch, Embedding, Prediction, TokenEmbedding

tracer = trace.get_tracer(__name__)


class DefaultModel(Model):
    def __init__(self, model_path: Path, device: torch.device, dtype: torch.dtype):
        mindietorch.set_device(device.index)
        model = torch.jit.load(next(Path(model_path).rglob("*.pt"))).eval().to(device)
        self.model_path = str(model_path)
        self.hidden_size = AutoConfig.from_pretrained(model_path, trust_remote_code=True).hidden_size
        super(DefaultModel, self).__init__(model=model, dtype=dtype, device=device)

    @property
    def batch_type(self) -> Type[PaddedBatch]:
        return PaddedBatch

    @tracer.start_as_current_span("embed")
    def embed(self, batch: PaddedBatch) -> List[Embedding]:
        kwargs = {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}
        output = self.model(kwargs["input_ids"].to(self.device), kwargs["attention_mask"].to(self.device))
        if isinstance(output, dict):
            embedding = output['last_hidden_state'].to('cpu')
        else:
            embedding = output[0].to('cpu')
        embedding = embedding[:, 0].contiguous()

        cpu_results = embedding.view(-1).tolist()

        return [
            Embedding(
                values=cpu_results[i * self.hidden_size : (i + 1) * self.hidden_size]
            )
            for i in range(len(batch))
        ]

    @tracer.start_as_current_span("embed_all")
    def embed_all(self, batch: PaddedBatch):
        kwargs = {"input_ids": batch.input_ids, "attention_mask": batch.attention_mask}
        output = self.model(kwargs["input_ids"].to(self.device), kwargs["attention_mask"].to(self.device))
        if isinstance(output, dict):
            embedding = output['last_hidden_state'].to('cpu').contiguous()
        else:
            embedding = output[0].to('cpu').contiguous()
        cpu_results = embedding.view(-1).tolist()

        embedding_result=[]
        for i in range(len(batch)):
            embedding_tmp=[
                Embedding(values=cpu_results[(j+i*batch.max_length) * self.hidden_size :
                                             (j + 1 + i*batch.max_length) * self.hidden_size])
                for j in range(batch.input_ids.size()[1])
            ]
            token_embeddings=TokenEmbedding(embeddings=embedding_tmp)
            embedding_result.append(token_embeddings)

        return embedding_result

    @tracer.start_as_current_span("predict")
    def predict(self, batch: PaddedBatch) -> List[Prediction]:
        logger.warning("embedding model does not support predict function")
        return []