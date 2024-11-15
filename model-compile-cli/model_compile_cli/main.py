# @Time     : 2024/11/15 17:03
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
import os
from pathlib import Path
from typing import Annotated, Optional

import torch
import typer
import mindietorch
from mindietorch._compile_spec import SUPPORT_SOC_NAME
from rich.progress import Progress, SpinnerColumn, TextColumn
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel

app = typer.Typer()


@app.command()
def socs():
    typer.secho("Support SOC name:", bold=True)
    for name in SUPPORT_SOC_NAME:
        typer.echo(
            typer.style(' - ', fg=typer.colors.GREEN, bold=True) +
            typer.style('SOC ', bold=True) +
            typer.style(name,fg=typer.colors.GREEN)
        )



@app.command()
def trace(model_name_or_path: Annotated[str, typer.Argument(help='Model name or path')],
          soc: Annotated[str, typer.Argument(help='SOC version type')],
          is_rerank_model: Annotated[
              Optional[bool], typer.Option("--rerank/--embedding", help='Model is rerank model')] = False,
          output_path: Annotated[Optional[Path], typer.Option()] = None
          ):
    """
    Load the portal gun
    """

    MIN_SHAPE = (1, 1)
    MAX_SHAPE = (300, 512)
    BATCH_SIZE = 300

    typer.echo("Load model ...")
    sentences = ["This is a sentence." for _ in range(BATCH_SIZE)]

    # load model
    with torch.no_grad():
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if is_rerank_model:
            model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, torchscript=True)
        else:
            model = AutoModel.from_pretrained(model_name_or_path, torchscript=True)
        model.eval()

        inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)

        typer.echo("Tracking model ...")
        inputs['input_ids'] = inputs['input_ids'].to(torch.int32)
        inputs['attention_mask'] = inputs['attention_mask'].to(torch.int32)
        model = torch.jit.trace(model, [inputs['input_ids'], inputs['attention_mask']], strict=False)

        typer.echo("Compiling model ...")
        dynamic_inputs = []
        dynamic_inputs.append(
            mindietorch.Input(min_shape=MIN_SHAPE, max_shape=MAX_SHAPE, dtype=inputs['input_ids'].dtype))
        dynamic_inputs.append(
            mindietorch.Input(min_shape=MIN_SHAPE, max_shape=MAX_SHAPE, dtype=inputs['attention_mask'].dtype))

        typer.echo("Compiling model ...")
        model = mindietorch.compile(
            model,
            inputs=dynamic_inputs,
            precision_policy=mindietorch.PrecisionPolicy.FP32,
            truncate_long_and_double=True,
            require_full_compilation=False,
            allow_tensor_replace_int=False,
            min_block_size=3,
            torch_executed_ops=[],
            soc_version=soc,  # Ascend310xxx为昇腾AI处理器类型，根据服务器设备类型配置
            optimization_level=0
        )

        if output_path is None:
            output_path = 'compiled_model.pt'
        else:
            output_path = os.path.join(output_path, f'compiled_model.pt')

        typer.echo("Saving compiled model ...")
        model.save(output_path)
        typer.echo("Save compiled model to '" + typer.style(output_path, fg=typer.colors.BLUE) + "'")
        typer.echo(
            typer.style('WARN: 请将编译后的pt文件保存在模型权重的第一级子目录', fg=typer.colors.YELLOW, bold=True))
