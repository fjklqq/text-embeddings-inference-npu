# @Time     : 2024/11/15 17:03
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
import os
from typing import Annotated, Optional, Tuple

import torch

import typer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModel,AutoConfig
from transformers.utils import  is_torch_npu_available
from .import_utils import is_mindietorch_available,print_info

if is_torch_npu_available():
    import torch_npu

if is_mindietorch_available():
    import mindietorch
    from mindietorch._compile_spec import SUPPORT_SOC_NAME


app = typer.Typer()


@app.command()
def socs():
    typer.secho("Support SOC name:", bold=True)
    for name in SUPPORT_SOC_NAME:
        typer.echo(
            typer.style(' - ', fg=typer.colors.GREEN, bold=True) +
            typer.style('SOC ', bold=True) +
            typer.style(name, fg=typer.colors.GREEN)
        )


def gen_rerank_inputs(model_name_or_path):
    pairs = [
        ['what is panda?', 'hi'],
        ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), '
                           'sometimes called a panda bear or simply panda, '
                           'is a bear species endemic to China.']
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    inputs['input_ids'] = inputs['input_ids'].to(torch.int32)
    inputs['attention_mask'] = inputs['attention_mask'].to(torch.int32)
    return inputs


def gen_embedding_inputs(model_name_or_path, batch_size):
    sentences = ["This is a sentence." for _ in range(batch_size)]
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)
    inputs['input_ids'] = inputs['input_ids'].to(torch.int32)
    inputs['attention_mask'] = inputs['attention_mask'].to(torch.int32)
    return inputs


@app.command(help='Use `torch.jit.trace` trace models')
def trace(model_name_or_path: Annotated[str, typer.Argument(help='Model name or path')],
          batch_size: Annotated[int, typer.Option(help='Max input shape')] = 300,
          is_rerank_model: Annotated[
              Optional[bool], typer.Option("--rerank/--embedding", help='Model is rerank model')] = False,
          ):
    print_info()
    typer.echo(typer.style('INFO: ', fg=typer.colors.BLUE, bold=True) + "Load model ...")

    if is_rerank_model:
        inputs = gen_rerank_inputs(model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, trust_remote_code=True,
                                                                   torchscript=True)
    else:
        inputs = gen_embedding_inputs(model_name_or_path, batch_size)
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, torchscript=True)

    typer.echo(typer.style('INFO: ', fg=typer.colors.BLUE, bold=True) + "Tracking model ...")
    with torch.no_grad():
        traced_model = torch.jit.trace(func=model, example_inputs=[inputs['input_ids'], inputs['attention_mask']],
                                       strict=False)

    output_path = os.path.join(model_name_or_path, 'traced_model.ts')
    traced_model.save(output_path)
    typer.echo(typer.style('INFO: ', fg=typer.colors.BLUE, bold=True) + "Model traceability successful!")


@app.command(help='Use `mindietorch.compile` compile models')
def compile(model_name_or_path: Annotated[str, typer.Argument(help='Model name or path')],
            soc: Annotated[str, typer.Option(help='SOC version name')] = None,
            min_shape: Annotated[Tuple[int, int], typer.Option(help='Min input shape')] = (1, 1),
            max_shape: Annotated[Tuple[int, int], typer.Option(help='Max input shape, default: (300, `max_position_embeddings`)',show_default=False)] = None,
            is_rerank_model: Annotated[
                Optional[bool], typer.Option("--rerank/--embedding", help='Model is rerank model')] = False,

            ):
    print_info()
    if not is_mindietorch_available():
        typer.echo(typer.style('ERROR: ', fg=typer.colors.RED, bold=True) + "Mindietorch is not installed!")
        return

    if soc is None:
        soc = torch_npu.npu.get_device_name()
        typer.echo(
            typer.style('WARN: ', fg=typer.colors.YELLOW, bold=True) +
            "Set soc version to " +
            typer.style(soc, fg=typer.colors.GREEN, bold=True)
        )
    if max_shape is None:
        config = AutoConfig.from_pretrained(model_name_or_path)
        max_shape = (300, config.max_position_embeddings)
        typer.echo(
            typer.style('WARN: ', fg=typer.colors.YELLOW, bold=True) +
            "Set max shape to " +
            typer.style(str(max_shape), fg=typer.colors.GREEN, bold=True)
        )

    typer.echo(typer.style('INFO: ', fg=typer.colors.BLUE, bold=True) + "Load traced model ...")
    traced_model = torch.jit.load(os.path.join(model_name_or_path, 'traced_model.ts'))
    traced_model.eval()
    dynamic_inputs = [
        mindietorch.Input(min_shape=min_shape, max_shape=max_shape, dtype=torch.int32),
        mindietorch.Input(min_shape=min_shape, max_shape=max_shape, dtype=torch.int32)
    ]
    if is_rerank_model:
        precision_policy = mindietorch.PrecisionPolicy.FP16
    else:
        precision_policy = mindietorch.PrecisionPolicy.FP32

    typer.echo(typer.style('INFO: ', fg=typer.colors.BLUE, bold=True) + "Compiling model ...")
    with torch.no_grad():
        compiled_model = mindietorch.compile(
            traced_model,
            inputs=dynamic_inputs,
            precision_policy=precision_policy,
            truncate_long_and_double=True,
            require_full_compilation=False,
            allow_tensor_replace_int=False,
            min_block_size=3,
            torch_executed_ops=[],
            soc_version=soc,
            optimization_level=0
        )
    typer.echo()
    output_path = os.path.join(model_name_or_path, 'compiled_model.pt')
    compiled_model.save(output_path)
    typer.echo(typer.style('INFO: ', fg=typer.colors.BLUE, bold=True) + "Model compilation successful!")


@app.command()
def trace_and_compile(model_name_or_path: Annotated[str, typer.Argument(help='Model name or path')],
                      soc: Annotated[str, typer.Option(help='SOC version name')] = None,
                      min_shape: Annotated[Tuple[int, int], typer.Option(help='Min input shape')] = (1, 1),
                      max_shape: Annotated[Tuple[int, int], typer.Option(help='Max input shape, default: (300, `max_position_embeddings`)',show_default=False)] = None,
                      batch_size: Annotated[int, typer.Option(help='Max input shape')] = 300,
                      is_rerank_model: Annotated[
                          Optional[bool], typer.Option("--rerank/--embedding", help='Model is rerank model')] = False,
                      ):
    print_info()
    if not is_mindietorch_available():
        typer.echo(typer.style('ERROR: ', fg=typer.colors.RED, bold=True) + "Mindietorch is not installed!")
        return

    typer.echo(typer.style('INFO: ', fg=typer.colors.BLUE, bold=True) + "Load model ...")

    if is_rerank_model:
        inputs = gen_rerank_inputs(model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, trust_remote_code=True,
                                                                   torchscript=True)
    else:
        inputs = gen_embedding_inputs(model_name_or_path, batch_size)
        model = AutoModel.from_pretrained(model_name_or_path, trust_remote_code=True, torchscript=True)

    if max_shape is None:
        max_shape = (300, model.config.max_position_embeddings)
        typer.echo(
            typer.style('INFO: ', fg=typer.colors.YELLOW, bold=True) +
            "Set max shape to " +
            typer.style(str(max_shape), fg=typer.colors.GREEN, bold=True)
        )

    typer.echo(typer.style('INFO: ', fg=typer.colors.BLUE, bold=True) + "Tracking model ...")
    with torch.no_grad():
        traced_model = torch.jit.trace(func=model, example_inputs=[inputs['input_ids'], inputs['attention_mask']],
                                       strict=False)
    typer.echo(typer.style('INFO: ', fg=typer.colors.BLUE, bold=True) + "Model traceability successful!")

    if soc is None:
        soc = torch_npu.npu.get_device_name()
        typer.echo(
            typer.style('INFO: ', fg=typer.colors.YELLOW, bold=True) +
            "Set soc version to " +
            typer.style(soc, fg=typer.colors.GREEN, bold=True)
        )

    traced_model.eval()
    dynamic_inputs = [
        mindietorch.Input(min_shape=min_shape, max_shape=max_shape, dtype=torch.int32),
        mindietorch.Input(min_shape=min_shape, max_shape=max_shape, dtype=torch.int32)
    ]
    if is_rerank_model:
        precision_policy = mindietorch.PrecisionPolicy.FP16,
    else:
        precision_policy = mindietorch.PrecisionPolicy.FP32,

    typer.echo(typer.style('INFO: ', fg=typer.colors.BLUE, bold=True) + "Compiling model ...")
    with torch.no_grad():
        compiled_model = mindietorch.compile(
            traced_model,
            inputs=dynamic_inputs,
            precision_policy=precision_policy,
            truncate_long_and_double=True,
            require_full_compilation=False,
            allow_tensor_replace_int=False,
            min_block_size=3,
            torch_executed_ops=[],
            soc_version=soc,
            optimization_level=0
        )
    typer.echo()
    output_path = os.path.join(model_name_or_path, 'compiled_model.pt')
    compiled_model.save(output_path)
    typer.echo(typer.style('INFO: ', fg=typer.colors.BLUE, bold=True) + "Model compilation successful!")
