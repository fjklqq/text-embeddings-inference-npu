# -*- coding: utf-8 -*-
# @Time     : 2024/11/23 22:14
# @Author   : Hang Zhou
# @Email    : fjklqq@gmail.com
# @Software : Python 3.11
# @About    :
import importlib.metadata
import importlib.util
import platform
import subprocess
import torch
import typer
import transformers
from packaging import version
from transformers.utils import is_torch_cuda_available, is_torch_npu_available


def _is_package_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _get_package_version(name: str) -> "Version":
    try:
        return version.parse(importlib.metadata.version(name))
    except Exception:
        return version.parse("0.0.0")


def is_mindietorch_available():
    return _is_package_available("mindietorch")

def print_info():
    typer.echo(
        typer.style('INFO: ', fg=typer.colors.BLUE, bold=True) +
        f"Platform: " + typer.style(platform.platform(), fg=typer.colors.GREEN, bold=True)
    )
    typer.echo(
        typer.style('INFO: ', fg=typer.colors.BLUE, bold=True) +
        f"Python version: " + typer.style(platform.python_version(), fg=typer.colors.GREEN, bold=True)
    )
    typer.echo(
        typer.style('INFO: ', fg=typer.colors.BLUE, bold=True) +
        f"Transformers version: " + typer.style(transformers.__version__, fg=typer.colors.GREEN, bold=True)
    )

    torch_version = torch.__version__
    if is_torch_cuda_available():
        torch_version += " (GPU)"
        typer.echo(
            typer.style('INFO: ', fg=typer.colors.BLUE, bold=True) +
            f"GPU type: " + typer.style(torch.cuda.get_device_name(), fg=typer.colors.GREEN, bold=True)
        )


    if is_torch_npu_available():
        torch_version += " (NPU)"
        typer.echo(
            typer.style('INFO: ', fg=typer.colors.BLUE, bold=True) +
            f"NPU type: " + typer.style(torch.npu.get_device_name(), fg=typer.colors.GREEN, bold=True)
        )
        typer.echo(
            typer.style('INFO: ', fg=typer.colors.BLUE, bold=True) +
            f"CANN version: " + typer.style(torch.version.cann, fg=typer.colors.GREEN, bold=True)
        )

    typer.echo(
        typer.style('INFO: ', fg=typer.colors.BLUE, bold=True) +
        f"PyTorch version: " + typer.style(torch_version, fg=typer.colors.GREEN, bold=True)
    )

    if is_mindietorch_available():
        p = subprocess.Popen("pip list|grep mindietorch|awk '{print $NF}'", shell=True, stdout=subprocess.PIPE)
        out, err = p.communicate()
        version= str(out, encoding="utf-8").strip()
        typer.echo(
            typer.style('INFO: ', fg=typer.colors.BLUE, bold=True) +
            f"Mindietorch version: " + typer.style(version, fg=typer.colors.GREEN, bold=True)
        )

