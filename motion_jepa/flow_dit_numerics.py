"""Opt-in PyTorch determinism with settings included in exact-resume identity."""
from __future__ import annotations

import os

import torch


def strict_numerics_requested(config):
    enabled = config.get('train', {}).get('deterministic_algorithms', False)
    if type(enabled) is not bool:
        raise ValueError('train.deterministic_algorithms must be a boolean')
    return enabled


def configure_numerics(config):
    """Call before Accelerator or any CUDA context is initialized.

    Historical configurations retain their existing behavior. This enables
    PyTorch's own deterministic implementations; it does not replace operators
    or assert that arbitrary distributed topologies are bitwise equivalent.
    """
    if not strict_numerics_requested(config):
        return
    workspace = os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if workspace not in (None, ':4096:8'):
        raise ValueError('strict Stage1 requires CUBLAS_WORKSPACE_CONFIG=:4096:8')
    if torch.cuda.is_initialized():
        raise ValueError('configure strict Stage1 numerics before CUDA initialization')
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def numerics_contract(config):
    """Return actual strict runtime settings; preserve legacy contracts verbatim."""
    if not strict_numerics_requested(config):
        return None
    if (os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8'
            or not torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled()
            or torch.backends.cudnn.benchmark
            or not torch.backends.cudnn.deterministic):
        raise ValueError('strict Stage1 numeric settings were not applied or changed')
    cuda_initialized = torch.cuda.is_initialized()
    return {
        'schema': 'stage1_torch_strict_numerics_v1',
        'torch_version': str(torch.__version__), 'cuda_version': torch.version.cuda,
        'cudnn_version': torch.backends.cudnn.version(),
        'cublas_workspace_config': os.environ['CUBLAS_WORKSPACE_CONFIG'],
        'deterministic_algorithms': True, 'warn_only': False,
        'cudnn_benchmark': False, 'cudnn_deterministic': True,
        'float32_matmul_precision': torch.get_float32_matmul_precision(),
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'matmul_allow_fp16_reduced_precision_reduction': (
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction),
        'matmul_allow_bf16_reduced_precision_reduction': (
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'flash_sdp': torch.backends.cuda.flash_sdp_enabled(),
        'memory_efficient_sdp': torch.backends.cuda.mem_efficient_sdp_enabled(),
        'math_sdp': torch.backends.cuda.math_sdp_enabled(),
        'cudnn_sdp': torch.backends.cuda.cudnn_sdp_enabled(),
        'math_sdp_allow_fp16_bf16_reduction': (
            torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()),
        'cuda_devices_per_rank': torch.cuda.device_count() if cuda_initialized else 0,
        'cuda_capability': list(torch.cuda.get_device_capability()) if cuda_initialized else None,
    }
