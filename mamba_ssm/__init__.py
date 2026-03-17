__version__ = "2.2.6.post3"

"""Lightweight mamba_ssm package init for Aether-lite.

This project only needs the low-level CUDA selective_scan ops
(`selective_scan_fn`, `mamba_inner_fn`).

The original upstream __init__ also imports high-level language models
(`Mamba`, `Mamba2`, `MambaLMHeadModel`) which in turn depend on
`huggingface_hub` APIs (e.g. list_repo_tree). On some environments the
installed huggingface_hub version does not provide these symbols, which
causes an ImportError during package import and prevents us from using
the CUDA kernels at all.

To keep the selective_scan CUDA path usable and avoid unnecessary
dependencies, we only re-export the ops that Aether-lite actually
uses. High-level model classes can be re-added if needed, but are not
required for JSCC training here.
"""

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

__all__ = [
    "selective_scan_fn",
    "mamba_inner_fn",
]
