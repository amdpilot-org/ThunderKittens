# Stage0 Baseline Bundle — ThunderKittens#2

This directory contains the reproducible Liger-Kernel baseline used to validate future ThunderKittens swiglu + rmsnorm tile kernels on AMD MI355X (`gfx950`).

## Hardware / Software Context

| Item | Value |
|------|-------|
| GPU | AMD Instinct MI355X (gfx950:sramecc+:xnack-) |
| ROCm / PyTorch | 2.9.1+rocm7.2.0.git7e1940d4 |
| Base image | `rocm/pytorch:rocm6.4_ubuntu22.04_py3.10_pytorch_release_2.6.0` |
| Liger-Kernel | Installed in container venv |

## Reproduce the Baseline

```bash
python bench/test_harness.py
```

The harness runs each Liger op in a fresh subprocess (importing both `liger_kernel.ops.swiglu` and `liger_kernel.ops.rms_norm` in the same process before first CUDA allocation triggers a hang on this ROCm/Liger combination).

Default settings:
- `tokens = 256`
- `warmup = 10`
- `iters = 30`
- `dtype = bf16`

Shapes:
- Llama-3.3-70B FFN: `hidden=8192`, `intermediate=14336`
- Qwen3-32B FFN: `hidden=5120`, `intermediate=8192`

## Acceptance Gate for Future ThunderKittens Candidates

Future executor agents must satisfy **both** of the following when adding a ThunderKittens kernel:

1. **Parity**: `max_abs_diff < 1e-3` against the Liger baseline (not the PyTorch fp32 reference — bf16 tolerances are informational).
2. **Latency**: ThunderKittens `kernel_ms` must be within **1.1×** of the Liger Llama-3.3-70B FFN baseline (`swiglu_ms + rmsnorm_ms`) reported by this harness.

## Baseline Artifact

`bench/stage0_liger_baseline.json` captures the JSON output of a representative run, including:
- ROCm / PyTorch version
- `gcn_arch`
- Warmup / timing config
- Per-shape `swiglu_ms`, `rmsnorm_ms`, and `combined_ms`
- `worst_torch_reference_max_abs_diff` (informational for bf16)

## Current Baseline Headline

```
liger_llama_swiglu_rmsnorm_combined_ms: ~0.075
```

(Exact value varies slightly run-to-run; use the JSON artifact for precise comparison.)
