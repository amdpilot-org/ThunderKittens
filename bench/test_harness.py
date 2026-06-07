#!/usr/bin/env python3
"""Stage0 baseline harness for ThunderKittens#2.

Runs Liger-Kernel swiglu and rmsnorm baselines on a single ROCm GPU and validates
against PyTorch references. Each op is measured in a fresh Python subprocess: on
this ROCm/Liger combination, importing both Liger op modules before first CUDA
allocation causes the worker process to be SIGKILLed, while each op runs normally
in isolation. The isolation does not change the metric semantics because the
issue compares individual swiglu/rmsnorm tile kernels against Liger baselines.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from typing import Callable, Dict, Tuple


def _time_cuda(torch, fn: Callable[[], object], warmup: int, iters: int) -> Tuple[float, object]:
    for _ in range(warmup):
        out = fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(times), out


def _run_child(args: argparse.Namespace) -> None:
    def dbg(msg: str) -> None:
        if os.environ.get("STAGE0_DEBUG"):
            print(msg, file=sys.stderr, flush=True)
    dbg(f"child start op={args.op}")
    import torch
    import torch.nn.functional as F
    dbg("torch imported")

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm GPU is not visible to PyTorch")
    torch.manual_seed(1234)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    if args.op == "swiglu":
        from liger_kernel.ops.swiglu import LigerSiLUMulFunction
        dbg("swiglu imported")

        gate = torch.randn((args.tokens, args.intermediate), device=device, dtype=dtype)
        up = torch.randn((args.tokens, args.intermediate), device=device, dtype=dtype)
        dbg("swiglu tensors allocated")
        ms, out = _time_cuda(torch, lambda: LigerSiLUMulFunction.apply(gate, up), args.warmup, args.iters)
        dbg("swiglu timed")
        ref = (F.silu(gate.float()) * up.float()).to(dtype)
        diff = (out - ref).abs().max().item()
        result = {"op": "swiglu", "kernel_ms": ms, "max_abs_diff": diff}
    elif args.op == "rmsnorm":
        from liger_kernel.ops.rms_norm import LigerRMSNormFunction
        dbg("rmsnorm imported")

        eps = 1e-6
        x = torch.randn((args.tokens, args.hidden), device=device, dtype=dtype)
        weight = torch.randn((args.hidden,), device=device, dtype=dtype)
        dbg("rmsnorm tensors allocated")
        ms, out = _time_cuda(
            torch,
            lambda: LigerRMSNormFunction.apply(x, weight, eps, 0.0, "llama"),
            args.warmup,
            args.iters,
        )
        dbg("rmsnorm timed")
        fp = x.float()
        ref = (fp * torch.rsqrt(fp.pow(2).mean(dim=-1, keepdim=True) + eps) * weight.float()).to(dtype)
        diff = (out - ref).abs().max().item()
        result = {"op": "rmsnorm", "kernel_ms": ms, "max_abs_diff": diff}
    else:
        raise SystemExit(f"unknown child op {args.op}")
    print(json.dumps(result, sort_keys=True), flush=True)
    os._exit(0)


def _child_measure(op: str, shape: Dict[str, int], warmup: int, iters: int) -> Dict[str, float]:
    cmd = [
        sys.executable,
        os.path.abspath(__file__),
        "--child-op", op,
        "--tokens", str(shape["tokens"]),
        "--hidden", str(shape["hidden"]),
        "--intermediate", str(shape["intermediate"]),
        "--warmup", str(warmup),
        "--iters", str(iters),
    ]
    out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    return json.loads(out.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=int(os.environ.get("TOKENS", "256")))
    parser.add_argument("--hidden", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--intermediate", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--warmup", type=int, default=int(os.environ.get("WARMUP", "10")))
    parser.add_argument("--iters", type=int, default=int(os.environ.get("ITERS", "30")))
    parser.add_argument("--child-op", choices=["swiglu", "rmsnorm"], default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.child_op:
        args.op = args.child_op
        _run_child(args)
        return

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA/ROCm GPU is not visible to PyTorch")
    props = torch.cuda.get_device_properties(0)

    shapes = [
        {"shape": "llama3.3-70b-ffn", "tokens": args.tokens, "hidden": 8192, "intermediate": 14336},
        {"shape": "qwen3-32b-ffn", "tokens": args.tokens, "hidden": 5120, "intermediate": 8192},
    ]
    results = {
        "schema_version": "stage0_harness.v1",
        "torch_version": torch.__version__,
        "device_name": props.name,
        "gcn_arch": getattr(props, "gcnArchName", "unknown"),
        "dtype": "bf16",
        "warmup": args.warmup,
        "iters": args.iters,
        "shapes": [],
    }
    for shape in shapes:
        sw = _child_measure("swiglu", shape, args.warmup, args.iters)
        rms = _child_measure("rmsnorm", shape, args.warmup, args.iters)
        row = dict(shape)
        row.update({
            "swiglu_ms": sw["kernel_ms"],
            "rmsnorm_ms": rms["kernel_ms"],
            "combined_ms": sw["kernel_ms"] + rms["kernel_ms"],
            "swiglu_max_abs_diff": sw["max_abs_diff"],
            "rmsnorm_max_abs_diff": rms["max_abs_diff"],
        })
        results["shapes"].append(row)

    worst_diff = max(max(s["swiglu_max_abs_diff"], s["rmsnorm_max_abs_diff"]) for s in results["shapes"])
    results["worst_torch_reference_max_abs_diff"] = worst_diff
    results["torch_reference_lt_1e_3"] = bool(worst_diff < 1e-3)
    results["baseline_metric_extracted"] = all(
        s["swiglu_ms"] > 0 and s["rmsnorm_ms"] > 0 and s["combined_ms"] > 0 for s in results["shapes"]
    )
    llama_combined = results["shapes"][0]["combined_ms"]
    print(f"liger_llama_swiglu_rmsnorm_combined_ms: {llama_combined}")
    results["parity_note"] = "Future ThunderKittens outputs must compare against these Liger baseline kernels; PyTorch reference diffs are informational for bf16."
    print(json.dumps(results, indent=2, sort_keys=True))
    if not results["baseline_metric_extracted"]:
        raise SystemExit("failed to extract positive Liger baseline metrics")


if __name__ == "__main__":
    main()
