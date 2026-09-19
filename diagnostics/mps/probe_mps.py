"""Read-only MPS diagnostics; every case runs in a fresh crash-isolated process."""
import argparse
import faulthandler
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback


SHAPES = {"row": (1, 1, 513), "column": (2, 1, 513, 1),
          "batch": (2, 3, 1, 17, 19)}


def emit(event, **kwargs):
    print(json.dumps({"event": event, **kwargs}), flush=True)


def metadata(tensor):
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
            "offset": tensor.storage_offset(), "dtype": str(tensor.dtype),
            "device": str(tensor.device), "contiguous": tensor.is_contiguous()}


def sync(torch, device):
    if device == "mps":
        torch.mps.synchronize()


def edges(torch, shape, device, dtype):
    height, width = shape[-2:]
    count = 1
    for extent in shape[:-2]:
        count *= extent
    mask = (torch.arange(count * height * width, device=device) % 3 != 0).reshape(count, height, width)
    padded = mask.new_zeros((count, height + height % 2, width + width % 2))
    padded[:, :height, :width] = mask
    tl, tr = padded[:, ::2, ::2], padded[:, ::2, 1::2]
    bl, br = padded[:, 1::2, ::2], padded[:, 1::2, 1::2]
    blocks = torch.arange(tl.numel(), device=device, dtype=dtype).reshape(tl.shape)
    return {
        "left": ((tl[:, :, 1:] | bl[:, :, 1:]) & (tr[:, :, :-1] | br[:, :, :-1]),
                 blocks[:, :, :-1], blocks[:, :, 1:]),
        "above": ((tl[:, 1:, :] | tr[:, 1:, :]) & (bl[:, :-1, :] | br[:, :-1, :]),
                  blocks[:, :-1, :], blocks[:, 1:, :]),
        "diag": (tl[:, 1:, 1:] & br[:, :-1, :-1], blocks[:, :-1, :-1], blocks[:, 1:, 1:]),
    }


def where_case(torch, device, shape_name, edge, layout, dtype_name):
    dtype = getattr(torch, dtype_name)
    cpu_args = edges(torch, SHAPES[shape_name], "cpu", dtype)[edge]
    expected = torch.where(*cpu_args)
    values = edges(torch, SHAPES[shape_name], device, dtype)[edge]
    if layout == "contiguous":
        values = tuple(t.contiguous() for t in values)
    elif layout == "flat":
        values = tuple(t.reshape(-1) for t in values)
    emit("before_where", tensors=[metadata(t) for t in values])
    sync(torch, device)
    actual = torch.where(*values)
    sync(torch, device)
    emit("after_where", result=metadata(actual))
    torch.testing.assert_close(actual.cpu().reshape(expected.shape), expected, rtol=0, atol=0)


def scatter_case(torch, device, dtype_name, layout):
    dtype = getattr(torch, dtype_name)
    base = torch.arange(8, dtype=dtype) + (2**40 if dtype == torch.int64 else 1000)
    index = torch.tensor([1, 1, 3, 3, 3, 6, 6, 7], dtype=torch.int64)
    src = torch.tensor([5, -3, 7, -9, 2, 1, -2, 0], dtype=dtype)
    expected = base.clone().scatter_reduce_(0, index, src, reduce="amin")
    target, index, src = base.to(device), index.to(device), src.to(device)
    if layout == "strided":
        def strided(t):
            storage = torch.zeros(t.numel() * 2, device=device, dtype=t.dtype)
            storage[::2] = t
            return storage[::2]
        index, src = strided(index), strided(src)
    elif layout == "empty":
        index, src = index[:0], src[:0]
        expected = base
    emit("before_scatter", tensors=[metadata(t) for t in (target, index, src)])
    sync(torch, device)
    target.scatter_reduce_(0, index, src, reduce="amin")
    sync(torch, device)
    emit("after_scatter", actual=target.cpu().tolist(), expected=expected.tolist())
    torch.testing.assert_close(target.cpu(), expected, rtol=0, atol=0)


def ccl_case(torch, device, source, shape_name, variant):
    spec = importlib.util.spec_from_file_location("diagnostic_ccl", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    shape = SHAPES[shape_name]
    image = torch.ones(shape, device=device, dtype=torch.float32)
    batch_count = image.numel() // (shape[-2] * shape[-1])
    stride = ((shape[-2] + 1) // 2) * ((shape[-1] + 1) // 2)
    expected = (torch.arange(batch_count, dtype=torch.int64) * stride + 1)
    expected = expected[:, None].expand(batch_count, shape[-2] * shape[-1]).reshape(shape)
    original_where = torch.where
    original_scatter = torch.Tensor.scatter_reduce_

    def traced_where(*args, **kwargs):
        emit("before_ccl_where", tensors=[metadata(t) for t in args])
        sync(torch, device)
        output_shape = args[0].shape
        assert all(t.shape == output_shape for t in args)
        if variant == "contiguous":
            args = tuple(t.contiguous() for t in args)
        elif variant == "flat":
            args = tuple(t.reshape(-1) for t in args)
        result = original_where(*args, **kwargs).reshape(output_shape)
        sync(torch, device)
        emit("after_ccl_where", result=metadata(result))
        return result

    def traced_scatter(self, dim, index, src, **kwargs):
        emit("before_ccl_scatter", tensors=[metadata(t) for t in (self, index, src)])
        sync(torch, device)
        result = original_scatter(self, dim, index, src, **kwargs)
        sync(torch, device)
        emit("after_ccl_scatter", minimum=int(result.min().cpu()), maximum=int(result.max().cpu()))
        return result

    torch.where = traced_where
    torch.Tensor.scatter_reduce_ = traced_scatter
    try:
        actual = module.connected_components_union_find(image)
        sync(torch, device)
    finally:
        torch.where = original_where
        torch.Tensor.scatter_reduce_ = original_scatter
    emit("ccl_labels", unique=actual.cpu().unique().tolist(), expected_unique=expected.unique().tolist())
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


def child(args):
    faulthandler.enable()
    import torch
    torch.set_num_threads(1)
    emit("environment", python=sys.version, torch=torch.__version__,
         torch_git=torch.version.git_version, platform=platform.platform(),
         mps_available=torch.backends.mps.is_available(), device=args.device,
         fallback=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"), case=args.case)
    if args.device == "mps":
        assert torch.backends.mps.is_available(), "MPS unavailable: refuse to skip"
    parts = args.case.split(":")
    if parts[0] == "where":
        where_case(torch, args.device, *parts[1:])
    elif parts[0] == "scatter":
        scatter_case(torch, args.device, *parts[1:])
    elif parts[0] == "ccl":
        ccl_case(torch, args.device, args.source, *parts[1:])
    elif parts[0] == "gather":
        data = torch.tensor([2**40, 2**40 + 3, -3, 8], device=args.device, dtype=torch.int64)
        index = torch.tensor([3, 1, 1, 0], device=args.device)
        actual = data[index]
        sync(torch, args.device)
        torch.testing.assert_close(actual.cpu(), torch.tensor([8, 2**40 + 3, 2**40 + 3, 2**40]), rtol=0, atol=0)
    emit("passed", case=args.case)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--output", type=Path, default=Path("mps-probe-results"))
    parser.add_argument("--case")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--pytest-repo", type=Path)
    args = parser.parse_args()
    if args.case:
        child(args)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [f"where:{shape}:{edge}:{layout}:{dtype}"
             for shape in SHAPES for edge in ("left", "above", "diag")
             for layout, dtype in (("native", "int64"), ("contiguous", "int64"),
                                   ("flat", "int64"), ("native", "int32"))]
    cases += [f"scatter:{dtype}:{layout}" for dtype in ("int64", "int32")
              for layout in ("contiguous", "strided", "empty")]
    cases += ["gather:int64"]
    cases += [f"ccl:{shape}:{variant}" for shape in SHAPES
              for variant in ("native", "contiguous", "flat")]
    results = []

    def run(name, command, cwd=None):
        env = dict(os.environ, PYTHONFAULTHANDLER="1", PYTHONUNBUFFERED="1",
                   OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   KORNIA_TEST_DEVICE=args.device, KORNIA_TEST_DTYPE="float32")
        stem = args.output / name.replace(":", "-")
        try:
            proc = subprocess.run(command, capture_output=True, text=True,
                                  timeout=args.timeout, cwd=cwd, env=env)
            stdout, stderr, code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired as error:
            stdout = error.stdout or b""
            stderr = error.stderr or b""
            stdout = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
            stderr = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
            code = "timeout"
        stem.with_suffix(".stdout.log").write_text(stdout, encoding="utf-8")
        stem.with_suffix(".stderr.log").write_text(stderr, encoding="utf-8")
        row = {"case": name, "returncode": code, "command": command}
        results.append(row)
        (args.output / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        emit("child_result", **row)

    for case in cases:
        run(case, [sys.executable, str(Path(__file__).resolve()), "--source", str(args.source.resolve()),
                   "--device", args.device, "--case", case])
    if args.pytest_repo:
        # Collect real node IDs, then isolate each original regression so a
        # fatal abort cannot swallow traceback reports for preceding failures.
        test_file = "tests/contrib/test_connected_components_union_find.py"
        env = dict(os.environ, KORNIA_TEST_DEVICE=args.device, KORNIA_TEST_DTYPE="float32")
        collected = subprocess.run([sys.executable, "-m", "pytest", test_file,
                                    "--collect-only", "-q", "-k", "long_connected_regions"],
                                   cwd=args.pytest_repo, env=env, capture_output=True, text=True, check=True)
        (args.output / "pytest-collection.log").write_text(collected.stdout + collected.stderr, encoding="utf-8")
        nodes = [line.strip() for line in collected.stdout.splitlines() if line.startswith(test_file + "::")]
        assert len(nodes) == 3, nodes
        for n, node in enumerate(nodes):
            run(f"pytest-long-{n}", [sys.executable, "-m", "pytest", node, "-vv", "-s", "--tb=long"],
                cwd=args.pytest_repo)
    emit("finished", processes=len(results), failures=sum(r["returncode"] != 0 for r in results))
    if any(row["returncode"] != 0 for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
