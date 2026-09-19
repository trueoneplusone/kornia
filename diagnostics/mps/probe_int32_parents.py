"""Small exact-integer probes for the proposed MPS parent representation.

This is diagnostic code, not a Kornia implementation or a huge-allocation test.
Run with the same interpreter and pinned torch version as the first MPS probe.
"""

import argparse
import json
import os
import platform
import sys

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    args = parser.parse_args()
    device = args.device
    if device == "mps":
        assert torch.backends.mps.is_available(), "MPS unavailable: refuse to skip"
    torch.set_num_threads(1)
    print(json.dumps({"event": "environment", "python": sys.version,
                      "torch": torch.__version__, "torch_git": torch.version.git_version,
                      "platform": platform.platform(), "device": device,
                      "fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")}), flush=True)

    def check(name, actual, expected):
        if device == "mps":
            torch.mps.synchronize()
        expected = torch.tensor(expected, dtype=actual.dtype)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
        print(json.dumps({"case": name, "result": actual.cpu().tolist(),
                          "dtype": str(actual.dtype), "passed": True}), flush=True)

    max_parent = torch.iinfo(torch.int32).max
    # Values near the range limit are checked without allocating 2**31 nodes.
    target = torch.tensor([max_parent, max_parent, max_parent, 0], dtype=torch.int32, device=device)
    index = torch.tensor([0, 0, 1, 1, 2, 3], dtype=torch.int64, device=device)
    src = torch.tensor([max_parent, max_parent - 1, max_parent, 0, max_parent, 1],
                       dtype=torch.int32, device=device)
    target.scatter_reduce_(0, index, src, reduce="amin")
    check("scatter-int32-limit-long-index", target, [max_parent - 1, 0, max_parent, 0])

    # Actual hooking destinations originate as int32 values and are widened.
    parents = torch.arange(6, dtype=torch.int32, device=device)
    sources = torch.tensor([0, 1, 4], dtype=torch.int64, device=device)
    targets = torch.tensor([1, 2, 5], dtype=torch.int64, device=device)
    source_roots, target_roots = parents[sources], parents[targets]
    parents.scatter_reduce_(0, torch.maximum(source_roots, target_roots).to(torch.int64),
                            torch.minimum(source_roots, target_roots), reduce="amin")
    check("hook-int32-parents-long-destination", parents, [0, 0, 1, 3, 4, 4])

    # Int32 advanced indices occur in the unmodified pointer-jumping expression.
    parents = torch.tensor([0, 0, 1, 2, 4, 4], dtype=torch.int32, device=device)
    check("pointer-jump-int32-index", parents[parents], [0, 0, 0, 1, 4, 4])
    check("pointer-jump-long-index", parents[parents.to(torch.int64)], [0, 0, 0, 1, 4, 4])

    # Widen BEFORE adding one: the public final label may equal 2**31.
    roots = torch.tensor([0, max_parent - 1, max_parent], dtype=torch.int32, device=device)
    labels = roots.to(torch.int64) + 1
    check("label-widen-before-one-based", labels, [1, 2**31 - 1, 2**31])
    mask = torch.tensor([True, False, True], dtype=torch.bool, device=device)
    check("label-mask-int64", labels * mask, [1, 0, 2**31])
    check("label-repeat-int64", labels.repeat_interleave(2),
          [1, 1, 2**31 - 1, 2**31 - 1, 2**31, 2**31])

    # No graph edges may leave a flattened image; a long offset can recover
    # global labels if a future implementation processes batch images locally.
    local = torch.tensor([[0, 2], [0, 2]], dtype=torch.int32, device=device)
    offsets = torch.tensor([[0], [2**31 + 7]], dtype=torch.int64, device=device)
    check("possible-local-parent-offset", local.to(torch.int64) + offsets + 1,
          [[1, 3], [2**31 + 8, 2**31 + 10]])


if __name__ == "__main__":
    main()
