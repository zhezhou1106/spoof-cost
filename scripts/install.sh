#!/usr/bin/env bash
# Fetch + patch Obfuscation Atlas, install spoof-cost[atlas], and verify imports.
# Requires: Python >=3.12, uv, CUDA 12.x on PATH for GPU runs.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="${LD_PRELOAD:-}"
export PATH="${PATH:-}"

if [[ -n "${CUDA_HOME:-}" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
fi

echo "=============================================="
echo "install host=$(hostname)"
echo "CUDA_HOME=${CUDA_HOME:-unset}"
echo "nvcc=$(command -v nvcc || true)"
nvcc --version 2>/dev/null | head -4 || true
nvidia-smi 2>/dev/null || true
echo "=============================================="

ATLAS_DIR="third_party/obfuscation-atlas"
ATLAS_REPO="${ATLAS_REPO:-https://github.com/AlignmentResearch/obfuscation-atlas}"
ATLAS_COMMIT="12abf65567e224306feb2d8c9ccfe4a0c90aa820"
ATLAS_PATCH="$ROOT/third_party/patches/obfuscation-atlas-spoof-cost.patch"

if [[ ! -d "$ATLAS_DIR/.git" ]]; then
  echo "+ cloning Obfuscation Atlas @ ${ATLAS_COMMIT:0:7}"
  git clone "$ATLAS_REPO" "$ATLAS_DIR"
fi
git -C "$ATLAS_DIR" checkout --quiet "$ATLAS_COMMIT"
if git -C "$ATLAS_DIR" apply --reverse --check "$ATLAS_PATCH" 2>/dev/null; then
  echo "+ Atlas patch already applied"
else
  echo "+ applying $(basename "$ATLAS_PATCH")"
  git -C "$ATLAS_DIR" apply "$ATLAS_PATCH"
fi

SYNC=(uv sync --extra atlas --extra dev)
if [[ -n "${UV_PYTHON_PLATFORM:-}" ]]; then
  SYNC+=(--python-platform "${UV_PYTHON_PLATFORM}")
fi
echo "+ ${SYNC[*]}"
"${SYNC[@]}"

echo "+ verifying imports"
uv run python - <<'PY'
import obfuscation_atlas
import afterburner

print("obfuscation_atlas ok")
print("afterburner ok")
try:
    import torch
    import vllm
except ImportError as exc:
    print(f"optional GPU import skipped: {exc}")
else:
    print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()} device_count={torch.cuda.device_count()}")
    print(f"vllm={vllm.__version__}")
    import afterburner.generation.vllm.server  # noqa: F401
    print("afterburner.generation.vllm.server ok")
    if torch.cuda.is_available():
        print(f"gpu0={torch.cuda.get_device_name(0)}")
PY

echo "install: OK"
