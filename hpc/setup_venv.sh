#!/bin/bash -l
# Execute once, on a cluster login / CPU node. Do not execute from a GPU training job.
set -Eeuo pipefail

PROJECT_ROOT=/vast.mnt/home/20201733/Master_Thesis
VENV="${VENV:-$PROJECT_ROOT/.venv-gnn}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PYTHON_MODULE="${PYTHON_MODULE:-Python/3.11.3-GCCcore-12.3.0}"

module purge
module load Umbrella/2024
module load "$PYTHON_MODULE"
# Do NOT load the preinstalled PyTorch 2.1.2 module: we use a pinned pip
# installation of PyTorch 2.5.1 and PyTorch Geometric 2.8.1 in our venv.
module list

command -v "$PYTHON_BIN" || { echo "Cannot find $PYTHON_BIN after loading $PYTHON_MODULE" >&2; exit 1; }
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3,11), sys.version'

if [[ -e "$VENV/bin/activate" ]]; then
    # Avoid accidentally reusing a venv created using the old EESSI Python.
    module_base="$("$PYTHON_BIN" -c 'import sys; print(sys.base_prefix)')"
    venv_base="$("$VENV/bin/python" -c 'import sys; print(sys.base_prefix)')"
    if [[ "$module_base" != "$venv_base" ]]; then
        echo "Existing venv was created using a different Python installation:" >&2
        echo "  module base: $module_base" >&2
        echo "  venv base:   $venv_base" >&2
        echo "Create a new venv via VENV=/path/to/new-venv or move the old one." >&2
        exit 1
    fi
    echo "Using existing venv $VENV (pip will check/update pinned versions)."
else
    "$PYTHON_BIN" -m venv "$VENV"
fi
source "$VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu121 'torch==2.5.1'
python -m pip install -r "$PROJECT_ROOT/hpc/requirements-gnn.txt"
python -m pip check
python - <<'PY'
import sys, torch, torch_geometric, numpy, pandas, lxml
from torch_geometric.nn import GraphNorm, TransformerConv
print('Python:', sys.version.split()[0]); print('Torch:', torch.__version__)
print('Torch CUDA build:', torch.version.cuda)
print('PyG:', torch_geometric.__version__)
print('CUDA available here (can be false on login nodes):', torch.cuda.is_available())
assert torch.version.cuda == '12.1', 'Expected CUDA 12.1 PyTorch wheel'
assert torch.__version__.startswith('2.5.1')
assert torch_geometric.__version__ == '2.8.1'
PY
python -m pip freeze > "$VENV/pip-freeze.txt"
echo "Environment ready: $VENV"
