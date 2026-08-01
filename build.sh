#!/usr/bin/env bash
set -o errexit

pip install --upgrade pip
pip install -r requirements.txt

# ── Export the embedding model to ONNX format ────────────────────────
# This temporarily installs torch + optimum ONLY for the export step.
# At runtime, only onnxruntime is used (torch is NOT imported).
if [ ! -f "./onnx_model/model.onnx" ]; then
    echo "==> Exporting all-MiniLM-L6-v2 to ONNX format..."
    pip install "optimum[onnxruntime]" "torch==2.2.2" --extra-index-url https://download.pytorch.org/whl/cpu --no-cache-dir
    optimum-cli export onnx \
        --model sentence-transformers/all-MiniLM-L6-v2 \
        --task feature-extraction \
        ./onnx_model/
    echo "==> ONNX model exported to ./onnx_model/"
else
    echo "==> ONNX model already exists, skipping export."
fi

python manage.py collectstatic --no-input
python manage.py migrate --no-input
