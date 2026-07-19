#!/usr/bin/env bash
set -o errexit

# Prefer CPU torch on Render to avoid huge NVIDIA CUDA wheels
export PIP_EXTRA_INDEX_URL="${PIP_EXTRA_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
pip install --upgrade pip
pip install "torch==2.2.2+cpu" || pip install "torch==2.2.2"
pip install -r requirements.txt
python manage.py collectstatic --no-input
python manage.py migrate --no-input
