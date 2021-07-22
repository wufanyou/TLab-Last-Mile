#!/bin/sh
set -eu

export PYTHONUNBUFFERED=1
exec python src/model_apply_main.py
