#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The rocket-userspace authors
#
# Fetch the model, the labels and a test image mknet.py turns into the .rnet blob
# tests/rk3576_net_gate.c runs. Everything this pulls is a well-known public artifact;
# none of it is committed, because the derived blob is ~7 MB and reproducing it is one
# command.
#
#   ./fetch.sh && python3 mknet.py
#
# mknet.py needs `tflite` (the schema) and `ai-edge-litert` (the reference
# interpreter), plus numpy and Pillow:
#
#   pip install --target=./pylibs numpy Pillow tflite ai-edge-litert
#   PYTHONPATH=./pylibs python3 mknet.py
set -e
cd "$(dirname "$0")"

MODEL=mobilenet_v1_1.0_224_quant.tflite
BASE=https://raw.githubusercontent.com/google-coral/test_data/master

[ -f "$MODEL" ]  || curl -fsSL -o "$MODEL"  "$BASE/$MODEL"
[ -f labels.txt ] || curl -fsSL -o labels.txt "$BASE/imagenet_labels.txt"
[ -f grace_hopper.bmp ] || curl -fsSL -o grace_hopper.bmp \
  https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/examples/label_image/testdata/grace_hopper.bmp

ls -l "$MODEL" labels.txt grace_hopper.bmp
