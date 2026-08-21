#!/bin/sh
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The rocket-userspace authors
#
# Fetch the models, the labels and a test image mknet.py turns into the .rnet blobs
# tests/rk3576_net_gate.c runs. Everything this pulls is a well-known public artifact;
# none of it is committed, because the derived blobs are ~7 MB each and reproducing them
# is one command.
#
#   ./fetch.sh && python3 mknet.py            # every network that is present
#   ./fetch.sh && python3 mknet.py v2         # just one
#
# MobileNetV1 is the feed-forward chain; MobileNetV2 adds the ten residual skips and the
# channel counts that are not multiples of 32, which is a different question about the
# part rather than a bigger instance of the same one. ResNet-18 is a third: 3x3-heavy
# where both MobileNets are 1x1-heavy, with a 7x7 stem, a max pool, and skips whose two
# operands are the same width so the graph is all identity-shortcut blocks. Inception V1 is
# a fourth, chosen for the axes it VARIES rather than for size: a concat/split topology
# (nine four-operand joins), nine producers with four readers, a k=7 pooling kernel,
# thirteen pooling layers, and 18+18 layers whose operand A is not the layer before. It
# arrives quantized like the MobileNets do. Inception V3 at 299 is a fifth, and it is
# the first with a NON-SQUARE kernel (1x7, 7x1, 1x3, 3x1), VALID padding, odd planes and
# a requantization edge -- axes every earlier gate cell on this part holds constant, and
# each one turned out to hide a shipped encoding that computed a plausible wrong surface.
#
# THE TWO MOBILENETS ARRIVE QUANTIZED; RESNET-18 DOES NOT. No per-tensor-quantized
# ResNet-18 is published, so this pulls the float checkpoint and the calibration images
# and mkresnet18.py assembles and quantizes the model locally. That needs TensorFlow's
# converter, which the blob builder does not:
#
#   uv venv .venv-build && uv pip install --python .venv-build/bin/python \
#       "tensorflow-cpu==2.20.*" numpy pillow tflite
#   ./fetch.sh && .venv-build/bin/python mkresnet18.py && python3 mknet.py r18
#
# mknet.py itself needs `tflite` (the schema) and `ai-edge-litert` (the reference
# interpreter), plus numpy and Pillow:
#
#   pip install --target=./pylibs numpy Pillow tflite ai-edge-litert
#   PYTHONPATH=./pylibs python3 mknet.py
set -e
cd "$(dirname "$0")"

BASE=https://raw.githubusercontent.com/google-coral/test_data/master

for MODEL in mobilenet_v1_1.0_224_quant.tflite mobilenet_v2_1.0_224_quant.tflite \
             inception_v1_224_quant.tflite inception_v3_299_quant.tflite; do
  [ -f "$MODEL" ] || curl -fsSL -o "$MODEL" "$BASE/$MODEL"
done
[ -f labels.txt ] || curl -fsSL -o labels.txt "$BASE/imagenet_labels.txt"
[ -f grace_hopper.bmp ] || curl -fsSL -o grace_hopper.bmp \
  https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/lite/examples/label_image/testdata/grace_hopper.bmp

# ---- ResNet-18: the float checkpoint, and the photographs that calibrate it ---------
# timm/resnet18.a1_in1k carries torchvision's ResNet-18 parameter layout under
# torchvision's own names, as safetensors — a JSON header and a raw tensor blob, which
# mkresnet18.py reads without a framework.
[ -f resnet18_a1_in1k.safetensors ] || curl -fsSL -o resnet18_a1_in1k.safetensors \
  https://huggingface.co/timm/resnet18.a1_in1k/resolve/main/model.safetensors

mkdir -p calib
for IMG in bird.bmp cat.bmp dragonfly.bmp grace_hopper.bmp hot_dog.jpg owl.jpg \
           parrot.jpg pets.jpg sunflower.bmp squat.bmp face.jpg kite_and_cold.jpg \
           cat_720p.jpg bird_segmentation.bmp dog_segmentation.bmp \
           missvickie_potato_chips.bmp; do
  [ -f "calib/$IMG" ] || curl -fsSL -o "calib/$IMG" "$BASE/$IMG"
done

ls -l mobilenet_v1_1.0_224_quant.tflite mobilenet_v2_1.0_224_quant.tflite \
      inception_v1_224_quant.tflite inception_v3_299_quant.tflite \
      resnet18_a1_in1k.safetensors labels.txt grace_hopper.bmp
ls calib | wc -l | sed 's/^/calibration images: /'
