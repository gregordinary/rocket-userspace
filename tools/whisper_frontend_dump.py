#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The rocket-userspace authors
"""Dump a Whisper model's two front-end convs and a 30 s log-mel, for
tests/conv1d_gelu_epilogue_probe.c.

    whisper_frontend_dump.py MODEL.bin AUDIO.wav OUTDIR

MODEL.bin is a whisper.cpp model in the legacy ggml format (ggml-base.bin, ggml-tiny.bin).
AUDIO.wav is 16 kHz mono 16-bit PCM. OUTDIR receives, little-endian:

    mel.f16   [n_mels][3000]           the log-mel the encoder reads
    w1.f16    [n_state][n_mels][3]     encoder.conv1.weight
    b1.f32    [n_state]                encoder.conv1.bias
    w2.f16    [n_state][n_state][3]    encoder.conv2.weight
    b2.f32    [n_state]                encoder.conv2.bias
    meta.txt  "n_mels n_state n_frames"

The mel follows whisper.cpp's recipe: 30 s of zeros appended, a reflected half window in
front, a periodic 400-sample Hann window at a hop of 160, the power spectrum through the
model's own mel filters, log10 floored at 1e-10, clamped to 8 below its maximum and mapped
by (x + 4) / 4. It is a real encoder input, not a bit-exact copy of whisper.cpp's.
Needs numpy.
"""
import os
import struct
import sys
import wave

import numpy as np

N_FFT, HOP, SR, N_FRAMES = 400, 160, 16000, 3000
WANT = ("encoder.conv1.weight", "encoder.conv1.bias",
        "encoder.conv2.weight", "encoder.conv2.bias")


def read_model(path):
    with open(path, "rb") as f:
        (magic,) = struct.unpack("<I", f.read(4))
        if magic != 0x67676D6C:
            sys.exit(f"{path}: not a legacy ggml whisper model (magic {magic:#x})")
        hp = struct.unpack("<11i", f.read(44))
        n_state, n_mels = hp[2], hp[9]
        n_mel, n_fft = struct.unpack("<2i", f.read(8))
        filters = np.frombuffer(f.read(4 * n_mel * n_fft), dtype="<f4").reshape(n_mel, n_fft)
        (n_vocab,) = struct.unpack("<i", f.read(4))
        for _ in range(n_vocab):
            (ln,) = struct.unpack("<I", f.read(4))
            f.seek(ln, os.SEEK_CUR)
        tensors = {}
        while len(tensors) < len(WANT):
            hdr = f.read(12)
            if len(hdr) < 12:
                sys.exit(f"{path}: missing {set(WANT) - set(tensors)}")
            n_dims, name_len, ttype = struct.unpack("<3i", hdr)
            ne = struct.unpack(f"<{n_dims}i", f.read(4 * n_dims))
            name = f.read(name_len).decode()
            count = int(np.prod(ne))
            if ttype not in (0, 1):
                sys.exit(f"{name}: tensor type {ttype} is neither f32 nor f16")
            size = count * (4 if ttype == 0 else 2)
            if name in WANT:
                dt = "<f4" if ttype == 0 else "<f2"
                # ggml's ne[0] is the fastest axis: [kw, ic, oc] is C order [oc][ic][kw].
                tensors[name] = np.frombuffer(f.read(size), dtype=dt).reshape(ne[::-1])
            else:
                f.seek(size, os.SEEK_CUR)
    return n_mels, n_state, filters, tensors


def log_mel(wav_path, filters):
    with wave.open(wav_path, "rb") as w:
        if w.getframerate() != SR or w.getnchannels() != 1 or w.getsampwidth() != 2:
            sys.exit(f"{wav_path}: want 16 kHz mono 16-bit PCM")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    x = pcm.astype(np.float32) / 32768.0
    x = np.concatenate([x[1:N_FFT // 2 + 1][::-1], x, np.zeros(SR * 30 + N_FFT, np.float32)])
    win = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(N_FFT) / N_FFT))
    frames = np.stack([x[i * HOP:i * HOP + N_FFT] * win for i in range(N_FRAMES)])
    power = np.abs(np.fft.rfft(frames, n=N_FFT, axis=1)) ** 2          # [3000][201]
    mel = np.log10(np.maximum(filters @ power.T, 1e-10))                # [n_mel][3000]
    mel = np.maximum(mel, mel.max() - 8.0)
    return ((mel + 4.0) / 4.0).astype(np.float32)


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    model, wav, out = sys.argv[1:]
    n_mels, n_state, filters, t = read_model(model)
    mel = log_mel(wav, filters)
    os.makedirs(out, exist_ok=True)
    mel.astype("<f2").tofile(os.path.join(out, "mel.f16"))
    t["encoder.conv1.weight"].astype("<f2").reshape(n_state, n_mels, 3).tofile(os.path.join(out, "w1.f16"))
    t["encoder.conv1.bias"].astype("<f4").reshape(n_state).tofile(os.path.join(out, "b1.f32"))
    t["encoder.conv2.weight"].astype("<f2").reshape(n_state, n_state, 3).tofile(os.path.join(out, "w2.f16"))
    t["encoder.conv2.bias"].astype("<f4").reshape(n_state).tofile(os.path.join(out, "b2.f32"))
    with open(os.path.join(out, "meta.txt"), "w") as f:
        f.write(f"{n_mels} {n_state} {N_FRAMES}\n")
    print(f"mel {mel.shape} in [{mel.min():.3f}, {mel.max():.3f}], n_state {n_state}, "
          f"conv1 {t['encoder.conv1.weight'].shape} {t['encoder.conv1.weight'].dtype}, "
          f"bias {t['encoder.conv1.bias'].shape} {t['encoder.conv1.bias'].dtype}")


if __name__ == "__main__":
    main()
