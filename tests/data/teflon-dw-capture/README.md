# Teflon depthwise-conv capture (test fixtures)

Fixtures for the native int8 depthwise-conv gates `replay_dw_mesa` and
`conv_dw_int8_runtime`: Mesa Teflon's captured buffers for one int8 `DEPTHWISE_CONV_2D`
run, the model it ran, and TFLite's own output for that model.

Capture shape: `IC=64, 8×8, K3×3, stride 1, pad 1`, per-tensor int8
(in `0.03657235`/`-2`, weight `0.00079638`/`0`, out `0.00691164`/`5`).

| File | What it is |
|---|---|
| `dw_pt.tflite` | the source TFLite model |
| `mesa-input-000-000.bin` | Mesa's captured input feature cube (NC1HWC2). The capture's input is constant zero |
| `mesa-weights-000-000.bin` | Mesa's captured weight cube |
| `mesa-biases-000-000.bin` | Mesa's captured int32 bias cube |
| `mesa-output-000-000.bin` | Mesa's captured output cube |
| `dw_w.bin` | the raw int8 filter `[C][KH][KW]`, from the model |
| `dw_bias.bin` | the int32 bias `[C]`, from the model |
| `litert-in.bin` | a seeded random int8 input `[C][H][W]` |
| `litert-out.bin` | TFLite's output for it `[C][H][W]`, from its reference kernels |

The capture is Mesa's program and Mesa's output, not the model's function. Mesa's rocket
driver is a uint8 driver, and Teflon handed it the model's int8 bytes and zero points as
if they were uint8. Every one of the capture's 4096 outputs differs from TFLite's for the
same model and input, by up to 133. So `replay_dw_mesa` uses the capture only as a check of
the register program: it feeds Mesa's cubes verbatim to our generator's program, with
Mesa's pad word, and compares against `mesa-output`.

`conv_dw_int8_runtime` gates the function. It runs the model's filter and bias on
`litert-in.bin` and compares against `litert-out.bin`, within one at a rounding boundary.
It also compares against an exact host model of the requant at five shapes.

`tests/dw_dump_capture.py` checks that Mesa's formulas, applied to the int8 bytes,
reproduce the captured weight and bias cubes, and writes `dw_w.bin` and `dw_bias.bin`.
`tests/dw_litert_ref.py` writes the two `litert-*` files. Both need `ai_edge_litert`,
`tflite_runtime` or `tensorflow`. Both gates default to this directory; override with
`argv[1]`.
