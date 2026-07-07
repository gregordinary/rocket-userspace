# Teflon depthwise-conv capture (test fixtures)

Ground-truth fixtures for the native int8 depthwise-conv gates `replay_dw_mesa` and
`conv_dw_int8_runtime`. They are Mesa Teflon's captured buffers for one int8
`DEPTHWISE_CONV_2D` run, used as the oracle the gates compare against byte-for-byte.

Capture shape: `IC=64, 8×8, K3×3, stride 1, pad 1`, per-tensor int8
(in `0.03657235`/`-2`, weight `0.00079638`/`0`, out `0.00691164`/`5`).

| File | What it is |
|---|---|
| `dw_pt.tflite` | the source TFLite model (the independent filter + bias) |
| `mesa-input-000-000.bin` | Mesa's captured input feature cube (uint8-centered NC1HWC2) |
| `mesa-weights-000-000.bin` | Mesa's captured weight cube |
| `mesa-biases-000-000.bin` | Mesa's captured int32 bias cube |
| `mesa-output-000-000.bin` | Mesa's captured output cube — the oracle |

`replay_dw_mesa` feeds the captured input/weight/bias cubes verbatim and checks our
int8-out regcmd reproduces `mesa-output`. `conv_dw_int8_runtime` drives the runtime with
the raw model tensors (`dw_dump_capture.py` descatters them) and checks the same. Both
default to this directory; override with `argv[1]`.
