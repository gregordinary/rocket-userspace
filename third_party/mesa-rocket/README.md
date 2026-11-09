# Mesa's rocket register database

These four files are the RK3588 NPU register database of Mesa's `rocket` Gallium driver and
the generator that turns it into a C header. They are copied unmodified from Mesa's
`src/gallium/drivers/rocket/` at commit `e46ee71b43b1693a2baf22740b0157e7d31b79cc`
(2026-09-20). Mesa distributes them under the MIT license. `registers.xml` and
`gen_header.py` contain the notice.

| File | What it is |
|---|---|
| `registers.xml` | Every register of the PC, CNA, CORE, DPU, DPU_RDMA, PPU, PPU_RDMA and global blocks, with its offset and named bit fields |
| `rules-ng.xsd` | The schema the database follows |
| `gen_header.py`, `gen_parser.py` | The generator: `REG_<BLOCK>_<REGISTER>` offsets, and per field a `__MASK`, a `__SHIFT` and a function that shifts and masks a value into place |

`src/rkt_registers.h` is the generated header. It is committed, so a build needs no Python.
Regenerate it after changing `registers.xml`:

```sh
cd third_party/mesa-rocket
PYTHONDONTWRITEBYTECODE=1 python3 gen_header.py --rnn . --xml ./registers.xml c-defines \
    > ../../src/rkt_registers.h
```

The `--xml` path must have a directory part, or the parser looks for the schema at
`/rules-ng.xsd`. `lxml` is optional: with it the parser also validates the database against
the schema. The generated header must be included after `<stdint.h>`.

The `regmap_current` test, registered when CMake finds Python 3, regenerates the header into
the build directory and compares it with `src/rkt_registers.h`. It ignores the copyright
line, whose closing year is the year the generator ran.

`src/npu_regmap.h` is the library's view of the map. It includes the generated header and
names what the database leaves unnamed: two registers the programs write as 0, and the
trailer's per-block enable bits, which the database names only on the global register.
