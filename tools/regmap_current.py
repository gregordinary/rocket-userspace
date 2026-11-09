#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 The rocket-userspace authors
"""Checks that src/rkt_registers.h is what third_party/mesa-rocket generates today.

usage: regmap_current.py SOURCE_DIR OUT_DIR

Runs Mesa's gen_header.py over registers.xml into OUT_DIR and compares the result with the
committed src/rkt_registers.h, ignoring the generator's copyright line, whose closing year is
the year it ran. Exits 0 when they match, 1 with a diff when they do not.
"""
import difflib
import os
import re
import subprocess
import sys

COPYRIGHT = re.compile(r"^Copyright \(C\) [0-9]{4}-[0-9]{4} by the following authors:$")


def lines(text):
    return [l for l in text.splitlines() if not COPYRIGHT.match(l)]


def main():
    src, out = sys.argv[1], sys.argv[2]
    regdir = os.path.join(src, "third_party", "mesa-rocket")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    gen = subprocess.run(
        [sys.executable, "gen_header.py", "--rnn", ".", "--xml", "./registers.xml", "c-defines"],
        cwd=regdir, env=env, capture_output=True, text=True)
    if gen.returncode != 0:
        print(gen.stderr, file=sys.stderr)
        return 1
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "rkt_registers.h"), "w") as f:
        f.write(gen.stdout)
    with open(os.path.join(src, "src", "rkt_registers.h")) as f:
        committed = f.read()
    want, got = lines(gen.stdout), lines(committed)
    if want == got:
        print("src/rkt_registers.h matches registers.xml")
        return 0
    sys.stdout.writelines(l + "\n" for l in list(difflib.unified_diff(
        got, want, "src/rkt_registers.h", "generated", lineterm=""))[:60])
    print("src/rkt_registers.h is stale: regenerate it (third_party/mesa-rocket/README.md)")
    return 1


sys.exit(main())
