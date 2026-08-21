#!/usr/bin/env python3
"""Synthesize a Module.symvers for an out-of-tree build against a running kernel
whose build tree is gone.

CONFIG_MODVERSIONS=y means the loader checks a CRC per imported symbol, and the
CRCs live in the kernel build's Module.symvers -- which a linux-image .deb does
not ship. But a module BUILT by that kernel carries the CRCs it needs, in its
__versions section: 64-byte entries of an 8-byte little-endian CRC and a
56-byte NUL-padded name. Reading them back off the shipped rocket.ko gives the
exact CRC for every symbol the driver imports, which is exactly the set a
rebuild of that driver needs.

The owning module per symbol comes from /proc/kallsyms so the generated
modinfo depends= line is right; the addresses being zeroed for an unprivileged
reader does not matter, only the [module] tag does.

A rebuild that adds CODE can import a symbol the shipped module never did, and
that symbol's CRC is in no __versions section this script has read -- modpost
then fails with "undefined!" for a symbol the kernel plainly exports. Any OTHER
module built by the same kernel and importing it carries the CRC, so pass the
extra names after the output path and the module tree is searched for them:

  mksymvers.py rocket.versions /proc/kallsyms Module.symvers \
      memdup_user __dma_sync_single_for_cpu

Searching costs a pass over /lib/modules, so it only runs for names the primary
module did not already supply.
"""
import os, struct, subprocess, sys

vers_bin, kallsyms, out = sys.argv[1], sys.argv[2], sys.argv[3]
wanted = set(sys.argv[4:])

owner = {}
with open(kallsyms) as f:
    for line in f:
        p = line.split()
        if len(p) >= 3:
            name = p[2]
            mod = p[3].strip("[]") if len(p) > 3 else "vmlinux"
            owner.setdefault(name, mod)

data = open(vers_bin, "rb").read()
assert len(data) % 64 == 0, "unexpected __versions entry size"
rows = []
for off in range(0, len(data), 64):
    crc, = struct.unpack_from("<Q", data, off)
    name = data[off + 8:off + 64].split(b"\0")[0].decode()
    assert name, "empty symbol name in __versions"
    rows.append((crc, name, owner.get(name, "vmlinux")))

# Extra symbols, harvested from whatever else this kernel built. A module may be
# compressed, so the section is read through objcopy rather than off the file.
missing = wanted - {name for _, name, _ in rows}
if missing:
    root = "/lib/modules/%s" % os.uname().release
    for dirpath, _, files in os.walk(root):
        if not missing:
            break
        for fn in files:
            if not fn.startswith("rocket") and ".ko" in fn:
                path = os.path.join(dirpath, fn)
                try:
                    blob = subprocess.run(
                        ["objcopy", "-O", "binary", "--only-section=__versions",
                         path, "/dev/stdout"],
                        capture_output=True, check=True).stdout
                except Exception:
                    continue
                for off in range(0, len(blob) - 63, 64):
                    crc, = struct.unpack_from("<Q", blob, off)
                    name = blob[off + 8:off + 64].split(b"\0")[0].decode(errors="replace")
                    if name in missing:
                        rows.append((crc, name, owner.get(name, "vmlinux")))
                        missing.discard(name)
                if not missing:
                    break
    if missing:
        sys.exit("no module on this system imports: %s" % ", ".join(sorted(missing)))

with open(out, "w") as f:
    for crc, name, mod in rows:
        # crc, symbol, owning module, export type, namespace. Every symbol here
        # was linked into a GPL module, so GPL is the safe declaration.
        f.write("0x%08x\t%s\t%s\tEXPORT_SYMBOL_GPL\t\n" % (crc & 0xffffffff, name, mod))

print("%d symbols, %d unresolved owners" %
      (len(rows), sum(1 for _, n, _ in rows if n not in owner)))
