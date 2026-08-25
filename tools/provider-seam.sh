#!/bin/bash
# Print librocketnpu's submit seam, or check a provider against it.
#
# The library reaches the kernel through one set of C symbols, and that set has more than
# one implementation: src/rocket_npu.c for the mainline `rocket` DRM-accel driver, and an
# external provider (rknpu-submit, for the vendor `rknpu` BSP driver) selected with
# -DROCKETNPU_PROVIDER=external. src/rocket_npu.c is compiled ONLY for the builtin
# provider, so a symbol added there and called from above the seam is undefined in every
# external build.
#
# Two things make that easy to ship. librocketnpu.a is a static archive, which is allowed
# to carry unresolved symbols, so a driver-only build is green and only the executables
# fail to link. And the obvious check runs the wrong way: verifying that every symbol the
# provider DEFINES is declared in these headers cannot see a symbol the library CALLS that
# no provider defines.
#
# THE SEAM IS EXACTLY the externally-visible rocket_* functions defined in
# src/rocket_npu.c, and a conforming provider defines exactly that set. Both halves are
# measured, not assumed: against the compiled builtin object and the shipping vendor
# provider the two sets came out equal, with nothing extra on either side. The live count
# is whatever this prints -- it is not pinned here, because the seam grows.
#
# The seam is read out of the SOURCE rather than out of a compiled object on purpose. A
# provider's own board is the place this check is most useful, and a BSP box cannot build
# the builtin provider at all -- it has no <drm/rocket_accel.h> -- so anything requiring a
# builtin build would be unrunnable exactly where it is wanted. The source read is exact:
# it reproduces `nm --defined-only --extern-only` on the built object symbol for symbol.
#
# Usage:
#   provider-seam.sh                      # print the seam, one symbol per line
#   provider-seam.sh <provider>           # check a provider against it
#
# <provider> is an archive, object or shared library (read with nm), or a .c source file
# (read the same way this reads rocket_npu.c). Exit 0 if the provider defines every seam
# symbol, 1 if any is missing. A rocket_* symbol the provider defines that is NOT in the
# seam is reported but not fatal: the library may define that name above the seam, in
# which case it is a duplicate waiting to happen, but a provider is also entitled to its
# own helpers.
#
# Override the driver tree with ROCKETNPU_SRC=<path-to-rocket-userspace>.
set -u

self_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
src_root=${ROCKETNPU_SRC:-$(dirname -- "$self_dir")}
builtin_c="$src_root/src/rocket_npu.c"

if [ ! -r "$builtin_c" ]; then
    echo "provider-seam: cannot read $builtin_c" >&2
    echo "provider-seam: set ROCKETNPU_SRC to a rocket-userspace source tree" >&2
    exit 2
fi

# Externally-visible rocket_* functions DEFINED in a C file. A definition starts at column
# 0 with its return type, so an indented call never matches; `static` is dropped because it
# is not externally visible; and a line ending in `;` is a prototype rather than a
# definition. A multi-line parameter list carries the name on its first line, which is the
# line this matches.
seam_from_source() {
    grep -E '^[A-Za-z_][A-Za-z0-9_ \t*]*\brocket_[a-z0-9_]+[ \t]*\(' "$1" \
        | grep -v '^static' \
        | sed 's/[[:space:]]*$//' \
        | grep -v ';$' \
        | grep -oE 'rocket_[a-z0-9_]+[ \t]*\(' \
        | tr -d '( \t' \
        | sort -u
}

symbols_from_binary() {
    if ! command -v nm >/dev/null 2>&1; then
        echo "provider-seam: nm not found; needed to read $1" >&2
        exit 2
    fi
    nm --defined-only --extern-only "$1" 2>/dev/null \
        | awk '{print $NF}' \
        | grep -E '^rocket_[a-z0-9_]+$' \
        | sort -u
}

seam=$(seam_from_source "$builtin_c")
if [ -z "$seam" ]; then
    echo "provider-seam: read no symbols from $builtin_c -- the source shape changed" >&2
    exit 2
fi

if [ $# -eq 0 ]; then
    printf '%s\n' "$seam"
    exit 0
fi

provider=$1
if [ ! -r "$provider" ]; then
    echo "provider-seam: cannot read provider '$provider'" >&2
    exit 2
fi

case "$provider" in
    *.c) defined=$(seam_from_source "$provider") ;;
    *)   defined=$(symbols_from_binary "$provider") ;;
esac

missing=$(comm -23 <(printf '%s\n' "$seam") <(printf '%s\n' "$defined"))
extra=$(comm -13 <(printf '%s\n' "$seam") <(printf '%s\n' "$defined"))

n_seam=$(printf '%s\n' "$seam" | grep -c .)

if [ -n "$extra" ]; then
    echo "provider-seam: $provider defines rocket_* symbols outside the seam:"
    printf '  %s\n' $extra
    echo "provider-seam: harmless if they are its own, a duplicate symbol if the library"
    echo "provider-seam: defines the same name above the seam."
fi

if [ -n "$missing" ]; then
    echo "provider-seam: FAIL -- $provider does not define $(printf '%s\n' "$missing" | grep -c .) of the seam's $n_seam symbols:" >&2
    printf '  %s\n' $missing >&2
    echo "provider-seam: every one is an undefined reference in an external-provider build." >&2
    exit 1
fi

echo "provider-seam: OK -- $provider defines all $n_seam seam symbols"
exit 0
