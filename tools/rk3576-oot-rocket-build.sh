#!/bin/bash
# Build the RK3576 `rocket` driver out of tree, so a driver change can be A/B'd on a board
# whose kernel source is not installed.
#
# The board ships `rocket` as a MODULE (CONFIG_DRM_ACCEL_ROCKET=m) built from a tree with
# the rk3576 series already in it, and the distro's linux-headers package carries
# Module.symvers. That is everything an out-of-tree build needs; no kernel source and no
# harvested CRCs.
#
# The source is reconstructed rather than copied: pristine mainline v7.1
# drivers/accel/rocket/ plus patches 0006 and 0008-0024 of patches/rk3576/npu. All 18 apply
# clean, and the uapi header that falls out is byte-identical to the one the headers package
# installs -- which is the check that the reconstruction is the source the running module
# was built from. The behavioural check is the gate list: an unmodified rebuild is 58 of 58
# rc=0 at taint 4096, against the shipped module's own run.
#
#   ./rk3576-oot-rocket-build.sh <patches-dir> <outdir> [extra.patch ...]
#
# Then, on the board:
#   sudo rmmod rocket && sudo insmod <outdir>/rocket.ko
#   echo 27708000.npu | sudo tee /sys/bus/platform/drivers/rocket/unbind   # ALWAYS
#   ... and `sudo rmmod rocket && sudo modprobe rocket` to put the shipped one back.
#
# TRAPS
#  - Core 1 binds on every insmod. Two jobs in flight compute wrong answers, so unbind it
#    every time, not just after a reboot.
#  - The uapi header must be OURS and it includes "drm.h" by quotes, so the whole installed
#    include/uapi/drm tree has to sit beside it with our rocket_accel.h laid over the top.
#    Shadowing rocket_accel.h alone fails to compile on drm.h.
#  - Loading it does not move the taint word on this image (4096 either way), so taint is
#    not how you tell which module is live. `dmesg | grep "Initialized rocket"` is -- the
#    interface version is the marker.
set -e
PATCHDIR=$(readlink -f "${1:?usage: $0 <patches/rk3576/npu> <outdir> [extra.patch ...]}")
OUT=${2:?usage: $0 <patches/rk3576/npu> <outdir> [extra.patch ...]}
shift 2
# The patch calls run from inside the source tree, so every path handed to them must be
# absolute -- a relative one silently resolves against the wrong directory.
EXTRA=(); for e in "$@"; do EXTRA+=("$(readlink -f "$e")"); done
mkdir -p "$OUT"; OUT=$(readlink -f "$OUT")
BASE=https://raw.githubusercontent.com/torvalds/linux/v7.1

rm -rf "$OUT" && mkdir -p "$OUT/drm" "$OUT/.src/drivers/accel/rocket" "$OUT/.src/include/uapi/drm"
for f in Kconfig Makefile rocket_core.c rocket_core.h rocket_device.c rocket_device.h \
         rocket_drv.c rocket_drv.h rocket_gem.c rocket_gem.h rocket_job.c rocket_job.h \
         rocket_registers.h; do
    curl -sfo "$OUT/.src/drivers/accel/rocket/$f" "$BASE/drivers/accel/rocket/$f"
done
curl -sfo "$OUT/.src/include/uapi/drm/rocket_accel.h" "$BASE/include/uapi/drm/rocket_accel.h"

for n in 0006 0008 0009 0010 0011 0012 0013 0014 0015 0016 0017 0018 0019 0020 0021 0022 \
         0023 0024; do
    p=$(ls "$PATCHDIR"/$n-*.patch)
    (cd "$OUT/.src" && patch -p1 -s --no-backup-if-mismatch < "$p")
done
for extra in ${EXTRA[@]+"${EXTRA[@]}"}; do
    (cd "$OUT/.src" && patch -p1 -s --no-backup-if-mismatch < "$extra")
done

cp "$OUT"/.src/drivers/accel/rocket/*.c "$OUT"/.src/drivers/accel/rocket/*.h "$OUT/"
cp "$OUT/.src/include/uapi/drm/rocket_accel.h" "$OUT/drm/"

cat > "$OUT/Kbuild" <<'EOF'
# The distro headers carry a rocket_accel.h of their own; ours must come first.
LINUXINCLUDE := -I$(src) $(LINUXINCLUDE)
obj-m := rocket.o
rocket-y := rocket_core.o rocket_device.o rocket_drv.o rocket_gem.o rocket_job.o
EOF
cat > "$OUT/Makefile" <<'EOF'
KDIR ?= /lib/modules/$(shell uname -r)/build
all:
	$(MAKE) -C $(KDIR) M=$(CURDIR) modules
clean:
	$(MAKE) -C $(KDIR) M=$(CURDIR) clean
EOF

cat <<EOF

Built the source tree at $OUT. On the BOARD, before the first make:

  H=/lib/modules/\$(uname -r)/build/include/uapi/drm
  cp \$H/*.h $OUT/drm/ && cp $OUT/.src/include/uapi/drm/rocket_accel.h $OUT/drm/
  diff \$H/rocket_accel.h $OUT/drm/rocket_accel.h   # identical == right base
  make -C $OUT -j\$(nproc)
EOF
