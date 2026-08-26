/* Which kernel driver this build reaches, as a short human name.
 *
 * NOT part of the submit seam: the seam is exactly what src/rocket_npu.c defines, and
 * this lives here so that adding it obliges no provider to define anything. The answer
 * is a build-time one because the provider is a build-time choice -- one build targets
 * one driver -- and no ioctl is asked, so a caller may print it before opening a device.
 *
 * The library cannot name an external provider's driver for it, so the default there
 * says only what is known. -DROCKETNPU_DRIVER_NAME=... supplies the specific name. */

#ifndef ROCKETNPU_DRIVER_NAME
#define ROCKETNPU_DRIVER_NAME "external submit provider"
#endif

const char *rocket_driver_name(void)
{
    return ROCKETNPU_DRIVER_NAME;
}
