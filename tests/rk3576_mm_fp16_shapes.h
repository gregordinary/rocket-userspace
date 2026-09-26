/* The shapes tests/rk3576_mm_fp16_gate.c runs, as { m, k, n }, shared with the golden
 * generator tools/rk3576-mm-fp16-golden.c. The first six are GA-5.1's; the rest move
 * one axis each (M, then a K and an N that are not powers of two, then N), and the
 * next three were the corners of the first envelope. The last five are one task at the
 * bounds tests/rk3576_mm_fp16_envelope.c measured: the data window at K 1536 and at
 * K 256, K 6144, N 8192, and an 8 MiB weight at the window's M. */
{ 1, 32, 64 }, { 1, 64, 64 }, { 1, 256, 64 }, { 1, 1024, 64 }, { 1, 2048, 64 }, { 4, 256, 64 },
{ 2, 256, 64 }, { 3, 256, 64 }, { 8, 256, 64 }, { 16, 256, 64 },
{ 1, 96, 64 }, { 3, 160, 80 },
{ 1, 256, 128 }, { 1, 256, 256 }, { 4, 512, 256 },
{ 16, 2048, 64 }, { 1, 2048, 256 }, { 16, 2048, 256 },
{ 85, 1536, 64 }, { 512, 256, 64 }, { 21, 6144, 256 }, { 16, 256, 8192 }, { 64, 2048, 2048 },
