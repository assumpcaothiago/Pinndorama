#include "BHaH_defines.h"
#include "NRPYELL_binary_format.h"
#include "intrinsics/simd_intrinsics.h"

/**
 * @file NRPYELL_derivatives_all_points.c
 * @brief Export selected finite-difference derivatives of the NRPyElliptic solution.
 *
 * The generated function "NRPYELL_derivatives_all_points" iterates over the grid
 * interior and evaluates derivatives of uu with respect to the compactified
 * numerical coordinates xx. Results are written into the enriched
 * NRPYELL_solution.bin field buffer. Ghost-zone entries are intentionally left to
 * the caller.
 */
void NRPYELL_derivatives_all_points(const commondata_struct *restrict commondata, const params_struct *restrict params, const REAL *restrict in_gfs,
                                    REAL *restrict NRPYELL_fields) {
#include "set_CodeParameters-simd.h"
#pragma omp parallel for collapse(2)
  for (int i2 = NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2++) {
    for (int i1 = NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1++) {
      for (int i0 = NGHOSTS; i0 < Nxx_plus_2NGHOSTS0 - NGHOSTS; i0 += simd_width) {

        /*
         * NRPy-Generated GF Access/FD Code, Step 1 of 2:
         * Read gridfunction(s) from main memory and compute FD stencils as needed.
         */
        const REAL_SIMD_ARRAY uu_i0m5_i1m5 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 5, i1 - 5, i2)]);
        const REAL_SIMD_ARRAY uu_i0m4_i1m5 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 4, i1 - 5, i2)]);
        const REAL_SIMD_ARRAY uu_i0m3_i1m5 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 3, i1 - 5, i2)]);
        const REAL_SIMD_ARRAY uu_i0m2_i1m5 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 2, i1 - 5, i2)]);
        const REAL_SIMD_ARRAY uu_i0m1_i1m5 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 1, i1 - 5, i2)]);
        const REAL_SIMD_ARRAY uu_i1m5 = ReadSIMD(&in_gfs[IDX4(UUGF, i0, i1 - 5, i2)]);
        const REAL_SIMD_ARRAY uu_i0p1_i1m5 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 1, i1 - 5, i2)]);
        const REAL_SIMD_ARRAY uu_i0p2_i1m5 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 2, i1 - 5, i2)]);
        const REAL_SIMD_ARRAY uu_i0p3_i1m5 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 3, i1 - 5, i2)]);
        const REAL_SIMD_ARRAY uu_i0p4_i1m5 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 4, i1 - 5, i2)]);
        const REAL_SIMD_ARRAY uu_i0p5_i1m5 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 5, i1 - 5, i2)]);
        const REAL_SIMD_ARRAY uu_i0m5_i1m4 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 5, i1 - 4, i2)]);
        const REAL_SIMD_ARRAY uu_i0m4_i1m4 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 4, i1 - 4, i2)]);
        const REAL_SIMD_ARRAY uu_i0m3_i1m4 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 3, i1 - 4, i2)]);
        const REAL_SIMD_ARRAY uu_i0m2_i1m4 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 2, i1 - 4, i2)]);
        const REAL_SIMD_ARRAY uu_i0m1_i1m4 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 1, i1 - 4, i2)]);
        const REAL_SIMD_ARRAY uu_i1m4 = ReadSIMD(&in_gfs[IDX4(UUGF, i0, i1 - 4, i2)]);
        const REAL_SIMD_ARRAY uu_i0p1_i1m4 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 1, i1 - 4, i2)]);
        const REAL_SIMD_ARRAY uu_i0p2_i1m4 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 2, i1 - 4, i2)]);
        const REAL_SIMD_ARRAY uu_i0p3_i1m4 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 3, i1 - 4, i2)]);
        const REAL_SIMD_ARRAY uu_i0p4_i1m4 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 4, i1 - 4, i2)]);
        const REAL_SIMD_ARRAY uu_i0p5_i1m4 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 5, i1 - 4, i2)]);
        const REAL_SIMD_ARRAY uu_i0m5_i1m3 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 5, i1 - 3, i2)]);
        const REAL_SIMD_ARRAY uu_i0m4_i1m3 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 4, i1 - 3, i2)]);
        const REAL_SIMD_ARRAY uu_i0m3_i1m3 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 3, i1 - 3, i2)]);
        const REAL_SIMD_ARRAY uu_i0m2_i1m3 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 2, i1 - 3, i2)]);
        const REAL_SIMD_ARRAY uu_i0m1_i1m3 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 1, i1 - 3, i2)]);
        const REAL_SIMD_ARRAY uu_i1m3 = ReadSIMD(&in_gfs[IDX4(UUGF, i0, i1 - 3, i2)]);
        const REAL_SIMD_ARRAY uu_i0p1_i1m3 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 1, i1 - 3, i2)]);
        const REAL_SIMD_ARRAY uu_i0p2_i1m3 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 2, i1 - 3, i2)]);
        const REAL_SIMD_ARRAY uu_i0p3_i1m3 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 3, i1 - 3, i2)]);
        const REAL_SIMD_ARRAY uu_i0p4_i1m3 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 4, i1 - 3, i2)]);
        const REAL_SIMD_ARRAY uu_i0p5_i1m3 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 5, i1 - 3, i2)]);
        const REAL_SIMD_ARRAY uu_i0m5_i1m2 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 5, i1 - 2, i2)]);
        const REAL_SIMD_ARRAY uu_i0m4_i1m2 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 4, i1 - 2, i2)]);
        const REAL_SIMD_ARRAY uu_i0m3_i1m2 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 3, i1 - 2, i2)]);
        const REAL_SIMD_ARRAY uu_i0m2_i1m2 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 2, i1 - 2, i2)]);
        const REAL_SIMD_ARRAY uu_i0m1_i1m2 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 1, i1 - 2, i2)]);
        const REAL_SIMD_ARRAY uu_i1m2 = ReadSIMD(&in_gfs[IDX4(UUGF, i0, i1 - 2, i2)]);
        const REAL_SIMD_ARRAY uu_i0p1_i1m2 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 1, i1 - 2, i2)]);
        const REAL_SIMD_ARRAY uu_i0p2_i1m2 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 2, i1 - 2, i2)]);
        const REAL_SIMD_ARRAY uu_i0p3_i1m2 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 3, i1 - 2, i2)]);
        const REAL_SIMD_ARRAY uu_i0p4_i1m2 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 4, i1 - 2, i2)]);
        const REAL_SIMD_ARRAY uu_i0p5_i1m2 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 5, i1 - 2, i2)]);
        const REAL_SIMD_ARRAY uu_i0m5_i1m1 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 5, i1 - 1, i2)]);
        const REAL_SIMD_ARRAY uu_i0m4_i1m1 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 4, i1 - 1, i2)]);
        const REAL_SIMD_ARRAY uu_i0m3_i1m1 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 3, i1 - 1, i2)]);
        const REAL_SIMD_ARRAY uu_i0m2_i1m1 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 2, i1 - 1, i2)]);
        const REAL_SIMD_ARRAY uu_i0m1_i1m1 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 1, i1 - 1, i2)]);
        const REAL_SIMD_ARRAY uu_i1m1 = ReadSIMD(&in_gfs[IDX4(UUGF, i0, i1 - 1, i2)]);
        const REAL_SIMD_ARRAY uu_i0p1_i1m1 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 1, i1 - 1, i2)]);
        const REAL_SIMD_ARRAY uu_i0p2_i1m1 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 2, i1 - 1, i2)]);
        const REAL_SIMD_ARRAY uu_i0p3_i1m1 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 3, i1 - 1, i2)]);
        const REAL_SIMD_ARRAY uu_i0p4_i1m1 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 4, i1 - 1, i2)]);
        const REAL_SIMD_ARRAY uu_i0p5_i1m1 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 5, i1 - 1, i2)]);
        const REAL_SIMD_ARRAY uu_i0m5 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 5, i1, i2)]);
        const REAL_SIMD_ARRAY uu_i0m4 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 4, i1, i2)]);
        const REAL_SIMD_ARRAY uu_i0m3 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 3, i1, i2)]);
        const REAL_SIMD_ARRAY uu_i0m2 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 2, i1, i2)]);
        const REAL_SIMD_ARRAY uu_i0m1 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 1, i1, i2)]);
        const REAL_SIMD_ARRAY uu = ReadSIMD(&in_gfs[IDX4(UUGF, i0, i1, i2)]);
        const REAL_SIMD_ARRAY uu_i0p1 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 1, i1, i2)]);
        const REAL_SIMD_ARRAY uu_i0p2 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 2, i1, i2)]);
        const REAL_SIMD_ARRAY uu_i0p3 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 3, i1, i2)]);
        const REAL_SIMD_ARRAY uu_i0p4 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 4, i1, i2)]);
        const REAL_SIMD_ARRAY uu_i0p5 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 5, i1, i2)]);
        const REAL_SIMD_ARRAY uu_i0m5_i1p1 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 5, i1 + 1, i2)]);
        const REAL_SIMD_ARRAY uu_i0m4_i1p1 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 4, i1 + 1, i2)]);
        const REAL_SIMD_ARRAY uu_i0m3_i1p1 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 3, i1 + 1, i2)]);
        const REAL_SIMD_ARRAY uu_i0m2_i1p1 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 2, i1 + 1, i2)]);
        const REAL_SIMD_ARRAY uu_i0m1_i1p1 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 1, i1 + 1, i2)]);
        const REAL_SIMD_ARRAY uu_i1p1 = ReadSIMD(&in_gfs[IDX4(UUGF, i0, i1 + 1, i2)]);
        const REAL_SIMD_ARRAY uu_i0p1_i1p1 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 1, i1 + 1, i2)]);
        const REAL_SIMD_ARRAY uu_i0p2_i1p1 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 2, i1 + 1, i2)]);
        const REAL_SIMD_ARRAY uu_i0p3_i1p1 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 3, i1 + 1, i2)]);
        const REAL_SIMD_ARRAY uu_i0p4_i1p1 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 4, i1 + 1, i2)]);
        const REAL_SIMD_ARRAY uu_i0p5_i1p1 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 5, i1 + 1, i2)]);
        const REAL_SIMD_ARRAY uu_i0m5_i1p2 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 5, i1 + 2, i2)]);
        const REAL_SIMD_ARRAY uu_i0m4_i1p2 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 4, i1 + 2, i2)]);
        const REAL_SIMD_ARRAY uu_i0m3_i1p2 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 3, i1 + 2, i2)]);
        const REAL_SIMD_ARRAY uu_i0m2_i1p2 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 2, i1 + 2, i2)]);
        const REAL_SIMD_ARRAY uu_i0m1_i1p2 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 1, i1 + 2, i2)]);
        const REAL_SIMD_ARRAY uu_i1p2 = ReadSIMD(&in_gfs[IDX4(UUGF, i0, i1 + 2, i2)]);
        const REAL_SIMD_ARRAY uu_i0p1_i1p2 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 1, i1 + 2, i2)]);
        const REAL_SIMD_ARRAY uu_i0p2_i1p2 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 2, i1 + 2, i2)]);
        const REAL_SIMD_ARRAY uu_i0p3_i1p2 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 3, i1 + 2, i2)]);
        const REAL_SIMD_ARRAY uu_i0p4_i1p2 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 4, i1 + 2, i2)]);
        const REAL_SIMD_ARRAY uu_i0p5_i1p2 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 5, i1 + 2, i2)]);
        const REAL_SIMD_ARRAY uu_i0m5_i1p3 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 5, i1 + 3, i2)]);
        const REAL_SIMD_ARRAY uu_i0m4_i1p3 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 4, i1 + 3, i2)]);
        const REAL_SIMD_ARRAY uu_i0m3_i1p3 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 3, i1 + 3, i2)]);
        const REAL_SIMD_ARRAY uu_i0m2_i1p3 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 2, i1 + 3, i2)]);
        const REAL_SIMD_ARRAY uu_i0m1_i1p3 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 1, i1 + 3, i2)]);
        const REAL_SIMD_ARRAY uu_i1p3 = ReadSIMD(&in_gfs[IDX4(UUGF, i0, i1 + 3, i2)]);
        const REAL_SIMD_ARRAY uu_i0p1_i1p3 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 1, i1 + 3, i2)]);
        const REAL_SIMD_ARRAY uu_i0p2_i1p3 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 2, i1 + 3, i2)]);
        const REAL_SIMD_ARRAY uu_i0p3_i1p3 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 3, i1 + 3, i2)]);
        const REAL_SIMD_ARRAY uu_i0p4_i1p3 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 4, i1 + 3, i2)]);
        const REAL_SIMD_ARRAY uu_i0p5_i1p3 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 5, i1 + 3, i2)]);
        const REAL_SIMD_ARRAY uu_i0m5_i1p4 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 5, i1 + 4, i2)]);
        const REAL_SIMD_ARRAY uu_i0m4_i1p4 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 4, i1 + 4, i2)]);
        const REAL_SIMD_ARRAY uu_i0m3_i1p4 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 3, i1 + 4, i2)]);
        const REAL_SIMD_ARRAY uu_i0m2_i1p4 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 2, i1 + 4, i2)]);
        const REAL_SIMD_ARRAY uu_i0m1_i1p4 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 1, i1 + 4, i2)]);
        const REAL_SIMD_ARRAY uu_i1p4 = ReadSIMD(&in_gfs[IDX4(UUGF, i0, i1 + 4, i2)]);
        const REAL_SIMD_ARRAY uu_i0p1_i1p4 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 1, i1 + 4, i2)]);
        const REAL_SIMD_ARRAY uu_i0p2_i1p4 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 2, i1 + 4, i2)]);
        const REAL_SIMD_ARRAY uu_i0p3_i1p4 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 3, i1 + 4, i2)]);
        const REAL_SIMD_ARRAY uu_i0p4_i1p4 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 4, i1 + 4, i2)]);
        const REAL_SIMD_ARRAY uu_i0p5_i1p4 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 5, i1 + 4, i2)]);
        const REAL_SIMD_ARRAY uu_i0m5_i1p5 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 5, i1 + 5, i2)]);
        const REAL_SIMD_ARRAY uu_i0m4_i1p5 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 4, i1 + 5, i2)]);
        const REAL_SIMD_ARRAY uu_i0m3_i1p5 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 3, i1 + 5, i2)]);
        const REAL_SIMD_ARRAY uu_i0m2_i1p5 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 2, i1 + 5, i2)]);
        const REAL_SIMD_ARRAY uu_i0m1_i1p5 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 - 1, i1 + 5, i2)]);
        const REAL_SIMD_ARRAY uu_i1p5 = ReadSIMD(&in_gfs[IDX4(UUGF, i0, i1 + 5, i2)]);
        const REAL_SIMD_ARRAY uu_i0p1_i1p5 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 1, i1 + 5, i2)]);
        const REAL_SIMD_ARRAY uu_i0p2_i1p5 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 2, i1 + 5, i2)]);
        const REAL_SIMD_ARRAY uu_i0p3_i1p5 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 3, i1 + 5, i2)]);
        const REAL_SIMD_ARRAY uu_i0p4_i1p5 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 4, i1 + 5, i2)]);
        const REAL_SIMD_ARRAY uu_i0p5_i1p5 = ReadSIMD(&in_gfs[IDX4(UUGF, i0 + 5, i1 + 5, i2)]);
        static const double dblFDPart1_NegativeOne_ = -1.0;
        MAYBE_UNUSED const REAL_SIMD_ARRAY FDPart1_NegativeOne_ = ConstSIMD(dblFDPart1_NegativeOne_);

        static const double dblFDPart1_Rational_1_1260 = 1.0 / 1260.0;
        const REAL_SIMD_ARRAY FDPart1_Rational_1_1260 = ConstSIMD(dblFDPart1_Rational_1_1260);

        static const double dblFDPart1_Rational_1_127008 = 1.0 / 127008.0;
        const REAL_SIMD_ARRAY FDPart1_Rational_1_127008 = ConstSIMD(dblFDPart1_Rational_1_127008);

        static const double dblFDPart1_Rational_1_1512 = 1.0 / 1512.0;
        const REAL_SIMD_ARRAY FDPart1_Rational_1_1512 = ConstSIMD(dblFDPart1_Rational_1_1512);

        static const double dblFDPart1_Rational_1_1587600 = 1.0 / 1587600.0;
        const REAL_SIMD_ARRAY FDPart1_Rational_1_1587600 = ConstSIMD(dblFDPart1_Rational_1_1587600);

        static const double dblFDPart1_Rational_1_21168 = 1.0 / 21168.0;
        const REAL_SIMD_ARRAY FDPart1_Rational_1_21168 = ConstSIMD(dblFDPart1_Rational_1_21168);

        static const double dblFDPart1_Rational_1_3150 = 1.0 / 3150.0;
        const REAL_SIMD_ARRAY FDPart1_Rational_1_3150 = ConstSIMD(dblFDPart1_Rational_1_3150);

        static const double dblFDPart1_Rational_1_5292 = 1.0 / 5292.0;
        const REAL_SIMD_ARRAY FDPart1_Rational_1_5292 = ConstSIMD(dblFDPart1_Rational_1_5292);

        static const double dblFDPart1_Rational_25_10584 = 25.0 / 10584.0;
        const REAL_SIMD_ARRAY FDPart1_Rational_25_10584 = ConstSIMD(dblFDPart1_Rational_25_10584);

        static const double dblFDPart1_Rational_25_126 = 25.0 / 126.0;
        const REAL_SIMD_ARRAY FDPart1_Rational_25_126 = ConstSIMD(dblFDPart1_Rational_25_126);

        static const double dblFDPart1_Rational_25_1764 = 25.0 / 1764.0;
        const REAL_SIMD_ARRAY FDPart1_Rational_25_1764 = ConstSIMD(dblFDPart1_Rational_25_1764);

        static const double dblFDPart1_Rational_25_254016 = 25.0 / 254016.0;
        const REAL_SIMD_ARRAY FDPart1_Rational_25_254016 = ConstSIMD(dblFDPart1_Rational_25_254016);

        static const double dblFDPart1_Rational_25_3024 = 25.0 / 3024.0;
        const REAL_SIMD_ARRAY FDPart1_Rational_25_3024 = ConstSIMD(dblFDPart1_Rational_25_3024);

        static const double dblFDPart1_Rational_25_36 = 25.0 / 36.0;
        const REAL_SIMD_ARRAY FDPart1_Rational_25_36 = ConstSIMD(dblFDPart1_Rational_25_36);

        static const double dblFDPart1_Rational_25_42336 = 25.0 / 42336.0;
        const REAL_SIMD_ARRAY FDPart1_Rational_25_42336 = ConstSIMD(dblFDPart1_Rational_25_42336);

        static const double dblFDPart1_Rational_25_441 = 25.0 / 441.0;
        const REAL_SIMD_ARRAY FDPart1_Rational_25_441 = ConstSIMD(dblFDPart1_Rational_25_441);

        static const double dblFDPart1_Rational_25_504 = 25.0 / 504.0;
        const REAL_SIMD_ARRAY FDPart1_Rational_25_504 = ConstSIMD(dblFDPart1_Rational_25_504);

        static const double dblFDPart1_Rational_25_7056 = 25.0 / 7056.0;
        const REAL_SIMD_ARRAY FDPart1_Rational_25_7056 = ConstSIMD(dblFDPart1_Rational_25_7056);

        static const double dblFDPart1_Rational_5269_1800 = 5269.0 / 1800.0;
        const REAL_SIMD_ARRAY FDPart1_Rational_5269_1800 = ConstSIMD(dblFDPart1_Rational_5269_1800);

        static const double dblFDPart1_Rational_5_1008 = 5.0 / 1008.0;
        const REAL_SIMD_ARRAY FDPart1_Rational_5_1008 = ConstSIMD(dblFDPart1_Rational_5_1008);

        static const double dblFDPart1_Rational_5_126 = 5.0 / 126.0;
        const REAL_SIMD_ARRAY FDPart1_Rational_5_126 = ConstSIMD(dblFDPart1_Rational_5_126);

        static const double dblFDPart1_Rational_5_21 = 5.0 / 21.0;
        const REAL_SIMD_ARRAY FDPart1_Rational_5_21 = ConstSIMD(dblFDPart1_Rational_5_21);

        static const double dblFDPart1_Rational_5_3 = 5.0 / 3.0;
        const REAL_SIMD_ARRAY FDPart1_Rational_5_3 = ConstSIMD(dblFDPart1_Rational_5_3);

        static const double dblFDPart1_Rational_5_504 = 5.0 / 504.0;
        const REAL_SIMD_ARRAY FDPart1_Rational_5_504 = ConstSIMD(dblFDPart1_Rational_5_504);

        static const double dblFDPart1_Rational_5_6 = 5.0 / 6.0;
        const REAL_SIMD_ARRAY FDPart1_Rational_5_6 = ConstSIMD(dblFDPart1_Rational_5_6);

        static const double dblFDPart1_Rational_5_84 = 5.0 / 84.0;
        const REAL_SIMD_ARRAY FDPart1_Rational_5_84 = ConstSIMD(dblFDPart1_Rational_5_84);

        const REAL_SIMD_ARRAY uu_dD0 = MulSIMD(
            invdxx0, FusedMulAddSIMD(FDPart1_Rational_5_504, SubSIMD(uu_i0m4, uu_i0p4),
                                     FusedMulAddSIMD(FDPart1_Rational_5_6, SubSIMD(uu_i0p1, uu_i0m1),
                                                     FusedMulAddSIMD(FDPart1_Rational_5_84, SubSIMD(uu_i0p3, uu_i0m3),
                                                                     FusedMulAddSIMD(FDPart1_Rational_1_1260, SubSIMD(uu_i0p5, uu_i0m5),
                                                                                     MulSIMD(FDPart1_Rational_5_21, SubSIMD(uu_i0m2, uu_i0p2)))))));
        const REAL_SIMD_ARRAY uu_dD1 = MulSIMD(
            invdxx1, FusedMulAddSIMD(FDPart1_Rational_5_504, SubSIMD(uu_i1m4, uu_i1p4),
                                     FusedMulAddSIMD(FDPart1_Rational_5_6, SubSIMD(uu_i1p1, uu_i1m1),
                                                     FusedMulAddSIMD(FDPart1_Rational_5_84, SubSIMD(uu_i1p3, uu_i1m3),
                                                                     FusedMulAddSIMD(FDPart1_Rational_1_1260, SubSIMD(uu_i1p5, uu_i1m5),
                                                                                     MulSIMD(FDPart1_Rational_5_21, SubSIMD(uu_i1m2, uu_i1p2)))))));
        const REAL_SIMD_ARRAY uu_dDD00 =
            MulSIMD(MulSIMD(invdxx0, invdxx0),
                    FusedMulAddSIMD(FDPart1_Rational_5_126, AddSIMD(uu_i0m3, uu_i0p3),
                                    FusedMulAddSIMD(FDPart1_Rational_5_3, AddSIMD(uu_i0m1, uu_i0p1),
                                                    FusedMulSubSIMD(FDPart1_Rational_1_3150, AddSIMD(uu_i0m5, uu_i0p5),
                                                                    FusedMulAddSIMD(FDPart1_Rational_5_1008, AddSIMD(uu_i0m4, uu_i0p4),
                                                                                    FusedMulAddSIMD(FDPart1_Rational_5_21, AddSIMD(uu_i0m2, uu_i0p2),
                                                                                                    MulSIMD(FDPart1_Rational_5269_1800, uu)))))));
        const REAL_SIMD_ARRAY uu_dDD01 = MulSIMD(
            invdxx0,
            MulSIMD(invdxx1,
                    FusedMulAddSIMD(
                        FDPart1_Rational_25_441, AddSIMD(uu_i0p2_i1p2, SubSIMD(uu_i0m2_i1m2, AddSIMD(uu_i0m2_i1p2, uu_i0p2_i1m2))),
                        FusedMulAddSIMD(
                            FDPart1_Rational_25_36, AddSIMD(uu_i0p1_i1p1, SubSIMD(uu_i0m1_i1m1, AddSIMD(uu_i0m1_i1p1, uu_i0p1_i1m1))),
                            FusedMulAddSIMD(
                                FDPart1_Rational_25_42336,
                                AddSIMD(AddSIMD(uu_i0m4_i1p3, uu_i0p3_i1m4),
                                        AddSIMD(uu_i0p4_i1m3, SubSIMD(uu_i0m3_i1p4, AddSIMD(AddSIMD(uu_i0m3_i1m4, uu_i0m4_i1m3),
                                                                                            AddSIMD(uu_i0p3_i1p4, uu_i0p4_i1p3))))),
                                FusedMulAddSIMD(
                                    FDPart1_Rational_25_254016,
                                    AddSIMD(uu_i0p4_i1p4, SubSIMD(uu_i0m4_i1m4, AddSIMD(uu_i0m4_i1p4, uu_i0p4_i1m4))),
                                    FusedMulAddSIMD(
                                        FDPart1_Rational_25_3024,
                                        AddSIMD(AddSIMD(uu_i0m4_i1p1, uu_i0p1_i1m4),
                                                AddSIMD(uu_i0p4_i1m1, SubSIMD(uu_i0m1_i1p4, AddSIMD(AddSIMD(uu_i0m1_i1m4, uu_i0m4_i1m1),
                                                                                                    AddSIMD(uu_i0p1_i1p4, uu_i0p4_i1p1))))),
                                        FusedMulAddSIMD(
                                            FDPart1_Rational_25_126,
                                            AddSIMD(AddSIMD(uu_i0m2_i1p1, uu_i0p1_i1m2),
                                                    AddSIMD(uu_i0p2_i1m1, SubSIMD(uu_i0m1_i1p2, AddSIMD(AddSIMD(uu_i0m1_i1m2, uu_i0m2_i1m1),
                                                                                                        AddSIMD(uu_i0p1_i1p2, uu_i0p2_i1p1))))),
                                            FusedMulAddSIMD(
                                                FDPart1_Rational_25_1764,
                                                AddSIMD(AddSIMD(uu_i0m3_i1p2, uu_i0p2_i1m3),
                                                        AddSIMD(uu_i0p3_i1m2, SubSIMD(uu_i0m2_i1p3, AddSIMD(AddSIMD(uu_i0m2_i1m3, uu_i0m3_i1m2),
                                                                                                            AddSIMD(uu_i0p2_i1p3, uu_i0p3_i1p2))))),
                                                FusedMulAddSIMD(
                                                    FDPart1_Rational_1_5292,
                                                    AddSIMD(
                                                        AddSIMD(uu_i0m5_i1p2, uu_i0p2_i1m5),
                                                        AddSIMD(uu_i0p5_i1m2, SubSIMD(uu_i0m2_i1p5, AddSIMD(AddSIMD(uu_i0m2_i1m5, uu_i0m5_i1m2),
                                                                                                            AddSIMD(uu_i0p2_i1p5, uu_i0p5_i1p2))))),
                                                    FusedMulAddSIMD(
                                                        FDPart1_Rational_25_10584,
                                                        AddSIMD(AddSIMD(uu_i0m4_i1m2, uu_i0p2_i1p4),
                                                                AddSIMD(uu_i0p4_i1p2,
                                                                        SubSIMD(uu_i0m2_i1m4, AddSIMD(AddSIMD(uu_i0m2_i1p4, uu_i0m4_i1p2),
                                                                                                      AddSIMD(uu_i0p2_i1m4, uu_i0p4_i1m2))))),
                                                        FusedMulAddSIMD(
                                                            FDPart1_Rational_1_1587600,
                                                            AddSIMD(uu_i0p5_i1p5, SubSIMD(uu_i0m5_i1m5, AddSIMD(uu_i0m5_i1p5, uu_i0p5_i1m5))),
                                                            FusedMulAddSIMD(
                                                                FDPart1_Rational_1_21168,
                                                                AddSIMD(AddSIMD(uu_i0m5_i1m3, uu_i0p3_i1p5),
                                                                        AddSIMD(uu_i0p5_i1p3,
                                                                                SubSIMD(uu_i0m3_i1m5, AddSIMD(AddSIMD(uu_i0m3_i1p5, uu_i0m5_i1p3),
                                                                                                              AddSIMD(uu_i0p3_i1m5, uu_i0p5_i1m3))))),
                                                                FusedMulAddSIMD(
                                                                    FDPart1_Rational_25_504,
                                                                    AddSIMD(
                                                                        AddSIMD(uu_i0m3_i1m1, uu_i0p1_i1p3),
                                                                        AddSIMD(uu_i0p3_i1p1,
                                                                                SubSIMD(uu_i0m1_i1m3, AddSIMD(AddSIMD(uu_i0m1_i1p3, uu_i0m3_i1p1),
                                                                                                              AddSIMD(uu_i0p1_i1m3, uu_i0p3_i1m1))))),
                                                                    FusedMulAddSIMD(
                                                                        FDPart1_Rational_25_7056,
                                                                        AddSIMD(uu_i0p3_i1p3,
                                                                                SubSIMD(uu_i0m3_i1m3, AddSIMD(uu_i0m3_i1p3, uu_i0p3_i1m3))),
                                                                        FusedMulAddSIMD(
                                                                            FDPart1_Rational_1_127008,
                                                                            AddSIMD(AddSIMD(uu_i0m5_i1p4, uu_i0p4_i1m5),
                                                                                    AddSIMD(uu_i0p5_i1m4,
                                                                                            SubSIMD(uu_i0m4_i1p5,
                                                                                                    AddSIMD(AddSIMD(uu_i0m4_i1m5, uu_i0m5_i1m4),
                                                                                                            AddSIMD(uu_i0p4_i1p5, uu_i0p5_i1p4))))),
                                                                            MulSIMD(
                                                                                FDPart1_Rational_1_1512,
                                                                                AddSIMD(
                                                                                    AddSIMD(uu_i0m5_i1m1, uu_i0p1_i1p5),
                                                                                    AddSIMD(
                                                                                        uu_i0p5_i1p1,
                                                                                        SubSIMD(uu_i0m1_i1m5,
                                                                                                AddSIMD(AddSIMD(uu_i0m1_i1p5, uu_i0m5_i1p1),
                                                                                                        AddSIMD(uu_i0p1_i1m5,
                                                                                                                uu_i0p5_i1m1))))))))))))))))))))));
        const REAL_SIMD_ARRAY uu_dDD11 =
            MulSIMD(MulSIMD(invdxx1, invdxx1),
                    FusedMulAddSIMD(FDPart1_Rational_5_126, AddSIMD(uu_i1m3, uu_i1p3),
                                    FusedMulAddSIMD(FDPart1_Rational_5_3, AddSIMD(uu_i1m1, uu_i1p1),
                                                    FusedMulSubSIMD(FDPart1_Rational_1_3150, AddSIMD(uu_i1m5, uu_i1p5),
                                                                    FusedMulAddSIMD(FDPart1_Rational_5_1008, AddSIMD(uu_i1m4, uu_i1p4),
                                                                                    FusedMulAddSIMD(FDPart1_Rational_5_21, AddSIMD(uu_i1m2, uu_i1p2),
                                                                                                    MulSIMD(FDPart1_Rational_5269_1800, uu)))))));

        /*
         * NRPy-Generated GF Access/FD Code, Step 2 of 2:
         * Evaluate SymPy expressions and write to main memory.
         */
        const REAL_SIMD_ARRAY __RHS_exp_0 = uu_dD0;
        const REAL_SIMD_ARRAY __RHS_exp_1 = uu_dD1;
        const REAL_SIMD_ARRAY __RHS_exp_2 = uu_dDD00;
        const REAL_SIMD_ARRAY __RHS_exp_3 = uu_dDD01;
        const REAL_SIMD_ARRAY __RHS_exp_4 = uu_dDD11;

        WriteSIMD(&NRPYELL_fields[IDX4(NRPYELL_SOL_UU_DD0GF, i0, i1, i2)], __RHS_exp_0);
        WriteSIMD(&NRPYELL_fields[IDX4(NRPYELL_SOL_UU_DD1GF, i0, i1, i2)], __RHS_exp_1);
        WriteSIMD(&NRPYELL_fields[IDX4(NRPYELL_SOL_UU_DD00GF, i0, i1, i2)], __RHS_exp_2);
        WriteSIMD(&NRPYELL_fields[IDX4(NRPYELL_SOL_UU_DD01GF, i0, i1, i2)], __RHS_exp_3);
        WriteSIMD(&NRPYELL_fields[IDX4(NRPYELL_SOL_UU_DD11GF, i0, i1, i2)], __RHS_exp_4);

      } // END LOOP: for (int i0 = NGHOSTS; i0 < Nxx_plus_2NGHOSTS0 - NGHOSTS; i0 += simd_width)
    } // END LOOP: for (int i1 = NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1++)
  } // END LOOP: for (int i2 = NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2++)
} // END FUNCTION: NRPYELL_derivatives_all_points
