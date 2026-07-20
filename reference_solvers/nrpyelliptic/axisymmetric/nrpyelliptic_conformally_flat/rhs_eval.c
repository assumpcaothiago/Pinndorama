#include "BHaH_defines.h"
#include "intrinsics/simd_intrinsics.h"
/**
 * Set RHSs for hyperbolic relaxation equation.
 */
void rhs_eval(const commondata_struct *restrict commondata, const params_struct *restrict params, const rfm_struct *restrict rfmstruct,
              const REAL *restrict auxevol_gfs, const REAL *restrict in_gfs, REAL *restrict rhs_gfs) {
#include "set_CodeParameters-simd.h"
#pragma omp parallel for collapse(2)
  for (int i2 = NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2++) {
    for (int i1 = NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1++) {
      const double NOSIMDf1_of_xx1 = rfmstruct->f1_of_xx1[i1];
      MAYBE_UNUSED const REAL_SIMD_ARRAY f1_of_xx1 = ConstSIMD(NOSIMDf1_of_xx1);
      const double NOSIMDf1_of_xx1__D1 = rfmstruct->f1_of_xx1__D1[i1];
      MAYBE_UNUSED const REAL_SIMD_ARRAY f1_of_xx1__D1 = ConstSIMD(NOSIMDf1_of_xx1__D1);
      const double NOSIMDf1_of_xx1__DD11 = rfmstruct->f1_of_xx1__DD11[i1];
      MAYBE_UNUSED const REAL_SIMD_ARRAY f1_of_xx1__DD11 = ConstSIMD(NOSIMDf1_of_xx1__DD11);
      const double NOSIMDf4_of_xx1 = rfmstruct->f4_of_xx1[i1];
      MAYBE_UNUSED const REAL_SIMD_ARRAY f4_of_xx1 = ConstSIMD(NOSIMDf4_of_xx1);
      const double NOSIMDf4_of_xx1__D1 = rfmstruct->f4_of_xx1__D1[i1];
      MAYBE_UNUSED const REAL_SIMD_ARRAY f4_of_xx1__D1 = ConstSIMD(NOSIMDf4_of_xx1__D1);
      const double NOSIMDf4_of_xx1__DD11 = rfmstruct->f4_of_xx1__DD11[i1];
      MAYBE_UNUSED const REAL_SIMD_ARRAY f4_of_xx1__DD11 = ConstSIMD(NOSIMDf4_of_xx1__DD11);

      for (int i0 = NGHOSTS; i0 < Nxx_plus_2NGHOSTS0 - NGHOSTS; i0 += simd_width) {
        MAYBE_UNUSED const REAL_SIMD_ARRAY f0_of_xx0 = ReadSIMD(&rfmstruct->f0_of_xx0[i0]);
        MAYBE_UNUSED const REAL_SIMD_ARRAY f0_of_xx0__D0 = ReadSIMD(&rfmstruct->f0_of_xx0__D0[i0]);
        MAYBE_UNUSED const REAL_SIMD_ARRAY f0_of_xx0__DD00 = ReadSIMD(&rfmstruct->f0_of_xx0__DD00[i0]);
        MAYBE_UNUSED const REAL_SIMD_ARRAY f0_of_xx0__DDD000 = ReadSIMD(&rfmstruct->f0_of_xx0__DDD000[i0]);
        MAYBE_UNUSED const REAL_SIMD_ARRAY f2_of_xx0 = ReadSIMD(&rfmstruct->f2_of_xx0[i0]);
        MAYBE_UNUSED const REAL_SIMD_ARRAY f2_of_xx0__D0 = ReadSIMD(&rfmstruct->f2_of_xx0__D0[i0]);
        MAYBE_UNUSED const REAL_SIMD_ARRAY f2_of_xx0__DD00 = ReadSIMD(&rfmstruct->f2_of_xx0__DD00[i0]);

        /*
         * NRPy+-Generated GF Access/FD Code, Step 1 of 2:
         * Read gridfunction(s) from main memory and compute FD stencils as needed.
         */
        const REAL_SIMD_ARRAY ADD_times_AUU = ReadSIMD(&auxevol_gfs[IDX4(ADD_TIMES_AUUGF, i0, i1, i2)]);
        const REAL_SIMD_ARRAY psi_background = ReadSIMD(&auxevol_gfs[IDX4(PSI_BACKGROUNDGF, i0, i1, i2)]);
        const REAL_SIMD_ARRAY uu_i1m5 = ReadSIMD(&in_gfs[IDX4(UUGF, i0, i1 - 5, i2)]);
        const REAL_SIMD_ARRAY uu_i1m4 = ReadSIMD(&in_gfs[IDX4(UUGF, i0, i1 - 4, i2)]);
        const REAL_SIMD_ARRAY uu_i1m3 = ReadSIMD(&in_gfs[IDX4(UUGF, i0, i1 - 3, i2)]);
        const REAL_SIMD_ARRAY uu_i1m2 = ReadSIMD(&in_gfs[IDX4(UUGF, i0, i1 - 2, i2)]);
        const REAL_SIMD_ARRAY uu_i1m1 = ReadSIMD(&in_gfs[IDX4(UUGF, i0, i1 - 1, i2)]);
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
        const REAL_SIMD_ARRAY uu_i1p1 = ReadSIMD(&in_gfs[IDX4(UUGF, i0, i1 + 1, i2)]);
        const REAL_SIMD_ARRAY uu_i1p2 = ReadSIMD(&in_gfs[IDX4(UUGF, i0, i1 + 2, i2)]);
        const REAL_SIMD_ARRAY uu_i1p3 = ReadSIMD(&in_gfs[IDX4(UUGF, i0, i1 + 3, i2)]);
        const REAL_SIMD_ARRAY uu_i1p4 = ReadSIMD(&in_gfs[IDX4(UUGF, i0, i1 + 4, i2)]);
        const REAL_SIMD_ARRAY uu_i1p5 = ReadSIMD(&in_gfs[IDX4(UUGF, i0, i1 + 5, i2)]);
        const REAL_SIMD_ARRAY variable_wavespeed = ReadSIMD(&auxevol_gfs[IDX4(VARIABLE_WAVESPEEDGF, i0, i1, i2)]);
        const REAL_SIMD_ARRAY vv = ReadSIMD(&in_gfs[IDX4(VVGF, i0, i1, i2)]);
        static const double dblFDPart1_NegativeOne_ = -1.0;
        MAYBE_UNUSED const REAL_SIMD_ARRAY FDPart1_NegativeOne_ = ConstSIMD(dblFDPart1_NegativeOne_);

        static const double dblFDPart1_Rational_1_1260 = 1.0 / 1260.0;
        const REAL_SIMD_ARRAY FDPart1_Rational_1_1260 = ConstSIMD(dblFDPart1_Rational_1_1260);

        static const double dblFDPart1_Rational_1_3150 = 1.0 / 3150.0;
        const REAL_SIMD_ARRAY FDPart1_Rational_1_3150 = ConstSIMD(dblFDPart1_Rational_1_3150);

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
        const REAL_SIMD_ARRAY uu_dDD11 =
            MulSIMD(MulSIMD(invdxx1, invdxx1),
                    FusedMulAddSIMD(FDPart1_Rational_5_126, AddSIMD(uu_i1m3, uu_i1p3),
                                    FusedMulAddSIMD(FDPart1_Rational_5_3, AddSIMD(uu_i1m1, uu_i1p1),
                                                    FusedMulSubSIMD(FDPart1_Rational_1_3150, AddSIMD(uu_i1m5, uu_i1p5),
                                                                    FusedMulAddSIMD(FDPart1_Rational_5_1008, AddSIMD(uu_i1m4, uu_i1p4),
                                                                                    FusedMulAddSIMD(FDPart1_Rational_5_21, AddSIMD(uu_i1m2, uu_i1p2),
                                                                                                    MulSIMD(FDPart1_Rational_5269_1800, uu)))))));

        /*
         * NRPy+-Generated GF Access/FD Code, Step 2 of 2:
         * Evaluate SymPy expressions and write to main memory.
         */
        static const double dblFDPart3_Integer_1 = 1.0;
        MAYBE_UNUSED const REAL_SIMD_ARRAY FDPart3_Integer_1 = ConstSIMD(dblFDPart3_Integer_1);

        static const double dblFDPart3_Integer_2 = 2.0;
        const REAL_SIMD_ARRAY FDPart3_Integer_2 = ConstSIMD(dblFDPart3_Integer_2);

        static const double dblFDPart3_NegativeOne_ = -1.0;
        MAYBE_UNUSED const REAL_SIMD_ARRAY FDPart3_NegativeOne_ = ConstSIMD(dblFDPart3_NegativeOne_);

        static const double dblFDPart3_Rational_1_2 = 1.0 / 2.0;
        const REAL_SIMD_ARRAY FDPart3_Rational_1_2 = ConstSIMD(dblFDPart3_Rational_1_2);

        static const double dblFDPart3_Rational_1_8 = 1.0 / 8.0;
        const REAL_SIMD_ARRAY FDPart3_Rational_1_8 = ConstSIMD(dblFDPart3_Rational_1_8);

        const REAL_SIMD_ARRAY FDPart3tmp0 = FusedMulAddSIMD(f0_of_xx0, f0_of_xx0, MulSIMD(f4_of_xx1, f4_of_xx1));
        const REAL_SIMD_ARRAY FDPart3tmp3 = MulSIMD(f2_of_xx0, f2_of_xx0);
        const REAL_SIMD_ARRAY FDPart3tmp1 = DivSIMD(FDPart3_Integer_1, FDPart3tmp0);
        const REAL_SIMD_ARRAY FDPart3tmp5 = DivSIMD(FDPart3_Integer_1, MulSIMD(FDPart3tmp0, FDPart3tmp0));
        const REAL_SIMD_ARRAY FDPart3tmp7 = DivSIMD(FDPart3_Integer_2, FDPart3tmp3);
        const REAL_SIMD_ARRAY __RHS_exp_0 = NegFusedMulAddSIMD(eta_damping, uu, vv);
        const REAL_SIMD_ARRAY __RHS_exp_1 = MulSIMD(
            MulSIMD(variable_wavespeed, variable_wavespeed),
            FusedMulAddSIMD(
                FDPart3_Rational_1_8,
                DivSIMD(ADD_times_AUU, MulSIMD(MulSIMD(MulSIMD(MulSIMD(MulSIMD(MulSIMD(AddSIMD(psi_background, uu), AddSIMD(psi_background, uu)),
                                                                               AddSIMD(psi_background, uu)),
                                                                       AddSIMD(psi_background, uu)),
                                                               AddSIMD(psi_background, uu)),
                                                       AddSIMD(psi_background, uu)),
                                               AddSIMD(psi_background, uu))),
                FusedMulAddSIMD(
                    MulSIMD(FDPart3tmp1, FDPart3tmp3), DivSIMD(uu_dDD00, MulSIMD(f0_of_xx0__D0, f0_of_xx0__D0)),
                    FusedMulAddSIMD(
                        MulSIMD(FDPart3tmp1, f1_of_xx1__D1), DivSIMD(uu_dD1, f1_of_xx1),
                        FusedMulSubSIMD(
                            FDPart3tmp1, uu_dDD11,
                            MulSIMD(
                                uu_dD0,
                                FusedMulSubSIMD(
                                    FDPart3tmp5, MulSIMD(MulSIMD(FDPart3_NegativeOne_, FDPart3tmp3), DivSIMD(f0_of_xx0, f0_of_xx0__D0)),
                                    FusedMulSubSIMD(
                                        MulSIMD(FDPart3tmp1, FDPart3tmp3), DivSIMD(DivSIMD(FDPart3_Integer_1, f0_of_xx0__D0), f0_of_xx0),
                                        DivSIMD(
                                            MulSIMD(MulSIMD(FDPart3_Rational_1_2, FDPart3tmp5),
                                                    MulSIMD(MulSIMD(MulSIMD(MulSIMD(f2_of_xx0, f2_of_xx0), f2_of_xx0), f2_of_xx0),
                                                            FusedMulAddSIMD(
                                                                MulSIMD(FDPart3tmp0, FDPart3tmp7), MulSIMD(f0_of_xx0__D0, f0_of_xx0__DD00),
                                                                FusedMulSubSIMD(
                                                                    FDPart3tmp7,
                                                                    MulSIMD(f0_of_xx0, MulSIMD(MulSIMD(f0_of_xx0__D0, f0_of_xx0__D0), f0_of_xx0__D0)),
                                                                    MulSIMD(f2_of_xx0__D0,
                                                                            MulSIMD(MulSIMD(FDPart3_Integer_2, FDPart3tmp0),
                                                                                    DivSIMD(MulSIMD(f0_of_xx0__D0, f0_of_xx0__D0),
                                                                                            MulSIMD(MulSIMD(f2_of_xx0, f2_of_xx0), f2_of_xx0)))))))),
                                            MulSIMD(MulSIMD(MulSIMD(f0_of_xx0__D0, f0_of_xx0__D0), f0_of_xx0__D0), f0_of_xx0__D0))))))))));

        WriteSIMD(&rhs_gfs[IDX4(UUGF, i0, i1, i2)], __RHS_exp_0);
        WriteSIMD(&rhs_gfs[IDX4(VVGF, i0, i1, i2)], __RHS_exp_1);

      } // END LOOP: for (int i0 = NGHOSTS; i0 < Nxx_plus_2NGHOSTS0 - NGHOSTS; i0 += simd_width)
    } // END LOOP: for (int i1 = NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1++)
  } // END LOOP: for (int i2 = NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2++)
} // END FUNCTION rhs_eval
