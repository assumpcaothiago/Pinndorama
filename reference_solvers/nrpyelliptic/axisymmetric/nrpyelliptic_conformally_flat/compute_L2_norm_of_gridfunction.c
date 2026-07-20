#include "BHaH_defines.h"
/**
 * Compute l2-norm of a gridfunction assuming a single grid.
 */
REAL compute_L2_norm_of_gridfunction(commondata_struct *restrict commondata, griddata_struct *restrict griddata, const REAL integration_radius,
                                     const int gf_index, const REAL *restrict in_gf) {

  // Unpack grid parameters assuming a single grid
  const int grid = 0;
  params_struct *restrict params = &griddata[grid].params;
#include "set_CodeParameters.h"

  // Define reference metric grid
  REAL *restrict xx[3];
  for (int ww = 0; ww < 3; ww++)
    xx[ww] = griddata[grid].xx[ww];

  // Set summation variables to compute l2-norm
  DOUBLE squared_sum = 0.0;
  DOUBLE volume_sum = 0.0;
#pragma omp parallel for reduction(+ : squared_sum, volume_sum)
  for (int i2 = NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2++) {
    MAYBE_UNUSED const REAL xx2 = xx[2][i2];
    for (int i1 = NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1++) {
      MAYBE_UNUSED const REAL xx1 = xx[1][i1];
      for (int i0 = NGHOSTS; i0 < Nxx_plus_2NGHOSTS0 - NGHOSTS; i0++) {
        MAYBE_UNUSED const REAL xx0 = xx[0][i0];

        /*
         *  Original SymPy expressions:
         *  "[const DOUBLE r = sqrt(AMAX**2*(exp(xx0/SINHWAA) - exp(-xx0/SINHWAA))**2*sin(xx1)**2/(exp(1/SINHWAA) - exp(-1/SINHWAA))**2 +
         * (AMAX**2*(exp(xx0/SINHWAA) - exp(-xx0/SINHWAA))**2/(exp(1/SINHWAA) - exp(-1/SINHWAA))**2 + bScale**2)*cos(xx1)**2)]"
         *  "[const DOUBLE sqrtdetgamma = AMAX**4*(exp(xx0/SINHWAA)/SINHWAA + exp(-xx0/SINHWAA)/SINHWAA)**2*(AMAX**2*(exp(xx0/SINHWAA) -
         * exp(-xx0/SINHWAA))**2/(exp(1/SINHWAA) - exp(-1/SINHWAA))**2 + bScale**2*sin(xx1)**2)**2*(exp(xx0/SINHWAA) -
         * exp(-xx0/SINHWAA))**2*sin(xx1)**2/((AMAX**2*(exp(xx0/SINHWAA) - exp(-xx0/SINHWAA))**2/(exp(1/SINHWAA) - exp(-1/SINHWAA))**2 +
         * bScale**2)*(exp(1/SINHWAA) - exp(-1/SINHWAA))**4)]"
         */
        const REAL tmp0 = ((sin(xx1)) * (sin(xx1)));
        const REAL tmp1 = (1.0 / (SINHWAA));
        const REAL tmp2 = exp(tmp1) - exp(-tmp1);
        const REAL tmp4 = exp(tmp1 * xx0);
        const REAL tmp5 = exp(-tmp1 * xx0);
        const REAL tmp6 = ((tmp4 - tmp5) * (tmp4 - tmp5));
        const REAL tmp7 = ((AMAX) * (AMAX)) * tmp6 / ((tmp2) * (tmp2));
        const REAL tmp9 = ((bScale) * (bScale)) + tmp7;
        const DOUBLE r = sqrt(tmp0 * tmp7 + tmp9 * ((cos(xx1)) * (cos(xx1))));
        const DOUBLE sqrtdetgamma = ((AMAX) * (AMAX) * (AMAX) * (AMAX)) * tmp0 * tmp6 *
                                    ((((bScale) * (bScale)) * tmp0 + tmp7) * (((bScale) * (bScale)) * tmp0 + tmp7)) *
                                    ((tmp1 * tmp4 + tmp1 * tmp5) * (tmp1 * tmp4 + tmp1 * tmp5)) / (((tmp2) * (tmp2) * (tmp2) * (tmp2)) * tmp9);

        if (r < integration_radius) {
          const DOUBLE gf_of_x = in_gf[IDX4(gf_index, i0, i1, i2)];
          const DOUBLE dV = sqrtdetgamma * dxx0 * dxx1 * dxx2;
          squared_sum += gf_of_x * gf_of_x * dV;
          volume_sum += dV;
        } // END if(r < integration_radius)

      } // END LOOP: for (int i0 = NGHOSTS; i0 < Nxx_plus_2NGHOSTS0 - NGHOSTS; i0++)
    } // END LOOP: for (int i1 = NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1++)
  } // END LOOP: for (int i2 = NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2++)

  // Compute and output the log of the l2-norm.
  return log10(1e-16 + sqrt(squared_sum / volume_sum)); // 1e-16 + ... avoids log10(0)
} // END FUNCTION compute_L2_norm_of_gridfunction
