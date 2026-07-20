#include "BHaH_defines.h"
/**
 * Compute variable wavespeed for all grids based on local grid spacing.
 */
void variable_wavespeed_gfs_all_points(commondata_struct *restrict commondata, griddata_struct *restrict griddata) {
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    // Unpack griddata struct:
    params_struct *restrict params = &griddata[grid].params;
#include "set_CodeParameters.h"
    REAL *restrict xx[3];
    for (int ww = 0; ww < 3; ww++)
      xx[ww] = griddata[grid].xx[ww];
    REAL *restrict in_gfs = griddata[grid].gridfuncs.auxevol_gfs;
#pragma omp parallel for
    for (int i2 = NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2++) {
      MAYBE_UNUSED const REAL xx2 = xx[2][i2];
      for (int i1 = NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1++) {
        MAYBE_UNUSED const REAL xx1 = xx[1][i1];
        for (int i0 = NGHOSTS; i0 < Nxx_plus_2NGHOSTS0 - NGHOSTS; i0++) {
          MAYBE_UNUSED const REAL xx0 = xx[0][i0];

          /*
           *  Original SymPy expressions:
           *  "[const REAL dsmin0 = AMAX*dxx0*(exp(xx0/SINHWAA)/SINHWAA + exp(-xx0/SINHWAA)/SINHWAA)*sqrt(AMAX**2*(exp(xx0/SINHWAA) -
           * exp(-xx0/SINHWAA))**2/(exp(1/SINHWAA) - exp(-1/SINHWAA))**2 + bScale**2*sin(xx1)**2)/(sqrt(AMAX**2*(exp(xx0/SINHWAA) -
           * exp(-xx0/SINHWAA))**2/(exp(1/SINHWAA) - exp(-1/SINHWAA))**2 + bScale**2)*(exp(1/SINHWAA) - exp(-1/SINHWAA)))]"
           *  "[const REAL dsmin1 = dxx1*sqrt(AMAX**2*(exp(xx0/SINHWAA) - exp(-xx0/SINHWAA))**2/(exp(1/SINHWAA) - exp(-1/SINHWAA))**2 +
           * bScale**2*sin(xx1)**2)]"
           *  "[const REAL dsmin2 = AMAX*dxx2*(exp(xx0/SINHWAA) - exp(-xx0/SINHWAA))*sin(xx1)/(exp(1/SINHWAA) - exp(-1/SINHWAA))]"
           */
          const REAL tmp1 = sin(xx1);
          const REAL tmp2 = (1.0 / (SINHWAA));
          const REAL tmp3 = exp(tmp2) - exp(-tmp2);
          const REAL tmp5 = exp(tmp2 * xx0);
          const REAL tmp6 = exp(-tmp2 * xx0);
          const REAL tmp10 = AMAX / tmp3;
          const REAL tmp7 = tmp5 - tmp6;
          const REAL tmp8 = ((AMAX) * (AMAX)) * ((tmp7) * (tmp7)) / ((tmp3) * (tmp3));
          const REAL tmp9 = sqrt(((bScale) * (bScale)) * ((tmp1) * (tmp1)) + tmp8);
          const REAL dsmin0 = dxx0 * tmp10 * tmp9 * (tmp2 * tmp5 + tmp2 * tmp6) / sqrt(((bScale) * (bScale)) + tmp8);
          const REAL dsmin1 = dxx1 * tmp9;
          const REAL dsmin2 = dxx2 * tmp1 * tmp10 * tmp7;

          // Set local wavespeed
          in_gfs[IDX4(VARIABLE_WAVESPEEDGF, i0, i1, i2)] = MINIMUM_GLOBAL_WAVESPEED * MIN(dsmin0, MIN(dsmin1, dsmin2)) / dt;

        } // END LOOP: for (int i0 = NGHOSTS; i0 < Nxx_plus_2NGHOSTS0 - NGHOSTS; i0++)
      } // END LOOP: for (int i1 = NGHOSTS; i1 < Nxx_plus_2NGHOSTS1 - NGHOSTS; i1++)
    } // END LOOP: for (int i2 = NGHOSTS; i2 < Nxx_plus_2NGHOSTS2 - NGHOSTS; i2++)
  } // END LOOP for(int grid=0; grid<commondata->NUMGRIDS; grid++)
} // END FUNCTION variable_wavespeed_gfs_all_points
