#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/**
 * Set AUXEVOL gridfunctions at all points.
 */
void auxevol_gfs_all_points(commondata_struct *restrict commondata, griddata_struct *restrict griddata) {
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    // Unpack griddata struct:
    params_struct *restrict params = &griddata[grid].params;
#include "set_CodeParameters.h"
    REAL *restrict xx[3];
    for (int ww = 0; ww < 3; ww++)
      xx[ww] = griddata[grid].xx[ww];
    REAL *restrict in_gfs = griddata[grid].gridfuncs.auxevol_gfs;
#pragma omp parallel for collapse(2)
    for (int i2 = 0; i2 < Nxx_plus_2NGHOSTS2; i2++) {
      for (int i1 = 0; i1 < Nxx_plus_2NGHOSTS1; i1++) {
        MAYBE_UNUSED const REAL xx2 = xx[2][i2];
        MAYBE_UNUSED const REAL xx1 = xx[1][i1];
        for (int i0 = 0; i0 < Nxx_plus_2NGHOSTS0; i0++) {
          MAYBE_UNUSED const REAL xx0 = xx[0][i0];

          auxevol_gfs_single_point(commondata, params, xx0, xx1, xx2, &in_gfs[IDX4(PSI_BACKGROUNDGF, i0, i1, i2)],
                                   &in_gfs[IDX4(ADD_TIMES_AUUGF, i0, i1, i2)]);
        } // END LOOP: for (int i0 = 0; i0 < Nxx_plus_2NGHOSTS0; i0++)
      } // END LOOP: for (int i1 = 0; i1 < Nxx_plus_2NGHOSTS1; i1++)
    } // END LOOP: for (int i2 = 0; i2 < Nxx_plus_2NGHOSTS2; i2++)
  }
} // END FUNCTION auxevol_gfs_all_points
