#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/**
 * Set initial guess to solutions of hyperbolic relaxation equation at all points.
 */
void initial_data(commondata_struct *restrict commondata, griddata_struct *restrict griddata) {
  // Attempt to read checkpoint file. If it doesn't exist, then continue. Otherwise return.
  if (read_checkpoint(commondata, griddata))
    return;
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    // Unpack griddata struct:
    params_struct *restrict params = &griddata[grid].params;
#include "set_CodeParameters.h"
    REAL *restrict xx[3];
    for (int ww = 0; ww < 3; ww++)
      xx[ww] = griddata[grid].xx[ww];
    REAL *restrict in_gfs = griddata[grid].gridfuncs.y_n_gfs;
#pragma omp parallel for collapse(2)
    for (int i2 = 0; i2 < Nxx_plus_2NGHOSTS2; i2++) {
      for (int i1 = 0; i1 < Nxx_plus_2NGHOSTS1; i1++) {
        MAYBE_UNUSED const REAL xx2 = xx[2][i2];
        MAYBE_UNUSED const REAL xx1 = xx[1][i1];
        for (int i0 = 0; i0 < Nxx_plus_2NGHOSTS0; i0++) {
          MAYBE_UNUSED const REAL xx0 = xx[0][i0];

          initial_guess_single_point(commondata, params, xx0, xx1, xx2, &in_gfs[IDX4(UUGF, i0, i1, i2)], &in_gfs[IDX4(VVGF, i0, i1, i2)]);
        } // END LOOP: for (int i0 = 0; i0 < Nxx_plus_2NGHOSTS0; i0++)
      } // END LOOP: for (int i1 = 0; i1 < Nxx_plus_2NGHOSTS1; i1++)
    } // END LOOP: for (int i2 = 0; i2 < Nxx_plus_2NGHOSTS2; i2++)
  }
} // END FUNCTION initial_data
