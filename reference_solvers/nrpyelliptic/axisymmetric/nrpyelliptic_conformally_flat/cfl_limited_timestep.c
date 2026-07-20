#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/**
 * Compute minimum timestep dt = CFL_FACTOR * ds_min.
 */
void cfl_limited_timestep(commondata_struct *restrict commondata, params_struct *restrict params, REAL *restrict xx[3]) {

  REAL ds_min = 1e38;
  LOOP_OMP("omp parallel for reduction(min:ds_min)", i0, 0, params->Nxx_plus_2NGHOSTS0, i1, 0, params->Nxx_plus_2NGHOSTS1, i2, 0,
           params->Nxx_plus_2NGHOSTS2) {
    REAL local_ds_min;
    ds_min_single_pt(commondata, params, xx[0][i0], xx[1][i1], xx[2][i2], &local_ds_min);
    ds_min = MIN(ds_min, local_ds_min);
  }
  commondata->dt = MIN(commondata->dt, ds_min * commondata->CFL_FACTOR);
} // END FUNCTION cfl_limited_timestep
