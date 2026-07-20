#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/**
 * Output diagnostic quantities at gridpoints closest to yz plane.
 */
void diagnostics_nearest_2d_yz_plane(commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3],
                                     MoL_gridfunctions_struct *restrict gridfuncs) {
  switch (params->CoordSystem_hash) {
  case SINHSYMTP:
    diagnostics_nearest_2d_yz_plane__rfm__SinhSymTP(commondata, params, xx, gridfuncs);
    break;
  default:
    fprintf(stderr, "ERROR in diagnostics_nearest_2d_yz_plane(): CoordSystem hash = %d not #define'd!\n", params->CoordSystem_hash);
    exit(1);
  }
} // END FUNCTION diagnostics_nearest_2d_yz_plane
