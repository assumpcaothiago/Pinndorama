#include "BHaH_defines.h"
/**
 * Compute initial guess at a single point.
 */
void initial_guess_single_point(const commondata_struct *restrict commondata, const params_struct *restrict params, const REAL xx0, const REAL xx1,
                                const REAL xx2, REAL *restrict uu_ID, REAL *restrict vv_ID) {
#include "set_CodeParameters.h"
  *uu_ID = 0;
  *vv_ID = 0;
} // END FUNCTION initial_guess_single_point
