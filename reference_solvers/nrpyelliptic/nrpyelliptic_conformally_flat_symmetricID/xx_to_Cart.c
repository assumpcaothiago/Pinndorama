#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"

/**
 * Compute Cartesian coordinates {x, y, z} = {xCart[0], xCart[1], xCart[2]} from the
 * local coordinate vector {xx[0], xx[1], xx[2]} = {xx0, xx1, xx2},
 * taking into account the possibility that the origin of this grid is off-center.
 */
void xx_to_Cart(const params_struct *restrict params, const REAL xx[3], REAL xCart[3]) {
  switch (params->CoordSystem_hash) {
  case SINHSYMTP:
    xx_to_Cart__rfm__SinhSymTP(params, xx, xCart);
    break;
  default:
    fprintf(stderr, "ERROR in xx_to_Cart(): CoordSystem hash = %d not #define'd!\n", params->CoordSystem_hash);
    exit(1);
  }
} // END FUNCTION xx_to_Cart
