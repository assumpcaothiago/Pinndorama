#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/**
 * Compute Cartesian coordinates {x, y, z} = {xCart[0], xCart[1], xCart[2]} in terms of
 * local grid coordinates {xx[0][i0], xx[1][i1], xx[2][i2]} = {xx0, xx1, xx2},
 * taking into account the possibility that the origin of this grid is off-center.
 */
void xx_to_Cart(const commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3], const int i0, const int i1,
                const int i2, REAL xCart[3]) {
  switch (params->CoordSystem_hash) {
  case SINHSYMTP:
    xx_to_Cart__rfm__SinhSymTP(commondata, params, xx, i0, i1, i2, xCart);
    break;
  default:
    fprintf(stderr, "ERROR in xx_to_Cart(): CoordSystem hash = %d not #define'd!\n", params->CoordSystem_hash);
    exit(1);
  }
} // END FUNCTION xx_to_Cart
