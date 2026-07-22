#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"

/**
 * Given Cartesian point (x,y,z), this function unshifts the grid back to the origin to output the corresponding
 *             (xx0,xx1,xx2) and the "closest" (i0,i1,i2) for the given grid
 */
void Cart_to_xx_and_nearest_i0i1i2(const params_struct *restrict params, const REAL xCart[3], REAL xx[3], int Cart_to_i0i1i2[3]) {
  switch (params->CoordSystem_hash) {
  case SINHSYMTP:
    Cart_to_xx_and_nearest_i0i1i2__rfm__SinhSymTP(params, xCart, xx, Cart_to_i0i1i2);
    break;
  default:
    fprintf(stderr, "ERROR in Cart_to_xx_and_nearest_i0i1i2(): CoordSystem hash = %d not #define'd!\n", params->CoordSystem_hash);
    exit(1);
  }
} // END FUNCTION Cart_to_xx_and_nearest_i0i1i2
