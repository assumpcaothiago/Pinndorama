#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"

/**
 * @file sqrt_detgammahat_d3xx_volume_element.c
 * @brief Compute the local 3D volume element sqrt(detgammahat) * d3xx at a point.
 *
 * This routine evaluates the positive volume element used when integrating scalar fields on a
 * 3D grid. The evaluation uses reference-metric data and grid spacings stored in the params
 * structure. The computed value is returned by reference via the dV pointer.
 *
 * The expression implemented is:
 *   dV = sqrt(detgammahat(xx0, xx1, xx2)) * abs(dxx0 * dxx1 * dxx2)
 * The absolute value ensures a positive volume element regardless of coordinate orientation.
 *
 * @param[in]  params  Pointer to parameter struct (reference-metric data, grid spacings, and sizes).
 * @param[in]  xx0     Local coordinate 0 at which to evaluate the volume element.
 * @param[in]  xx1     Local coordinate 1 at which to evaluate the volume element.
 * @param[in]  xx2     Local coordinate 2 at which to evaluate the volume element.
 * @param[out] dV      Pointer to the location where the volume element will be stored.
 *
 * @return     void. The result is written to *dV.
 *
 * Note: If a user-editable block is provided in the implementation, users may add custom logic,
 * such as scaling or additional diagnostics, prior to writing the result.
 *
 */
void sqrt_detgammahat_d3xx_volume_element__rfm__SinhSymTP(const params_struct *restrict params, const REAL xx0, const REAL xx1, const REAL xx2,
                                                          REAL *restrict dV) {
  const REAL AMAX = params->AMAX;
  const REAL SINHWAA = params->SINHWAA;
  const REAL bScale = params->bScale;
  const REAL dxx0 = params->dxx0;
  const REAL dxx1 = params->dxx1;
  const REAL dxx2 = params->dxx2;
  /*
   *  Original SymPy expression:
   *  "*dV = AMAX**2*sqrt((1 - exp(2*xx0/SINHWAA))**2*(AMAX**2*(1 - exp(2*xx0/SINHWAA))**2*exp(2/SINHWAA) + bScale**2*(1 -
   * exp(2/SINHWAA))**2*exp(2*xx0/SINHWAA)*sin(xx1)**2)**2*(exp(2*xx0/SINHWAA) + 1)**2*exp(2*(2 - 3*xx0)/SINHWAA)*sin(xx1)**2/(SINHWAA**2*(1 -
   * exp(2/SINHWAA))**6*(AMAX**2*(1 - exp(2*xx0/SINHWAA))**2*exp(2/SINHWAA) + bScale**2*(1 -
   * exp(2/SINHWAA))**2*exp(2*xx0/SINHWAA))))*Abs(dxx0*dxx1*dxx2)"
   */
  const REAL tmp1 = ((sin(xx1)) * (sin(xx1)));
  const REAL tmp2 = 2 / SINHWAA;
  const REAL tmp3 = exp(tmp2 * xx0);
  const REAL tmp4 = exp(tmp2);
  const REAL tmp5 = 1 - tmp4;
  const REAL tmp6 = ((1 - tmp3) * (1 - tmp3));
  const REAL tmp7 = ((AMAX) * (AMAX)) * tmp4 * tmp6;
  const REAL tmp8 = ((bScale) * (bScale)) * tmp3 * ((tmp5) * (tmp5));
  *dV = ((AMAX) * (AMAX)) *
        sqrt(tmp1 * tmp6 * ((tmp3 + 1) * (tmp3 + 1)) * ((tmp1 * tmp8 + tmp7) * (tmp1 * tmp8 + tmp7)) * exp(tmp2 * (2 - 3 * xx0)) /
             (((SINHWAA) * (SINHWAA)) * pow(tmp5, 6) * (tmp7 + tmp8))) *
        fabs(dxx0 * dxx1 * dxx2);
} // END FUNCTION sqrt_detgammahat_d3xx_volume_element__rfm__SinhSymTP
