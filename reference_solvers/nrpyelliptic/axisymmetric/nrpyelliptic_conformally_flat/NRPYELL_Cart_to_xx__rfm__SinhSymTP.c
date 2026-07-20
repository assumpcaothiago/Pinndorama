#include "BHaH_defines.h"
/**
 * Given Cartesian point (x,y,z), this function outputs the corresponding (xx0,xx1,xx2)
 */
void NRPYELL_Cart_to_xx__rfm__SinhSymTP(const REAL AMAX, const REAL bScale, const REAL SINHWAA, const REAL xCart[3], REAL xx[3]) {

  // Set (Cartx, Carty, Cartz).
  REAL Cartx = xCart[0];
  REAL Carty = xCart[1];
  REAL Cartz = xCart[2];
  /*
   *  Original SymPy expressions:
   *  "[xx[0] = SINHWAA*acsch(sqrt(2)*AMAX*csch(1/SINHWAA)/sqrt(Cartx**2 + Carty**2 + Cartz**2 - bScale**2 + sqrt((Cartx**2 + Carty**2 + (-Cartz +
   * bScale)**2)*(Cartx**2 + Carty**2 + (Cartz + bScale)**2))))]"
   *  "[xx[1] = acos(sqrt(2)*Cartz/sqrt(Cartx**2 + Carty**2 + Cartz**2 + bScale**2 + sqrt((Cartx**2 + Carty**2 + (-Cartz + bScale)**2)*(Cartx**2 +
   * Carty**2 + (Cartz + bScale)**2))))]"
   *  "[xx[2] = atan2(Carty, Cartx)]"
   */
  const REAL tmp2 = ((Cartx) * (Cartx)) + ((Carty) * (Carty));
  const REAL tmp3 =
      ((Cartz) * (Cartz)) + tmp2 + sqrt((tmp2 + ((-Cartz + bScale) * (-Cartz + bScale))) * (tmp2 + ((Cartz + bScale) * (Cartz + bScale))));
  xx[0] = SINHWAA * (log(sqrt(1 + (1.0 / 2.0) * (-((bScale) * (bScale)) + tmp3) /
                                      (((AMAX) * (AMAX)) * ((((1.0 / ((1.0 / 2.0) * exp(exp(-log(SINHWAA))) - 1.0 / 2.0 * exp(-1 / SINHWAA))))) *
                                                            (((1.0 / ((1.0 / 2.0) * exp(exp(-log(SINHWAA))) - 1.0 / 2.0 * exp(-1 / SINHWAA)))))))) +
                         (1.0 / 2.0) * M_SQRT2 * sqrt(-((bScale) * (bScale)) + tmp3) /
                             (AMAX * ((1.0 / ((1.0 / 2.0) * exp(exp(-log(SINHWAA))) - 1.0 / 2.0 * exp(-1 / SINHWAA)))))));
  xx[1] = acos(M_SQRT2 * Cartz / sqrt(((bScale) * (bScale)) + tmp3));
  xx[2] = atan2(Carty, Cartx);
} // END FUNCTION NRPYELL_Cart_to_xx__rfm__SinhSymTP
