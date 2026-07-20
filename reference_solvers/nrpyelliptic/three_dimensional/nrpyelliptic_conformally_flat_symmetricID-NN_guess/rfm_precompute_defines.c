#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"

/**
 * @file rfm_precompute_defines.c
 * @brief Computes and populates arrays with precomputed reference metric quantities.
 *
 * - params->is_host==true: rfmstruct & xx[] are HOST; run CPU loops.
 * - params->is_host==false: rfmstruct & xx[] are DEVICE/UVM; launch CUDA kernels.
 *
 * @param[in]  commondata  Global simulation metadata (unused).
 * @param[in]  params      Grid parameters and dimensions.
 * @param[out] rfmstruct   Struct containing arrays to populate.
 * @param[in]  xx          Pointers to 1D coordinate arrays.
 *
 * @return void.
 *
 */
void rfm_precompute_defines(const commondata_struct *restrict commondata, const params_struct *restrict params, rfm_struct *restrict rfmstruct,
                            REAL *restrict xx[3]) {
  switch (params->CoordSystem_hash) {
  case SINHSYMTP:
    rfm_precompute_defines__rfm__SinhSymTP(commondata, params, rfmstruct, xx);
    break;
  default:
    fprintf(stderr, "ERROR in rfm_precompute_defines(): CoordSystem hash = %d not #define'd!\n", params->CoordSystem_hash);
    exit(1);
  }
} // END FUNCTION rfm_precompute_defines
