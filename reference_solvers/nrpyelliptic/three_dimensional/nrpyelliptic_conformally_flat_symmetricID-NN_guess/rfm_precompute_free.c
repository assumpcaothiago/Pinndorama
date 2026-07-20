#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"

/**
 * @file rfm_precompute_free.c
 * @brief Frees memory for precomputed reference metric quantities.
 *
 * - params->is_host==true: free HOST memory.
 * - params->is_host==false: free DEVICE memory.
 *
 * @param[in]  commondata  Global simulation metadata (unused).
 * @param[in]  params      Grid parameters.
 * @param[out] rfmstruct   Struct whose members will be freed.
 *
 * @return void.
 *
 */
void rfm_precompute_free(const commondata_struct *restrict commondata, const params_struct *restrict params, rfm_struct *restrict rfmstruct) {
  switch (params->CoordSystem_hash) {
  case SINHSYMTP:
    rfm_precompute_free__rfm__SinhSymTP(commondata, params, rfmstruct);
    break;
  default:
    fprintf(stderr, "ERROR in rfm_precompute_free(): CoordSystem hash = %d not #define'd!\n", params->CoordSystem_hash);
    exit(1);
  }
} // END FUNCTION rfm_precompute_free
