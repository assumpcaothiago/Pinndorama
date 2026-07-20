#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"

/**
 * @file rfm_precompute_malloc.c
 * @brief Allocates memory for precomputed reference metric quantities.
 *
 * - When params->is_host==true, perform HOST allocation.
 * - When params->is_host==false, perform DEVICE allocation.
 *
 * @param[in]  commondata  Global simulation metadata (unused).
 * @param[in]  params      Grid parameters (is_host, dims).
 * @param[out] rfmstruct   Destination struct for allocated pointers.
 *
 * @return void.
 *
 */
void rfm_precompute_malloc(const commondata_struct *restrict commondata, const params_struct *restrict params, rfm_struct *restrict rfmstruct) {
  switch (params->CoordSystem_hash) {
  case SINHSYMTP:
    rfm_precompute_malloc__rfm__SinhSymTP(commondata, params, rfmstruct);
    break;
  default:
    fprintf(stderr, "ERROR in rfm_precompute_malloc(): CoordSystem hash = %d not #define'd!\n", params->CoordSystem_hash);
    exit(1);
  }
} // END FUNCTION rfm_precompute_malloc
