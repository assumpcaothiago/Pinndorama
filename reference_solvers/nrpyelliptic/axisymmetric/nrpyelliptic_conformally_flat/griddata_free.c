#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
/**
 * Free all memory within the griddata struct,
 * except perhaps non_y_n_gfs (e.g., after a regrid, in which non_y_n_gfs are freed first).
 */
void griddata_free(const commondata_struct *restrict commondata, griddata_struct *restrict griddata,
                   const bool free_non_y_n_gfs_and_core_griddata_pointers) {
  // Free memory allocated inside griddata[].
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    rfm_precompute_free(commondata, &griddata[grid].params, griddata[grid].rfmstruct);
    free(griddata[grid].rfmstruct);

    free(griddata[grid].bcstruct.inner_bc_array);
    for (int ng = 0; ng < NGHOSTS * 3; ng++)
      free(griddata[grid].bcstruct.pure_outer_bc_array[ng]);

    MoL_free_memory_y_n_gfs(&griddata[grid].gridfuncs);
    if (free_non_y_n_gfs_and_core_griddata_pointers)
      MoL_free_memory_non_y_n_gfs(&griddata[grid].gridfuncs);
    for (int i = 0; i < 3; i++)
      free(griddata[grid].xx[i]);
  } // END for(int grid=0;grid<commondata->NUMGRIDS;grid++)
  if (free_non_y_n_gfs_and_core_griddata_pointers)
    free(griddata);
} // END FUNCTION griddata_free
