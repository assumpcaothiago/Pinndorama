#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
#include "diagnostic_gfs.h"

/**
 * @brief Dispatch nearest-sampled diagnostics by invoking specialized helper routines for 0D, 1D, and 2D outputs.
 *
 * The diagnostics_nearest() dispatcher coordinates sampling of diagnostic gridfunction data from caller-provided
 * buffers and delegates output to three helper functions:
 *   - diagnostics_nearest_grid_center(): emits 0D diagnostics at the index triplet nearest the physical center.
 *   - diagnostics_nearest_1d_y_and_z_axes(): emits 1D diagnostics along lines nearest the y and z axes.
 *   - diagnostics_nearest_2d_xy_and_yz_planes(): emits 2D diagnostics across planes nearest the xy and yz planes.
 *
 * A single USER-EDIT block appears before the per-grid loop. In that block, users select which_gfs arrays (by enum)
 * for each dimensionality. These selections are applied uniformly to all grids. The dispatcher itself performs no
 * memory allocation or deallocation; all buffers are owned by the caller. Helper routines may perform file I/O.
 *
 * @param[in] commondata
 *   Pointer to shared simulation metadata and runtime context, including NUMGRIDS and iteration/time information.
 *
 * @param[in] griddata
 *   Pointer to an array of per-grid data structures. For grid index "grid", griddata[grid] provides parameters,
 *   coordinates, and strides required by the diagnostics helper routines.
 *
 * @param[in] gridfuncs_diags
 *   Array of length MAXNUMGRIDS. For each grid index "grid", gridfuncs_diags[grid] must point to caller-owned
 *   REAL diagnostic gridfunction data that serve as the sampling source.
 *
 * @pre
 *   - For each active grid, gridfuncs_diags[grid] is non-null and points to valid diagnostic data.
 *   - which_gfs indices selected in the USER-EDIT block map to valid diagnostic gridfunctions.
 *   - Helper symbols diagnostics_nearest_grid_center(), diagnostics_nearest_1d_y_and_z_axes(), and
 *     diagnostics_nearest_2d_xy_and_yz_planes() are available at link time.
 *
 * @post
 *   - For each grid, helper routines may emit 0D, 1D (y and z), and 2D (xy and yz) diagnostic outputs.
 *   - No memory is allocated or freed by this dispatcher.
 *
 * @return void
 *
 * @note The USER-EDIT block is for selecting which diagnostic gridfunctions to sample. Keep it concise and avoid
 *       per-grid logic there, as the dispatcher handles iteration over grids.
 *
 */
void diagnostics_nearest(commondata_struct *restrict commondata, griddata_struct *restrict griddata,
                         const REAL *restrict gridfuncs_diags[MAXNUMGRIDS]) {
  // --- USER-EDIT: Select diagnostic gridfunctions to sample (applies to all grids) ---

  // 0D diagnostics: nearest point to the grid center.
  const int which_gfs_0d[] = {DIAG_RESIDUALGF, DIAG_UUGF};

  // 1D diagnostics: nearest lines to the y and z axes.
  const int which_gfs_1d[] = {DIAG_RESIDUALGF, DIAG_UUGF};

  // 2D diagnostics: nearest planes to the xy and yz coordinate planes.
  const int which_gfs_2d[] = {DIAG_RESIDUALGF, DIAG_UUGF};

  // --- END USER-EDIT ---

  // Loop once over all grids and call the helpers using the selections above.
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    const params_struct *restrict params = &griddata[grid].params;
    const REAL *restrict xx[3] = {griddata[grid].xx[0], griddata[grid].xx[1], griddata[grid].xx[2]};

    // 0D
    const int NUM_nearest_GFS_0d = (int)(sizeof which_gfs_0d / sizeof which_gfs_0d[0]);
    diagnostics_nearest_grid_center(commondata, grid, params, xx, NUM_nearest_GFS_0d, which_gfs_0d, diagnostic_gf_names, gridfuncs_diags);

    // 1D
    const int NUM_nearest_GFS_1d = (int)(sizeof which_gfs_1d / sizeof which_gfs_1d[0]);
    diagnostics_nearest_1d_y_and_z_axes(commondata, grid, params, xx, NUM_nearest_GFS_1d, which_gfs_1d, diagnostic_gf_names, gridfuncs_diags);

    // 2D
    const int NUM_nearest_GFS_2d = (int)(sizeof which_gfs_2d / sizeof which_gfs_2d[0]);
    diagnostics_nearest_2d_xy_and_yz_planes(commondata, grid, params, xx, NUM_nearest_GFS_2d, which_gfs_2d, diagnostic_gf_names, gridfuncs_diags);
  } // END loop over grids
} // END FUNCTION diagnostics_nearest
