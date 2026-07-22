#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
#include "diagnostics/diagnostics_nearest_common.h"
#include "stdlib.h"

/**
 * @file diagnostics_nearest_2d_xy_and_yz_planes.c
 * @brief Sample and write 2D diagnostics on the nearest xy and yz planes for a single SinhSymTP grid.
 *
 * Overview:
 * For the specified grid at the current time, this routine:
 *  - Locates interior slices that are nearest to the xy and yz planes, with selection rules
 *    specialized by the runtime coordinate system.
 *  - Converts native coordinates xx to Cartesian coordinates (x, y, z) using xx_to_Cart so that
 *    the first columns of each row contain mapped coordinates.
 *  - Writes two files per call, one for the xy plane and one for the yz plane. Each file begins
 *    with a time comment and a header, followed by one row per interior point that contains the
 *    mapped coordinates and sampled diagnostic values.
 *  - Performs sampling without interpolation; values are read directly from gridfuncs_diags[grid].
 *
 * Plane selection notes (examples, not exhaustive):
 *  - Cartesian: xy fixes i2 at mid; yz fixes i0 at mid.
 *  - Cylindrical: xy fixes z at mid; yz emits two phi slices near +/- pi/2 to realize x=0.
 *  - Spherical and SymTP: xy fixes the polar-like angle at mid; yz emits two phi-like slices near +/- pi/2.
 *  - Wedge: xy may emit two z-like slices at quarter indices; yz fixes the across-wedge index at mid.
 *
 * If a user-editable block is provided in the implementation, users may insert custom logic such as
 * adding extra columns or filtering before rows are written.
 *
 * @param[in,out] commondata            Pointer to common runtime data used for time and I/O.
 * @param[in]     grid                  Grid index to process.
 * @param[in]     params                Pointer to simulation and grid parameters (sizes, names, strides).
 * @param[in]     xx                    Native grid coordinates; xx[d][i_d] gives the coordinate along dimension d.
 * @param[in]     NUM_GFS_NEAREST       Number of diagnostic gridfunctions to sample at each interior point.
 * @param[in]     which_gfs             Array of length NUM_GFS_NEAREST specifying which gridfunctions to sample.
 * @param[in]     diagnostic_gf_names   Array of length NUM_GFS_NEAREST with human-readable names for headers.
 * @param[in]     gridfuncs_diags       Array of pointers; gridfuncs_diags[grid] points to this grid's diagnostic data.
 *
 * @return        void                  No return value. On success two text files are written and closed. Fatal I/O
 *                                      or allocation failures result in program termination.
 */
void diagnostics_nearest_2d_xy_and_yz_planes__rfm__SinhSymTP(commondata_struct *restrict commondata, const int grid,
                                                             const params_struct *restrict params, const REAL *restrict xx[3],
                                                             const int NUM_GFS_NEAREST, const int which_gfs[], const char **diagnostic_gf_names,
                                                             const REAL *restrict gridfuncs_diags[]) {
#include "set_CodeParameters.h"
  // Build filename component with runtime coordinate system name and grid number
  char coordsys_with_grid[128];
  snprintf(coordsys_with_grid, sizeof(coordsys_with_grid), "%s-grid%02d", params->CoordSystemName, grid);

  // Open output files (one file per timestep, per plane, for this grid)
  FILE *out_xy = open_outfile("out2d-xy", coordsys_with_grid, commondata, /*include_time=*/1);
  FILE *out_yz = open_outfile("out2d-yz", coordsys_with_grid, commondata, /*include_time=*/1);

  if (!out_xy || !out_yz) {
    if (out_xy)
      fclose(out_xy);
    if (out_yz)
      fclose(out_yz);
    fprintf(stderr, "Error: Cannot open output files for grid %d.\n", grid);
    exit(1);
  } // END IF cannot open output files

  // Write time comment and headers
  diag_write_time_comment(out_xy, commondata->time);
  diag_write_header(out_xy, "x y", NUM_GFS_NEAREST, which_gfs, diagnostic_gf_names);
  diag_write_time_comment(out_yz, commondata->time);
  diag_write_header(out_yz, "y z", NUM_GFS_NEAREST, which_gfs, diagnostic_gf_names);

  // Active grid data pointer and reusable row buffer
  const REAL *restrict src = gridfuncs_diags[grid];
  const int NUM_COLS = 2 + NUM_GFS_NEAREST;
  REAL *row = (REAL *)malloc(sizeof(REAL) * (size_t)NUM_COLS);
  if (!row) {
    fprintf(stderr, "Error: Failed to allocate memory for row buffer.\n");
    exit(1);
  } // END IF row allocation failure

  // Interior grid counts and loop bounds
  MAYBE_UNUSED const int N0int = params->Nxx_plus_2NGHOSTS0 - 2 * NGHOSTS;
  MAYBE_UNUSED const int N1int = params->Nxx_plus_2NGHOSTS1 - 2 * NGHOSTS;
  MAYBE_UNUSED const int N2int = params->Nxx_plus_2NGHOSTS2 - 2 * NGHOSTS;
  const int i0_end = params->Nxx_plus_2NGHOSTS0 - NGHOSTS;
  const int i1_end = params->Nxx_plus_2NGHOSTS1 - NGHOSTS;
  const int i2_end = params->Nxx_plus_2NGHOSTS2 - NGHOSTS;

  // Fixed-point index helpers
  MAYBE_UNUSED const int i0_mid = params->Nxx_plus_2NGHOSTS0 / 2;
  MAYBE_UNUSED const int i1_mid = params->Nxx_plus_2NGHOSTS1 / 2;
  MAYBE_UNUSED const int i2_mid = params->Nxx_plus_2NGHOSTS2 / 2;
  MAYBE_UNUSED const int i1_q1 = (int)(NGHOSTS + 0.25 * (REAL)N1int - 0.5);
  MAYBE_UNUSED const int i1_q3 = (int)(NGHOSTS + 0.75 * (REAL)N1int - 0.5);
  MAYBE_UNUSED const int i2_q1 = (int)(NGHOSTS + 0.25 * (REAL)N2int - 0.5);
  MAYBE_UNUSED const int i2_q3 = (int)(NGHOSTS + 0.75 * (REAL)N2int - 0.5);

  // --- Sample and write data for the xy-plane for SinhSymTP ---
  {
    const int i1 = i1_mid;
    for (int i2 = NGHOSTS; i2 < i2_end; i2++) {
      for (int i0 = NGHOSTS; i0 < i0_end; i0++) {
        const int idx3 = IDX3P(params, i0, i1, i2);
        REAL xCart[3], xOrig[3] = {xx[0][i0], xx[1][i1], xx[2][i2]};
        xx_to_Cart(params, xOrig, xCart);
        row[0] = xCart[0];
        row[1] = xCart[1];
        for (int gf_idx = 0; gf_idx < NUM_GFS_NEAREST; gf_idx++) {
          const int gf = which_gfs[gf_idx];
          row[2 + gf_idx] = src[IDX4Ppt(params, gf, idx3)];
        } // END LOOP over gridfunctions
        diag_write_row(out_xy, 2 + NUM_GFS_NEAREST, row);
      } // END LOOP over i0
    } // END LOOP over i2
  } // END BLOCK xy-plane output

  // --- Sample and write data for the yz-plane for SinhSymTP ---

  {
    const int i2_slices[2] = {i2_q1, i2_q3};
    for (int slice = 0; slice < 2; slice++) {
      const int i2 = i2_slices[slice];
      for (int i1 = NGHOSTS; i1 < i1_end; i1++) {
        for (int i0 = NGHOSTS; i0 < i0_end; i0++) {
          const int idx3 = IDX3P(params, i0, i1, i2);
          REAL xCart[3], xOrig[3] = {xx[0][i0], xx[1][i1], xx[2][i2]};
          xx_to_Cart(params, xOrig, xCart);
          row[0] = xCart[1];
          row[1] = xCart[2];
          for (int gf_idx = 0; gf_idx < NUM_GFS_NEAREST; gf_idx++) {
            const int gf = which_gfs[gf_idx];
            row[2 + gf_idx] = src[IDX4Ppt(params, gf, idx3)];
          } // END LOOP over gridfunctions
          diag_write_row(out_yz, 2 + NUM_GFS_NEAREST, row);
        } // END LOOP over i0
      } // END LOOP over i1

    } // END LOOP over slices
  } // END BLOCK yz-plane output

  // Finalize
  free(row);
  fclose(out_xy);
  fclose(out_yz);
} // END FUNCTION diagnostics_nearest_2d_xy_and_yz_planes__rfm__SinhSymTP
