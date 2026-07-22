#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
#include "diagnostics/diagnostics_nearest_common.h"

// Data point for sorting by physical axis coordinate
typedef struct {
  REAL coord; // physical y or z
  int idx3;   // 3D index
} data_point_1d_struct;

// qsort comparator (file scope to satisfy -std=c11)
/**
 * @brief Compare two data_point_1d_struct items by their coord value.
 * @param a Pointer to the left-hand data_point_1d_struct.
 * @param b Pointer to the right-hand data_point_1d_struct.
 * @return Negative if a < b, zero if equal, positive if a > b.
 */
static int compare_by_coord(const void *a, const void *b) {
  const REAL lv = ((const data_point_1d_struct *)a)->coord;
  const REAL rv = ((const data_point_1d_struct *)b)->coord;
  return (lv > rv) - (lv < rv);
} // END FUNCTION compare_by_coord

/**
 * @file diagnostics_nearest_1d_y_and_z_axes.c
 * @brief Write 1D diagnostics for a single grid by sampling, without interpolation, axis-aligned
 *        lines nearest to the physical y and z axes, and append rows per timestep to persistent files.
 *
 * The function "diagnostics_nearest_1d_y_and_z_axes" appends diagnostics for a single grid to two
 * per-grid text files whose names encode the runtime coordinate system, grid number, and convergence
 * factor:
 *
 *   out1d-y-<CoordSystemName>-grid<XX>-conv_factor-<CF>.txt
 *   out1d-z-<CoordSystemName>-grid<XX>-conv_factor-<CF>.txt
 *
 * For each axis, the routine:
 *   1) Selects one or more index-line samples based on the coordinate family (e.g., Cartesian,
 *      Spherical, Cylindrical, SymTP, Wedge, Spherical_Ring).
 *   2) Converts logical coordinates xx to Cartesian via xx_to_Cart and extracts y (xCart[1]) or
 *      z (xCart[2]) as the axis coordinate.
 *   3) Buffers (axis_coord, idx3) pairs, then sorts them in ascending order using qsort.
 *   4) Writes a single-line time comment, an axis-specific header ("y" or "z"), and streams rows:
 *      [axis_coord, values of selected diagnostic gridfunctions].
 *
 * Gridfunction values are loaded from the flattened diagnostic array using IDX4Ppt with a 3D index
 * constructed by IDX3P. Sampling occurs at grid points only; no interpolation is performed.
 * On allocation or file-open failure, the routine prints an error message to stderr and terminates.
 * If a user-editable block is provided in the implementation, users may add custom logic such as
 * additional columns or filtering before rows are written.
 *
 * @param[in]  commondata           Pointer to global simulation metadata (e.g., time and step counters).
 * @param[in]  grid                 Zero-based grid index used for selecting data and for file naming.
 * @param[in]  params               Pointer to per-grid parameters (sizes, ghost zones, strides, names).
 * @param[in]  xx                   Array of 3 pointers to logical coordinates per dimension; used for
 *                                  coordinate conversion to Cartesian space.
 * @param[in]  NUM_GFS_NEAREST      Number of gridfunctions to include per output row.
 * @param[in]  which_gfs            Array of length NUM_GFS_NEAREST identifying the gridfunction indices.
 * @param[in]  diagnostic_gf_names  Array of NUM_GFS_NEAREST C strings used to construct column headers.
 * @param[in]  gridfuncs_diags      Array of per-grid pointers to flattened diagnostic data; this routine
 *                                  reads from gridfuncs_diags[grid].
 *
 * @return     void
 */
void diagnostics_nearest_1d_y_and_z_axes__rfm__SinhSymTP(commondata_struct *restrict commondata, const int grid, const params_struct *restrict params,
                                                         const REAL *restrict xx[3], const int NUM_GFS_NEAREST, const int which_gfs[],
                                                         const char **diagnostic_gf_names, const REAL *restrict gridfuncs_diags[]) {
#include "set_CodeParameters.h"
  // Interior counts
  MAYBE_UNUSED const int N0int = params->Nxx_plus_2NGHOSTS0 - 2 * NGHOSTS;
  MAYBE_UNUSED const int N1int = params->Nxx_plus_2NGHOSTS1 - 2 * NGHOSTS;
  MAYBE_UNUSED const int N2int = params->Nxx_plus_2NGHOSTS2 - 2 * NGHOSTS;

  // Common fixed-point helpers
  MAYBE_UNUSED const int i0_mid = params->Nxx_plus_2NGHOSTS0 / 2;
  MAYBE_UNUSED const int i1_mid = params->Nxx_plus_2NGHOSTS1 / 2;
  MAYBE_UNUSED const int i2_mid = params->Nxx_plus_2NGHOSTS2 / 2;

  MAYBE_UNUSED const int i1_min = NGHOSTS;
  MAYBE_UNUSED const int i1_max = params->Nxx_plus_2NGHOSTS1 - NGHOSTS - 1;
  MAYBE_UNUSED const int i2_min = NGHOSTS;

  MAYBE_UNUSED const int i0_rmin = NGHOSTS; // rho = 0 (Cylindrical)
  MAYBE_UNUSED const int i1_pmin = NGHOSTS; // phi = -pi

  // Quarter-plane indices for cell-centered grids
  MAYBE_UNUSED const int i2_q1 = (int)(NGHOSTS + 0.25 * (REAL)N2int - 0.5);
  MAYBE_UNUSED const int i2_q3 = (int)(NGHOSTS + 0.75 * (REAL)N2int - 0.5);
  MAYBE_UNUSED const int i1_q1 = (int)(NGHOSTS + 0.25 * (REAL)N1int - 0.5);
  MAYBE_UNUSED const int i1_q3 = (int)(NGHOSTS + 0.75 * (REAL)N1int - 0.5);

  // File naming: out1d-AXIS-<CoordSystemName>-gridXX-...
  char coordsys_with_grid[128];
  snprintf(coordsys_with_grid, sizeof(coordsys_with_grid), "%s-grid%02d", params->CoordSystemName, grid);

  FILE *out_y = open_outfile("out1d-y", coordsys_with_grid, commondata, /*include_time=*/1);
  FILE *out_z = open_outfile("out1d-z", coordsys_with_grid, commondata, /*include_time=*/1);
  if (!out_y || !out_z) {
    if (out_y)
      fclose(out_y);
    if (out_z)
      fclose(out_z);
    fprintf(stderr, "Error: Cannot open output files for grid %d.\n", grid);
    exit(1);
  } // END IF file open failure

  // Emit time comment then axis-specific headers (no 'time' column)
  diag_write_time_comment(out_y, commondata->time);
  diag_write_time_comment(out_z, commondata->time);
  diag_write_header(out_y, "y", NUM_GFS_NEAREST, which_gfs, diagnostic_gf_names);
  diag_write_header(out_z, "z", NUM_GFS_NEAREST, which_gfs, diagnostic_gf_names);

  // Source pointer for this grid
  const REAL *restrict src = gridfuncs_diags[grid];

  // Allocate buffers (tight upper bounds)
  const int max_y = N0int + N0int;
  const int max_z = N0int + N0int;
  data_point_1d_struct *data_points_y = max_y > 0 ? (data_point_1d_struct *)malloc(sizeof(data_point_1d_struct) * (size_t)max_y) : NULL;
  data_point_1d_struct *data_points_z = max_z > 0 ? (data_point_1d_struct *)malloc(sizeof(data_point_1d_struct) * (size_t)max_z) : NULL;

  // Row buffer: [axis_coord, gfs...]
  REAL *row = (REAL *)malloc(sizeof(REAL) * (size_t)(1 + NUM_GFS_NEAREST));
  if ((max_y > 0 && !data_points_y) || (max_z > 0 && !data_points_z) || !row) {
    fprintf(stderr, "Error: Allocation failure in diagnostics_nearest_1d_y_and_z_axes.\\n");
    free(data_points_y);
    free(data_points_z);
    free(row);
    exit(1);
  } // END IF allocation failure

  // ----------------------
  // Build y-axis samples
  // ----------------------
  int count_y = 0;
  for (int i0 = NGHOSTS; i0 < params->Nxx_plus_2NGHOSTS0 - NGHOSTS; i0++) {
    const int i1 = i1_mid;
    const int i2 = i2_q1;
    const int idx3 = IDX3P(params, i0, i1, i2);
    REAL xCart[3], xOrig[3] = {xx[0][i0], xx[1][i1], xx[2][i2]};
    xx_to_Cart(params, xOrig, xCart);
    data_points_y[count_y].coord = xCart[1];
    data_points_y[count_y].idx3 = idx3;
    count_y++;
  } // END LOOP over i0
  for (int i0 = NGHOSTS; i0 < params->Nxx_plus_2NGHOSTS0 - NGHOSTS; i0++) {
    const int i1 = i1_mid;
    const int i2 = i2_q3;
    const int idx3 = IDX3P(params, i0, i1, i2);
    REAL xCart[3], xOrig[3] = {xx[0][i0], xx[1][i1], xx[2][i2]};
    xx_to_Cart(params, xOrig, xCart);
    data_points_y[count_y].coord = xCart[1];
    data_points_y[count_y].idx3 = idx3;
    count_y++;
  } // END LOOP over i0

  if (count_y > 1)
    qsort(data_points_y, (size_t)count_y, sizeof(data_point_1d_struct), compare_by_coord);
  for (int p = 0; p < count_y; ++p) {
    row[0] = data_points_y[p].coord;
    const int idx3 = data_points_y[p].idx3;
    for (int gf_i = 0; gf_i < NUM_GFS_NEAREST; ++gf_i) {
      const int gf = which_gfs[gf_i];
      row[1 + gf_i] = src[IDX4Ppt(params, gf, idx3)];
    } // END LOOP over gridfunctions
    diag_write_row(out_y, 1 + NUM_GFS_NEAREST, row);
  } // END LOOP over *sorted* points closest to y-axis.

  // ----------------------
  // Build z-axis samples
  // ----------------------
  int count_z = 0;
  for (int i0 = NGHOSTS; i0 < params->Nxx_plus_2NGHOSTS0 - NGHOSTS; i0++) {
    const int i1 = i1_min;
    const int i2 = i2_min;
    const int idx3 = IDX3P(params, i0, i1, i2);
    REAL xCart[3], xOrig[3] = {xx[0][i0], xx[1][i1], xx[2][i2]};
    xx_to_Cart(params, xOrig, xCart);
    data_points_z[count_z].coord = xCart[2];
    data_points_z[count_z].idx3 = idx3;
    count_z++;
  } // END LOOP over i0
  for (int i0 = NGHOSTS; i0 < params->Nxx_plus_2NGHOSTS0 - NGHOSTS; i0++) {
    const int i1 = i1_max;
    const int i2 = i2_min;
    const int idx3 = IDX3P(params, i0, i1, i2);
    REAL xCart[3], xOrig[3] = {xx[0][i0], xx[1][i1], xx[2][i2]};
    xx_to_Cart(params, xOrig, xCart);
    data_points_z[count_z].coord = xCart[2];
    data_points_z[count_z].idx3 = idx3;
    count_z++;
  } // END LOOP over i0

  if (count_z > 1)
    qsort(data_points_z, (size_t)count_z, sizeof(data_point_1d_struct), compare_by_coord);
  for (int p = 0; p < count_z; ++p) {
    row[0] = data_points_z[p].coord;
    const int idx3 = data_points_z[p].idx3;
    for (int gf_i = 0; gf_i < NUM_GFS_NEAREST; ++gf_i) {
      const int gf = which_gfs[gf_i];
      row[1 + gf_i] = src[IDX4Ppt(params, gf, idx3)];
    } // END LOOP over gridfunctions
    diag_write_row(out_z, 1 + NUM_GFS_NEAREST, row);
  } // END LOOP over *sorted* points closest to z-axis.

  // Cleanup
  free(data_points_y);
  free(data_points_z);
  free(row);
  fclose(out_y);
  fclose(out_z);
} // END FUNCTION diagnostics_nearest_1d_y_and_z_axes__rfm__SinhSymTP
