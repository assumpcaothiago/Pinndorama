#include "../BHaH_defines.h"
#include "../BHaH_function_prototypes.h"
/**
 * Initializes a cell-centered grid in SinhSymTP coordinates based on physical dimensions (grid_physical_size).
 *
 * Inputs:
 * - Nx[] inputs: Specifies new grid dimensions, if needed.
 * - params.convergence_factor (set to 1.0 by default): Factor by which grid resolution is increased; set to 1.0 by default.
 * - set_xxmin_xxmax_to_defaults: Whether to set xxmin[3], xxmax[3] to default values set in reference_metric.py.
 *
 * Parameter outputs:
 * - Nxx: Number of grid points in each direction.
 * - Nxx_plus_2NGHOSTS: Total grid points including ghost zones.
 * - dxx: Grid spacing.
 * - invdxx: Inverse of grid spacing.
 *
 * Grid setup output:
 * - xx: Coordinate values for each (cell-centered) grid point.
 *
 */
void numerical_grid_params_Nxx_dxx_xx__rfm__SinhSymTP(const commondata_struct *restrict commondata, params_struct *restrict params,
                                                      REAL *restrict xx[3], const int Nx[3], const bool set_xxmin_xxmax_to_defaults) {
  // Set default values for the grid resolution in each dimension.
  params->Nxx0 = 256;
  params->Nxx1 = 256;
  params->Nxx2 = 2;

  // If all components of Nx[] are set to valid values (i.e., not -1), override the default values with Nx[].
  if (Nx[0] != -1 && Nx[1] != -1 && Nx[2] != -1) {
    params->Nxx0 = Nx[0];
    params->Nxx1 = Nx[1];
    params->Nxx2 = Nx[2];
  }
  snprintf(params->CoordSystemName, 50, "SinhSymTP");

  // Resize grid by convergence_factor; used for convergence testing.
  {
    // convergence_factor does not increase resolution across an axis of symmetry (Nxx == 2):
    if (params->Nxx0 != 2)
      params->Nxx0 *= commondata->convergence_factor;
    if (params->Nxx1 != 2)
      params->Nxx1 *= commondata->convergence_factor;
    if (params->Nxx2 != 2)
      params->Nxx2 *= commondata->convergence_factor;
  }

  // Set the full grid size; including the ghostzones (of width NGHOSTS) on the boundaries.
  params->Nxx_plus_2NGHOSTS0 = params->Nxx0 + 2 * NGHOSTS;
  params->Nxx_plus_2NGHOSTS1 = params->Nxx1 + 2 * NGHOSTS;
  params->Nxx_plus_2NGHOSTS2 = params->Nxx2 + 2 * NGHOSTS;

  {
#include "../set_CodeParameters.h"
    // Set grid size to a function of grid_physical_size (grid_physical_size set in set_CodeParameters.h above):
    params->AMAX = grid_physical_size;
  }
  if (set_xxmin_xxmax_to_defaults) {
#include "../set_CodeParameters.h"
    // Set {xxmin[], xxmax[]} to default values, which could be functions of other rfm params (set in set_CodeParameters.h above):
    params->xxmin0 = 0;
    params->xxmin1 = 0;
    params->xxmin2 = -PI;
    params->xxmax0 = 1;
    params->xxmax1 = PI;
    params->xxmax2 = PI;
  }

  // Set quantities that depend on Nxx and {xxmin, xxmax}: dxx, invdxx.
  params->dxx0 = (params->xxmax0 - params->xxmin0) / ((REAL)params->Nxx0);
  params->dxx1 = (params->xxmax1 - params->xxmin1) / ((REAL)params->Nxx1);
  params->dxx2 = (params->xxmax2 - params->xxmin2) / ((REAL)params->Nxx2);

  params->invdxx0 = ((REAL)params->Nxx0) / (params->xxmax0 - params->xxmin0);
  params->invdxx1 = ((REAL)params->Nxx1) / (params->xxmax1 - params->xxmin1);
  params->invdxx2 = ((REAL)params->Nxx2) / (params->xxmax2 - params->xxmin2);

  // Set up uniform, cell-centered, topologically Cartesian numerical grid,
  //   centered at (xxmin[i] + xxmax[i])/2 in direction i, and store
  //   {xx[0], xx[1], xx[2]} arrays.
  xx[0] = (REAL *restrict)malloc(sizeof(REAL) * params->Nxx_plus_2NGHOSTS0);
  xx[1] = (REAL *restrict)malloc(sizeof(REAL) * params->Nxx_plus_2NGHOSTS1);
  xx[2] = (REAL *restrict)malloc(sizeof(REAL) * params->Nxx_plus_2NGHOSTS2);
  for (int j = 0; j < params->Nxx_plus_2NGHOSTS0; j++)
    xx[0][j] = params->xxmin0 + ((REAL)(j - NGHOSTS) + (1.0 / 2.0)) * params->dxx0;
  for (int j = 0; j < params->Nxx_plus_2NGHOSTS1; j++)
    xx[1][j] = params->xxmin1 + ((REAL)(j - NGHOSTS) + (1.0 / 2.0)) * params->dxx1;
  for (int j = 0; j < params->Nxx_plus_2NGHOSTS2; j++)
    xx[2][j] = params->xxmin2 + ((REAL)(j - NGHOSTS) + (1.0 / 2.0)) * params->dxx2;
} // END FUNCTION numerical_grid_params_Nxx_dxx_xx__rfm__SinhSymTP
