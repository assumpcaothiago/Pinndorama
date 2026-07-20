#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"

/**
 * Kernel: initial_data_host.
 * Kernel to initialize all grid points.
 */
static void initial_data_host(commondata_struct *restrict commondata, const params_struct *restrict params, const REAL *restrict x0, const REAL *restrict x1, const REAL *restrict x2,
                              const bool use_parametric_nn, REAL *restrict in_gfs) {
  MAYBE_UNUSED const int Nxx_plus_2NGHOSTS0 = params->Nxx_plus_2NGHOSTS0;
  MAYBE_UNUSED const int Nxx_plus_2NGHOSTS1 = params->Nxx_plus_2NGHOSTS1;
  MAYBE_UNUSED const int Nxx_plus_2NGHOSTS2 = params->Nxx_plus_2NGHOSTS2;

  MAYBE_UNUSED const REAL invdxx0 = params->invdxx0;
  MAYBE_UNUSED const REAL invdxx1 = params->invdxx1;
  MAYBE_UNUSED const REAL invdxx2 = params->invdxx2;

#pragma omp parallel for
  for (int i2 = 0; i2 < Nxx_plus_2NGHOSTS2; i2++) {
    MAYBE_UNUSED const REAL xx2 = x2[i2];
    for (int i1 = 0; i1 < Nxx_plus_2NGHOSTS1; i1++) {
      MAYBE_UNUSED const REAL xx1 = x1[i1];
      for (int i0 = 0; i0 < Nxx_plus_2NGHOSTS0; i0++) {
        MAYBE_UNUSED const REAL xx0 = x0[i0];

        if (use_parametric_nn) {
          neural_net_guess_single_point(commondata, params, xx0, xx1, xx2, &in_gfs[IDX4(UUGF, i0, i1, i2)], &in_gfs[IDX4(VVGF, i0, i1, i2)]);
        } else {
          initial_guess_single_point(xx0, xx1, xx2, &in_gfs[IDX4(UUGF, i0, i1, i2)], &in_gfs[IDX4(VVGF, i0, i1, i2)]);
        }

      } // END LOOP: for (int i0 = 0; i0 < Nxx_plus_2NGHOSTS0; i0++)
    } // END LOOP: for (int i1 = 0; i1 < Nxx_plus_2NGHOSTS1; i1++)
  } // END LOOP: for (int i2 = 0; i2 < Nxx_plus_2NGHOSTS2; i2++)
} // END FUNCTION initial_data_host

/**
 * Set initial guess to solutions of hyperbolic relaxation equation at all points.
 */
void initial_data(commondata_struct *restrict commondata, griddata_struct *restrict griddata) {
  // Attempt to read checkpoint file. If it doesn't exist, then continue. Otherwise return.
  if (read_checkpoint(commondata, griddata))
    return;

  bool use_parametric_nn = false;
  if (strcmp(commondata->initial_guess, "zero") == 0) {
    fprintf(stderr, "Using zero initial guess.\n");
  } else if (strcmp(commondata->initial_guess, "parametric_nn") == 0) {
    use_parametric_nn = true;
    // All grids share the immutable SinhSymTP geometry. Validate the checkpoint
    // provenance and runtime physics exactly once, before any OpenMP loop.
    validate_parametric_nn_compatibility(commondata, &griddata[0].params);
  } else {
    fprintf(stderr, "ERROR: initial_guess must be \"zero\" or \"parametric_nn\"; got \"%s\"\n", commondata->initial_guess);
    exit(1);
  }
  for (int grid = 0; grid < commondata->NUMGRIDS; grid++) {
    // Unpack griddata struct:
    params_struct *restrict params = &griddata[grid].params;
    REAL *restrict x0 = griddata[grid].xx[0];
    REAL *restrict x1 = griddata[grid].xx[1];
    REAL *restrict x2 = griddata[grid].xx[2];
    REAL *restrict in_gfs = griddata[grid].gridfuncs.y_n_gfs;
    initial_data_host(commondata, params, x0, x1, x2, use_parametric_nn, in_gfs);
  }
} // END FUNCTION initial_data
