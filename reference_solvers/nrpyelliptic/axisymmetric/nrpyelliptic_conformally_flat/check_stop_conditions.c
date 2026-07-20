#include "BHaH_defines.h"
/**
 * Evaluate stop conditions.
 */
void check_stop_conditions(commondata_struct *restrict commondata, griddata_struct *restrict griddata) {
  // Since this version of NRPyElliptic is unigrid, we simply set the grid index to 0
  const int grid = 0;

  // Set params
  params_struct *restrict params = &griddata[grid].params;
#include "set_CodeParameters.h"

  // Check if total number of iteration steps has been reached
  if ((nn >= nn_max) || (log10_current_residual < log10_residual_tolerance)) {
    printf("\nExiting main loop after %8d iterations\n", nn);
    printf("The tolerance for the logarithmic residual is %.8e\n", log10_residual_tolerance);
    printf("Exiting relaxation with logarithmic residual of %.8e\n", log10_current_residual);
    commondata->stop_relaxation = true;
  }
} // END FUNCTION check_stop_conditions
