#include "BHaH_defines.h"
/**
 * Set commondata_struct to default values specified within NRPy+.
 */
void commondata_struct_set_to_default(commondata_struct *restrict commondata) {

  // Set commondata_struct variables to default
  commondata->CFL_FACTOR = 1.0;                         // nrpy.infrastructures.BHaH.MoLtimestepping.MoL_register_all::CFL_FACTOR
  commondata->MINIMUM_GLOBAL_WAVESPEED = 0.7;           // nrpy.equations.nrpyelliptic.CommonParams::MINIMUM_GLOBAL_WAVESPEED
  commondata->NUMGRIDS = 1;                             // nrpy.grid::NUMGRIDS
  commondata->P0_x = 0.0;                               // nrpy.equations.nrpyelliptic.CommonParams::P0_x
  commondata->P0_y = 0.0;                               // nrpy.equations.nrpyelliptic.CommonParams::P0_y
  commondata->P0_z = 0.0;                               // nrpy.equations.nrpyelliptic.CommonParams::P0_z
  commondata->P1_x = 0.0;                               // nrpy.equations.nrpyelliptic.CommonParams::P1_x
  commondata->P1_y = 0.0;                               // nrpy.equations.nrpyelliptic.CommonParams::P1_y
  commondata->P1_z = 0.0;                               // nrpy.equations.nrpyelliptic.CommonParams::P1_z
  commondata->S0_x = 0.0;                               // nrpy.equations.nrpyelliptic.CommonParams::S0_x
  commondata->S0_y = 0.0;                               // nrpy.equations.nrpyelliptic.CommonParams::S0_y
  commondata->S0_z = 0.8;                               // nrpy.equations.nrpyelliptic.CommonParams::S0_z
  commondata->S1_x = 0.0;                               // nrpy.equations.nrpyelliptic.CommonParams::S1_x
  commondata->S1_y = 0.0;                               // nrpy.equations.nrpyelliptic.CommonParams::S1_y
  commondata->S1_z = -0.6;                              // nrpy.equations.nrpyelliptic.CommonParams::S1_z
  commondata->bare_mass_0 = 0.5;                        // nrpy.equations.nrpyelliptic.CommonParams::bare_mass_0
  commondata->bare_mass_1 = 0.5;                        // nrpy.equations.nrpyelliptic.CommonParams::bare_mass_1
  commondata->checkpoint_every = 50.0;                  // nrpy.infrastructures.BHaH.checkpointing::checkpoint_every
  commondata->convergence_factor = 1.0;                 // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::convergence_factor
  commondata->diagnostics_output_every = 100;           // nrpy.infrastructures.BHaH.nrpyelliptic.diagnostics::diagnostics_output_every
  commondata->eta_damping = 11.0;                       // nrpy.equations.nrpyelliptic.CommonParams::eta_damping
  commondata->log10_current_residual = 1.0;             // nrpy.infrastructures.BHaH.nrpyelliptic.diagnostics::log10_current_residual
  commondata->log10_residual_tolerance = -15.8;         // nrpy.infrastructures.BHaH.nrpyelliptic.diagnostics::log10_residual_tolerance
  commondata->nn_max = 10000;                           // nrpy.infrastructures.BHaH.nrpyelliptic.diagnostics::nn_max
  commondata->output_progress_every = 1;                // nrpy.infrastructures.BHaH.diagnostics.progress_indicator::output_progress_every
  commondata->stop_relaxation = false;                  // nrpy.infrastructures.BHaH.nrpyelliptic.diagnostics::stop_relaxation
  commondata->t_final = 1000000.0;                      // nrpy.infrastructures.BHaH.MoLtimestepping.MoL_register_all::t_final
  commondata->zPunc = 2.5;                              // nrpy.equations.nrpyelliptic.CommonParams::zPunc
  snprintf(commondata->outer_bc_type, 50, "radiation"); // nrpy.infrastructures.BHaH.CurviBoundaryConditions.CurviBoundaryConditions::outer_bc_type
} // END FUNCTION commondata_struct_set_to_default
