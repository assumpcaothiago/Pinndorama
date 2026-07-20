#include "BHaH_defines.h"

/**
 * Set commondata_struct to default values specified within NRPy.
 */
void commondata_struct_set_to_default(commondata_struct *restrict commondata) {
  // Set commondata_struct variables to default
  commondata->CFL_FACTOR = 1.0;                  // nrpy.infrastructures.BHaH.MoLtimestepping.register_all::CFL_FACTOR
  commondata->MINIMUM_GLOBAL_WAVESPEED = 0.55;   // nrpy.infrastructures.BHaH.nrpyelliptic.auxevol_gfs_set_to_constant::MINIMUM_GLOBAL_WAVESPEED
  commondata->NUMGRIDS = 1;                      // nrpy.grid::NUMGRIDS
  commondata->P0_x = 0.09530152296974252;        // nrpy.equations.nrpyelliptic.ConformallyFlat_SourceTerms::P0_x
  commondata->P0_y = 0.0;                        // nrpy.equations.nrpyelliptic.ConformallyFlat_SourceTerms::P0_y
  commondata->P0_z = -0.00084541526517121;       // nrpy.equations.nrpyelliptic.ConformallyFlat_SourceTerms::P0_z
  commondata->P1_x = -0.09530152296974252;       // nrpy.equations.nrpyelliptic.ConformallyFlat_SourceTerms::P1_x
  commondata->P1_y = 0.0;                        // nrpy.equations.nrpyelliptic.ConformallyFlat_SourceTerms::P1_y
  commondata->P1_z = 0.00084541526517121;        // nrpy.equations.nrpyelliptic.ConformallyFlat_SourceTerms::P1_z
  commondata->S0_x = 0.0;                        // nrpy.equations.nrpyelliptic.ConformallyFlat_SourceTerms::S0_x
  commondata->S0_y = 0.09509112426035504;        // nrpy.equations.nrpyelliptic.ConformallyFlat_SourceTerms::S0_y
  commondata->S0_z = 0.0;                        // nrpy.equations.nrpyelliptic.ConformallyFlat_SourceTerms::S0_z
  commondata->S1_x = 0.0;                        // nrpy.equations.nrpyelliptic.ConformallyFlat_SourceTerms::S1_x
  commondata->S1_y = -0.09156449704142013;       // nrpy.equations.nrpyelliptic.ConformallyFlat_SourceTerms::S1_y
  commondata->S1_z = 0.0;                        // nrpy.equations.nrpyelliptic.ConformallyFlat_SourceTerms::S1_z
  commondata->bare_mass_0 = 0.5184199353358704;  // nrpy.equations.nrpyelliptic.ConformallyFlat_SourceTerms::bare_mass_0
  commondata->bare_mass_1 = 0.39193567996522616; // nrpy.equations.nrpyelliptic.ConformallyFlat_SourceTerms::bare_mass_1
  commondata->checkpoint_every = 50.0;           // nrpy.infrastructures.BHaH.checkpointing::checkpoint_every
  commondata->convergence_factor = 1.0;          // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::convergence_factor
  commondata->diagnostics_output_every = 0.08;   // nrpy.infrastructures.BHaH.diagnostics.diagnostics::diagnostics_output_every
  commondata->eta_damping = 70.0;                // nrpy.equations.nrpyelliptic.ConformallyFlat_RHSs::eta_damping
  commondata->log10_current_residual = 1.0;      // nrpy.infrastructures.BHaH.nrpyelliptic.stop_conditions_check::log10_current_residual
  commondata->log10_residual_tolerance = -15.9;  // nrpy.infrastructures.BHaH.nrpyelliptic.stop_conditions_check::log10_residual_tolerance
  commondata->nn_max = 30000;                    // nrpy.infrastructures.BHaH.nrpyelliptic.stop_conditions_check::nn_max
  commondata->output_progress_every = 1;         // nrpy.infrastructures.BHaH.diagnostics.progress_indicator::output_progress_every
  commondata->stop_relaxation = false;           // nrpy.infrastructures.BHaH.nrpyelliptic.stop_conditions_check::stop_relaxation
  commondata->t_final = 1000000.0;               // nrpy.infrastructures.BHaH.MoLtimestepping.register_all::t_final
  commondata->zPunc = 5.0;                       // nrpy.equations.nrpyelliptic.ConformallyFlat_SourceTerms::zPunc
  snprintf(commondata->initial_guess, 32, "zero"); // Publication initializer is opt-in.
  snprintf(commondata->outer_bc_type, 50, "radiation"); // nrpy.infrastructures.BHaH.CurviBoundaryConditions.register_all::outer_bc_type
} // END FUNCTION commondata_struct_set_to_default
