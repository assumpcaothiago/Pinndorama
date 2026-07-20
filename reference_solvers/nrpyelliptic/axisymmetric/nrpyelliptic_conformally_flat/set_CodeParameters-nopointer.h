MAYBE_UNUSED const REAL AMAX = params.AMAX;                             // nrpy.reference_metric_SinhSymTP::AMAX
MAYBE_UNUSED const REAL bare_mass_0 = commondata.bare_mass_0;           // nrpy.equations.nrpyelliptic.CommonParams::bare_mass_0
MAYBE_UNUSED const REAL bare_mass_1 = commondata.bare_mass_1;           // nrpy.equations.nrpyelliptic.CommonParams::bare_mass_1
MAYBE_UNUSED const REAL bScale = params.bScale;                         // nrpy.reference_metric_SinhSymTP::bScale
MAYBE_UNUSED const REAL Cart_originx = params.Cart_originx;             // nrpy.grid::Cart_originx
MAYBE_UNUSED const REAL Cart_originy = params.Cart_originy;             // nrpy.grid::Cart_originy
MAYBE_UNUSED const REAL Cart_originz = params.Cart_originz;             // nrpy.grid::Cart_originz
MAYBE_UNUSED const REAL CFL_FACTOR = commondata.CFL_FACTOR;             // nrpy.infrastructures.BHaH.MoLtimestepping.MoL_register_all::CFL_FACTOR
MAYBE_UNUSED const REAL checkpoint_every = commondata.checkpoint_every; // nrpy.infrastructures.BHaH.checkpointing::checkpoint_every
MAYBE_UNUSED const REAL convergence_factor =
    commondata.convergence_factor;                                 // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::convergence_factor
MAYBE_UNUSED const int CoordSystem_hash = params.CoordSystem_hash; // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::CoordSystem_hash
char CoordSystemName[50];                                          // nrpy.reference_metric::CoordSystemName
{
  // Safely copy string with snprintf, which guarantees null termination
  snprintf(CoordSystemName, sizeof(CoordSystemName), "%s", params.CoordSystemName);
}
MAYBE_UNUSED const int diagnostics_output_every =
    commondata.diagnostics_output_every;                            // nrpy.infrastructures.BHaH.nrpyelliptic.diagnostics::diagnostics_output_every
MAYBE_UNUSED const REAL dt = commondata.dt;                         // nrpy.infrastructures.BHaH.MoLtimestepping.MoL_register_all::dt
MAYBE_UNUSED const REAL dxx0 = params.dxx0;                         // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::dxx0
MAYBE_UNUSED const REAL dxx1 = params.dxx1;                         // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::dxx1
MAYBE_UNUSED const REAL dxx2 = params.dxx2;                         // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::dxx2
MAYBE_UNUSED const REAL eta_damping = commondata.eta_damping;       // nrpy.equations.nrpyelliptic.CommonParams::eta_damping
MAYBE_UNUSED const REAL grid_hole_radius = params.grid_hole_radius; // nrpy.reference_metric::grid_hole_radius
MAYBE_UNUSED const int grid_idx = params.grid_idx;                  // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::grid_idx
MAYBE_UNUSED const REAL grid_physical_size = params.grid_physical_size; // nrpy.reference_metric::grid_physical_size
MAYBE_UNUSED const bool grid_rotates = params.grid_rotates;             // nrpy.grid::grid_rotates
MAYBE_UNUSED const REAL invdxx0 = params.invdxx0;                       // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::invdxx0
MAYBE_UNUSED const REAL invdxx1 = params.invdxx1;                       // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::invdxx1
MAYBE_UNUSED const REAL invdxx2 = params.invdxx2;                       // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::invdxx2
MAYBE_UNUSED const REAL log10_current_residual =
    commondata.log10_current_residual; // nrpy.infrastructures.BHaH.nrpyelliptic.diagnostics::log10_current_residual
MAYBE_UNUSED const REAL log10_residual_tolerance =
    commondata.log10_residual_tolerance; // nrpy.infrastructures.BHaH.nrpyelliptic.diagnostics::log10_residual_tolerance
MAYBE_UNUSED const REAL MINIMUM_GLOBAL_WAVESPEED =
    commondata.MINIMUM_GLOBAL_WAVESPEED;                               // nrpy.equations.nrpyelliptic.CommonParams::MINIMUM_GLOBAL_WAVESPEED
MAYBE_UNUSED const int nn = commondata.nn;                             // nrpy.infrastructures.BHaH.MoLtimestepping.MoL_register_all::nn
MAYBE_UNUSED const int nn_0 = commondata.nn_0;                         // nrpy.infrastructures.BHaH.MoLtimestepping.MoL_register_all::nn_0
MAYBE_UNUSED const int nn_max = commondata.nn_max;                     // nrpy.infrastructures.BHaH.nrpyelliptic.diagnostics::nn_max
MAYBE_UNUSED const int NUMGRIDS = commondata.NUMGRIDS;                 // nrpy.grid::NUMGRIDS
MAYBE_UNUSED const int Nxx0 = params.Nxx0;                             // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::Nxx0
MAYBE_UNUSED const int Nxx1 = params.Nxx1;                             // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::Nxx1
MAYBE_UNUSED const int Nxx2 = params.Nxx2;                             // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::Nxx2
MAYBE_UNUSED const int Nxx_plus_2NGHOSTS0 = params.Nxx_plus_2NGHOSTS0; // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::Nxx_plus_2NGHOSTS0
MAYBE_UNUSED const int Nxx_plus_2NGHOSTS1 = params.Nxx_plus_2NGHOSTS1; // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::Nxx_plus_2NGHOSTS1
MAYBE_UNUSED const int Nxx_plus_2NGHOSTS2 = params.Nxx_plus_2NGHOSTS2; // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::Nxx_plus_2NGHOSTS2
char outer_bc_type[50]; // nrpy.infrastructures.BHaH.CurviBoundaryConditions.CurviBoundaryConditions::outer_bc_type
{
  // Safely copy string with snprintf, which guarantees null termination
  snprintf(outer_bc_type, sizeof(outer_bc_type), "%s", commondata.outer_bc_type);
}
MAYBE_UNUSED const REAL P0_x = commondata.P0_x;                       // nrpy.equations.nrpyelliptic.CommonParams::P0_x
MAYBE_UNUSED const REAL P0_y = commondata.P0_y;                       // nrpy.equations.nrpyelliptic.CommonParams::P0_y
MAYBE_UNUSED const REAL P0_z = commondata.P0_z;                       // nrpy.equations.nrpyelliptic.CommonParams::P0_z
MAYBE_UNUSED const REAL P1_x = commondata.P1_x;                       // nrpy.equations.nrpyelliptic.CommonParams::P1_x
MAYBE_UNUSED const REAL P1_y = commondata.P1_y;                       // nrpy.equations.nrpyelliptic.CommonParams::P1_y
MAYBE_UNUSED const REAL P1_z = commondata.P1_z;                       // nrpy.equations.nrpyelliptic.CommonParams::P1_z
MAYBE_UNUSED const REAL PI = params.PI;                               // nrpy.reference_metric::PI
MAYBE_UNUSED const REAL S0_x = commondata.S0_x;                       // nrpy.equations.nrpyelliptic.CommonParams::S0_x
MAYBE_UNUSED const REAL S0_y = commondata.S0_y;                       // nrpy.equations.nrpyelliptic.CommonParams::S0_y
MAYBE_UNUSED const REAL S0_z = commondata.S0_z;                       // nrpy.equations.nrpyelliptic.CommonParams::S0_z
MAYBE_UNUSED const REAL S1_x = commondata.S1_x;                       // nrpy.equations.nrpyelliptic.CommonParams::S1_x
MAYBE_UNUSED const REAL S1_y = commondata.S1_y;                       // nrpy.equations.nrpyelliptic.CommonParams::S1_y
MAYBE_UNUSED const REAL S1_z = commondata.S1_z;                       // nrpy.equations.nrpyelliptic.CommonParams::S1_z
MAYBE_UNUSED const REAL SINHWAA = params.SINHWAA;                     // nrpy.reference_metric_SinhSymTP::SINHWAA
MAYBE_UNUSED const REAL SQRT1_2 = params.SQRT1_2;                     // nrpy.reference_metric::SQRT1_2
MAYBE_UNUSED const bool stop_relaxation = commondata.stop_relaxation; // nrpy.infrastructures.BHaH.nrpyelliptic.diagnostics::stop_relaxation
MAYBE_UNUSED const REAL t_0 = commondata.t_0;                         // nrpy.infrastructures.BHaH.MoLtimestepping.MoL_register_all::t_0
MAYBE_UNUSED const REAL t_final = commondata.t_final;                 // nrpy.infrastructures.BHaH.MoLtimestepping.MoL_register_all::t_final
MAYBE_UNUSED const REAL time = commondata.time;                       // nrpy.infrastructures.BHaH.MoLtimestepping.MoL_register_all::time
MAYBE_UNUSED const REAL xxmax0 = params.xxmax0;                       // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::xxmax0
MAYBE_UNUSED const REAL xxmax1 = params.xxmax1;                       // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::xxmax1
MAYBE_UNUSED const REAL xxmax2 = params.xxmax2;                       // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::xxmax2
MAYBE_UNUSED const REAL xxmin0 = params.xxmin0;                       // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::xxmin0
MAYBE_UNUSED const REAL xxmin1 = params.xxmin1;                       // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::xxmin1
MAYBE_UNUSED const REAL xxmin2 = params.xxmin2;                       // nrpy.infrastructures.BHaH.numerical_grids_and_timestep::xxmin2
MAYBE_UNUSED const REAL zPunc = commondata.zPunc;                     // nrpy.equations.nrpyelliptic.CommonParams::zPunc
