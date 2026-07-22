void apply_bcs_inner_only(const commondata_struct *restrict commondata, const params_struct *restrict params, const bc_struct *restrict bcstruct,
                          REAL *restrict gfs);
void apply_bcs_inner_only_specific_gfs(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                       const bc_struct *restrict bcstruct, REAL *restrict gfs, const int num_gfs, const int8_t *gf_parities,
                                       const int *gfs_to_sync);
void apply_bcs_outerextrap_and_inner(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                     const bc_struct *restrict bcstruct, REAL *restrict gfs);
void apply_bcs_outerradiation_and_inner(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                        const bc_struct *restrict bcstruct, REAL *restrict xx[3], const REAL custom_wavespeed[NUM_EVOL_GFS],
                                        const REAL custom_f_infinity[NUM_EVOL_GFS], REAL *restrict gfs, REAL *restrict rhs_gfs);
void apply_bcs_outerradiation_and_inner__rfm__SinhSymTP(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                                        const bc_struct *restrict bcstruct, REAL *restrict xx[3],
                                                        const REAL custom_wavespeed[NUM_EVOL_GFS], const REAL custom_f_infinity[NUM_EVOL_GFS],
                                                        REAL *restrict gfs, REAL *restrict rhs_gfs);
void auxevol_gfs_set_to_constant(commondata_struct *restrict commondata, params_struct *restrict params, REAL *restrict xx[3],
                                 MoL_gridfunctions_struct *restrict gridfuncs);
void auxevol_gfs_set_to_constant__rfm__SinhSymTP(commondata_struct *restrict commondata, params_struct *restrict params, REAL *restrict xx[3],
                                                 MoL_gridfunctions_struct *restrict gridfuncs);
void bcstruct_set_up(const commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3],
                     bc_struct *restrict bcstruct);
void bcstruct_set_up__rfm__SinhSymTP(const commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3],
                                     bc_struct *restrict bcstruct);
void Cart_to_xx_and_nearest_i0i1i2(const params_struct *restrict params, const REAL xCart[3], REAL xx[3], int Cart_to_i0i1i2[3]);
void Cart_to_xx_and_nearest_i0i1i2__rfm__SinhSymTP(const params_struct *restrict params, const REAL xCart[3], REAL xx[3], int Cart_to_i0i1i2[3]);
void cfl_limited_timestep(commondata_struct *restrict commondata, params_struct *restrict params, REAL *restrict xx[3]);
void cmdline_input_and_parfile_parser(commondata_struct *restrict commondata, int argc, const char *argv[]);
void commondata_struct_set_to_default(commondata_struct *restrict commondata);
void diagnostic_gfs_set(const commondata_struct *restrict commondata, const griddata_struct *restrict griddata,
                        REAL *restrict diagnostic_gfs[MAXNUMGRIDS]);
void diagnostics(commondata_struct *restrict commondata, griddata_struct *restrict griddata);
void diagnostics_nearest(commondata_struct *restrict commondata, griddata_struct *restrict griddata,
                         const REAL *restrict gridfuncs_diags[MAXNUMGRIDS]);
void diagnostics_nearest_1d_y_and_z_axes(commondata_struct *restrict commondata, const int grid, const params_struct *restrict params,
                                         const REAL *restrict xx[3], const int NUM_GFS_NEAREST, const int which_gfs[],
                                         const char **diagnostic_gf_names, const REAL *restrict gridfuncs_diags[]);
void diagnostics_nearest_1d_y_and_z_axes__rfm__SinhSymTP(commondata_struct *restrict commondata, const int grid, const params_struct *restrict params,
                                                         const REAL *restrict xx[3], const int NUM_GFS_NEAREST, const int which_gfs[],
                                                         const char **diagnostic_gf_names, const REAL *restrict gridfuncs_diags[]);
void diagnostics_nearest_2d_xy_and_yz_planes(commondata_struct *restrict commondata, const int grid, const params_struct *restrict params,
                                             const REAL *restrict xx[3], const int NUM_GFS_NEAREST, const int which_gfs[],
                                             const char **diagnostic_gf_names, const REAL *restrict gridfuncs_diags[]);
void diagnostics_nearest_2d_xy_and_yz_planes__rfm__SinhSymTP(commondata_struct *restrict commondata, const int grid,
                                                             const params_struct *restrict params, const REAL *restrict xx[3],
                                                             const int NUM_GFS_NEAREST, const int which_gfs[], const char **diagnostic_gf_names,
                                                             const REAL *restrict gridfuncs_diags[]);
void diagnostics_nearest_grid_center(commondata_struct *restrict commondata, const int grid, const params_struct *restrict params,
                                     const REAL *restrict xx[3], const int NUM_GFS_NEAREST, const int which_gfs[], const char **diagnostic_gf_names,
                                     const REAL *restrict gridfuncs_diags[]);
void diagnostics_nearest_grid_center__rfm__SinhSymTP(commondata_struct *restrict commondata, const int grid, const params_struct *restrict params,
                                                     const REAL *restrict xx[3], const int NUM_GFS_NEAREST, const int which_gfs[],
                                                     const char **diagnostic_gf_names, const REAL *restrict gridfuncs_diags[]);
void diagnostics_volume_integration(commondata_struct *restrict commondata, griddata_struct *restrict griddata,
                                    const REAL *restrict gridfuncs_diags[MAXNUMGRIDS]);
void ds_min_single_pt(const params_struct *restrict params, const REAL xx0, const REAL xx1, const REAL xx2, REAL *restrict ds_min);
void ds_min_single_pt__rfm__SinhSymTP(const params_struct *restrict params, const REAL xx0, const REAL xx1, const REAL xx2, REAL *restrict ds_min);
void griddata_free(commondata_struct *restrict commondata, griddata_struct *restrict griddata,
                   const bool free_non_y_n_gfs_and_core_griddata_pointers);
void initial_data(commondata_struct *restrict commondata, griddata_struct *restrict griddata);
void initial_guess_single_point(const REAL xx0, const REAL xx1, const REAL xx2, REAL *restrict uu_ID, REAL *restrict vv_ID);
REAL eval_uNN(const REAL xx0, const REAL xx1, const REAL S);
void validate_parametric_nn_compatibility(const commondata_struct *restrict commondata, const params_struct *restrict params);
void neural_net_guess_single_point(const commondata_struct *restrict commondata, const params_struct *restrict params, const REAL xx0, const REAL xx1,
                                   const REAL xx2, REAL *restrict uu_ID, REAL *restrict vv_ID);
int interpolation_3d_general__uniform_src_grid(const int n_interp_ghosts, const REAL src_dxx0, const REAL src_dxx1, const REAL src_dxx2,
                                               const int src_Nxx_plus_2NGHOSTS0, const int src_Nxx_plus_2NGHOSTS1, const int src_Nxx_plus_2NGHOSTS2,
                                               const int NUM_INTERP_GFS, REAL *restrict src_x0x1x2[3],
                                               const REAL *restrict src_gf_ptrs[NUM_INTERP_GFS], const int num_dst_pts, const REAL dst_x0x1x2[][3],
                                               REAL *restrict dst_data[NUM_INTERP_GFS]);
int main(int argc, const char *argv[]);
void MoL_free_intermediate_stage_gfs(MoL_gridfunctions_struct *restrict gridfuncs);
void MoL_malloc_intermediate_stage_gfs(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                       MoL_gridfunctions_struct *restrict gridfuncs);
void MoL_step_forward_in_time(commondata_struct *restrict commondata, griddata_struct *restrict griddata);
void numerical_grid_params_Nxx_dxx_xx(const commondata_struct *restrict commondata, params_struct *restrict params, REAL *restrict xx[3],
                                      const int Nx[3], const bool apply_convergence_factor_and_set_xxminmax_defaults);
void numerical_grid_params_Nxx_dxx_xx__rfm__SinhSymTP(const commondata_struct *restrict commondata, params_struct *restrict params,
                                                      REAL *restrict xx[3], const int Nx[3],
                                                      const bool apply_convergence_factor_and_set_xxminmax_defaults);
void numerical_grids_and_timestep(commondata_struct *restrict commondata, griddata_struct *restrict griddata, bool calling_for_first_time);
void params_struct_set_to_default(commondata_struct *restrict commondata, griddata_struct *restrict griddata);
void progress_indicator(commondata_struct *restrict commondata, const griddata_struct *restrict griddata);
int read_checkpoint(commondata_struct *restrict commondata, griddata_struct *restrict griddata);
void residual_H_compute_all_points(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                   const rfm_struct *restrict rfmstruct, const REAL *restrict auxevol_gfs, const REAL *restrict in_gfs,
                                   REAL *restrict dest_gf_address);
void residual_H_compute_all_points__rfm__SinhSymTP(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                                   const rfm_struct *restrict rfmstruct, const REAL *restrict auxevol_gfs,
                                                   const REAL *restrict in_gfs, REAL *restrict dest_gf_address);
void rfm_precompute_defines(const commondata_struct *restrict commondata, const params_struct *restrict params, rfm_struct *restrict rfmstruct,
                            REAL *restrict xx[3]);
void rfm_precompute_defines__rfm__SinhSymTP(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                            rfm_struct *restrict rfmstruct, REAL *restrict xx[3]);
void rfm_precompute_free(const commondata_struct *restrict commondata, const params_struct *restrict params, rfm_struct *restrict rfmstruct);
void rfm_precompute_free__rfm__SinhSymTP(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                         rfm_struct *restrict rfmstruct);
void rfm_precompute_malloc(const commondata_struct *restrict commondata, const params_struct *restrict params, rfm_struct *restrict rfmstruct);
void rfm_precompute_malloc__rfm__SinhSymTP(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                           rfm_struct *restrict rfmstruct);
void rhs_eval(const commondata_struct *restrict commondata, const params_struct *restrict params, const rfm_struct *restrict rfmstruct,
              const REAL *restrict auxevol_gfs, const REAL *restrict in_gfs, REAL *restrict rhs_gfs);
void rhs_eval__rfm__SinhSymTP(const commondata_struct *restrict commondata, const params_struct *restrict params,
                              const rfm_struct *restrict rfmstruct, const REAL *restrict auxevol_gfs, const REAL *restrict in_gfs,
                              REAL *restrict rhs_gfs);
void sqrt_detgammahat_d3xx_volume_element(const params_struct *restrict params, const REAL xx0, const REAL xx1, const REAL xx2, REAL *restrict dV);
void sqrt_detgammahat_d3xx_volume_element__rfm__SinhSymTP(const params_struct *restrict params, const REAL xx0, const REAL xx1, const REAL xx2,
                                                          REAL *restrict dV);
void stop_conditions_check(commondata_struct *restrict commondata);
void write_checkpoint(const commondata_struct *restrict commondata, griddata_struct *restrict griddata);
void write_NRPYELL_binary(const commondata_struct *restrict commondata, griddata_struct *restrict griddata);
void xx_to_Cart(const params_struct *restrict params, const REAL xx[3], REAL xCart[3]);
void xx_to_Cart__rfm__SinhSymTP(const params_struct *restrict params, const REAL xx[3], REAL xCart[3]);
