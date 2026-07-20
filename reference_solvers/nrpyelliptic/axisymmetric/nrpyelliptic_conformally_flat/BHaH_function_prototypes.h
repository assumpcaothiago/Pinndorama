void apply_bcs_inner_only(const commondata_struct *restrict commondata, const params_struct *restrict params, const bc_struct *restrict bcstruct,
                          REAL *restrict gfs);
void apply_bcs_outerextrap_and_inner(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                     const bc_struct *restrict bcstruct, REAL *restrict gfs);
void apply_bcs_outerradiation_and_inner(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                        const bc_struct *restrict bcstruct, REAL *restrict xx[3], const REAL custom_wavespeed[NUM_EVOL_GFS],
                                        const REAL custom_f_infinity[NUM_EVOL_GFS], REAL *restrict gfs, REAL *restrict rhs_gfs);
void apply_bcs_outerradiation_and_inner__rfm__SinhSymTP(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                                        const bc_struct *restrict bcstruct, REAL *restrict xx[3],
                                                        const REAL custom_wavespeed[NUM_EVOL_GFS], const REAL custom_f_infinity[NUM_EVOL_GFS],
                                                        REAL *restrict gfs, REAL *restrict rhs_gfs);
void auxevol_gfs_all_points(commondata_struct *restrict commondata, griddata_struct *restrict griddata);
void auxevol_gfs_single_point(const commondata_struct *restrict commondata, const params_struct *restrict params, const REAL xx0, const REAL xx1,
                              const REAL xx2, REAL *restrict psi_background, REAL *restrict ADD_times_AUU);
void bcstruct_set_up(const commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3],
                     bc_struct *restrict bcstruct);
void bcstruct_set_up__rfm__SinhSymTP(const commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3],
                                     bc_struct *restrict bcstruct);
void cfl_limited_timestep(commondata_struct *restrict commondata, params_struct *restrict params, REAL *restrict xx[3]);
void check_stop_conditions(commondata_struct *restrict commondata, griddata_struct *restrict griddata);
void cmdline_input_and_parfile_parser(commondata_struct *restrict commondata, int argc, const char *argv[]);
void commondata_struct_set_to_default(commondata_struct *restrict commondata);
REAL compute_L2_norm_of_gridfunction(commondata_struct *restrict commondata, griddata_struct *restrict griddata, const REAL integration_radius,
                                     const int gf_index, const REAL *restrict in_gf);
void compute_residual_all_points(const commondata_struct *restrict commondata, const params_struct *restrict params,
                                 const rfm_struct *restrict rfmstruct, const REAL *restrict auxevol_gfs, const REAL *restrict in_gfs,
                                 REAL *restrict aux_gfs);
void NRPYELL_derivatives_all_points(const commondata_struct *restrict commondata, const params_struct *restrict params, const REAL *restrict in_gfs,
                                    REAL *restrict NRPYELL_fields);
void diagnostics(commondata_struct *restrict commondata, griddata_struct *restrict griddata);
void diagnostics_nearest_1d_y_axis(commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3],
                                   MoL_gridfunctions_struct *restrict gridfuncs);
void diagnostics_nearest_1d_y_axis__rfm__SinhSymTP(commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3],
                                                   MoL_gridfunctions_struct *restrict gridfuncs);
void diagnostics_nearest_1d_z_axis(commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3],
                                   MoL_gridfunctions_struct *restrict gridfuncs);
void diagnostics_nearest_1d_z_axis__rfm__SinhSymTP(commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3],
                                                   MoL_gridfunctions_struct *restrict gridfuncs);
void diagnostics_nearest_2d_xy_plane(commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3],
                                     MoL_gridfunctions_struct *restrict gridfuncs);
void diagnostics_nearest_2d_xy_plane__rfm__SinhSymTP(commondata_struct *restrict commondata, const params_struct *restrict params,
                                                     REAL *restrict xx[3], MoL_gridfunctions_struct *restrict gridfuncs);
void diagnostics_nearest_2d_yz_plane(commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3],
                                     MoL_gridfunctions_struct *restrict gridfuncs);
void diagnostics_nearest_2d_yz_plane__rfm__SinhSymTP(commondata_struct *restrict commondata, const params_struct *restrict params,
                                                     REAL *restrict xx[3], MoL_gridfunctions_struct *restrict gridfuncs);
void ds_min_single_pt(const commondata_struct *restrict commondata, const params_struct *restrict params, const REAL xx0, const REAL xx1,
                      const REAL xx2, REAL *restrict ds_min);
void ds_min_single_pt__rfm__SinhSymTP(const commondata_struct *restrict commondata, const params_struct *restrict params, const REAL xx0,
                                      const REAL xx1, const REAL xx2, REAL *restrict ds_min);
void griddata_free(const commondata_struct *restrict commondata, griddata_struct *restrict griddata,
                   const bool free_non_y_n_gfs_and_core_griddata_pointers);
void initial_data(commondata_struct *restrict commondata, griddata_struct *restrict griddata);
void initial_guess_single_point(const commondata_struct *restrict commondata, const params_struct *restrict params, const REAL xx0, const REAL xx1,
                                const REAL xx2, REAL *restrict uu_ID, REAL *restrict vv_ID);
void initialize_constant_auxevol(commondata_struct *restrict commondata, griddata_struct *restrict griddata);
int interpolation_2d_general__uniform_src_grid(const int N_interp_GHOSTS, const REAL src_dxx0, const REAL src_dxx1, const int src_Nxx_plus_2NGHOSTS0,
                                               const int src_Nxx_plus_2NGHOSTS1, const int NUM_INTERP_GFS, REAL *restrict src_x0x1x2[3],
                                               const REAL *restrict src_gf_ptrs[], const int num_dst_pts, const REAL dst_x0x1[][2],
                                               REAL *restrict dst_data[]);
int main(int argc, const char *argv[]);
void MoL_free_memory_non_y_n_gfs(MoL_gridfunctions_struct *restrict gridfuncs);
void MoL_free_memory_y_n_gfs(MoL_gridfunctions_struct *restrict gridfuncs);
void MoL_malloc_non_y_n_gfs(const commondata_struct *restrict commondata, const params_struct *restrict params,
                            MoL_gridfunctions_struct *restrict gridfuncs);
void MoL_malloc_y_n_gfs(const commondata_struct *restrict commondata, const params_struct *restrict params,
                        MoL_gridfunctions_struct *restrict gridfuncs);
void MoL_step_forward_in_time(commondata_struct *restrict commondata, griddata_struct *restrict griddata);
void NRPYELL_Cart_to_xx__rfm__SinhSymTP(const REAL AMAX, const REAL bScale, const REAL SINHWAA, const REAL xCart[3], REAL xx[3]);
void numerical_grid_params_Nxx_dxx_xx(const commondata_struct *restrict commondata, params_struct *restrict params, REAL *restrict xx[3],
                                      const int Nx[3], const bool set_xxmin_xxmax_to_defaults);
void numerical_grid_params_Nxx_dxx_xx__rfm__SinhSymTP(const commondata_struct *restrict commondata, params_struct *restrict params,
                                                      REAL *restrict xx[3], const int Nx[3], const bool set_xxmin_xxmax_to_defaults);
void numerical_grids_and_timestep(commondata_struct *restrict commondata, griddata_struct *restrict griddata, bool calling_for_first_time);
void params_struct_set_to_default(commondata_struct *restrict commondata, griddata_struct *restrict griddata);
void progress_indicator(commondata_struct *restrict commondata, const griddata_struct *restrict griddata);
int read_checkpoint(commondata_struct *restrict commondata, griddata_struct *restrict griddata);
void read_NRPYELL_binary(int *restrict NRPYELL_Nxx_plus_2NGHOSTS0, int *restrict NRPYELL_Nxx_plus_2NGHOSTS1, int *restrict NRPYELL_Nxx_plus_2NGHOSTS2,
                         int *restrict NRPYELL_NGHOSTS, int *restrict NRPYELL_TOTAL_PTS, int *restrict NRPYELL_NUM_FIELDS,
                         REAL *restrict NRPYELL_AMAX, REAL *restrict NRPYELL_bScale, REAL *restrict NRPYELL_SINHWAA,
                         REAL *restrict NRPYELL_dxx0, REAL *restrict NRPYELL_dxx1, REAL *restrict NRPYELL_dxx2,
                         REAL *restrict *NRPYELL_xx0, REAL *restrict *NRPYELL_xx1, REAL *restrict *NRPYELL_xx2, REAL *restrict *NRPYELL_fields);
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
void variable_wavespeed_gfs_all_points(commondata_struct *restrict commondata, griddata_struct *restrict griddata);
void write_checkpoint(const commondata_struct *restrict commondata, griddata_struct *restrict griddata);
void write_NRPYELL_binary(const commondata_struct *restrict commondata, griddata_struct *restrict griddata);
void xx_to_Cart(const commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3], const int i0, const int i1,
                const int i2, REAL xCart[3]);
void xx_to_Cart__rfm__SinhSymTP(const commondata_struct *restrict commondata, const params_struct *restrict params, REAL *restrict xx[3],
                                const int i0, const int i1, const int i2, REAL xCart[3]);
