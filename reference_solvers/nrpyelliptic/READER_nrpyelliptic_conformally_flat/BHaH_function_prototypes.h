int interpolation_3d_general__uniform_src_grid(const int n_interp_ghosts, const REAL src_dxx0, const REAL src_dxx1, const REAL src_dxx2,
                                               const int src_Nxx_plus_2NGHOSTS0, const int src_Nxx_plus_2NGHOSTS1, const int src_Nxx_plus_2NGHOSTS2,
                                               const int NUM_INTERP_GFS, REAL *restrict src_x0x1x2[3],
                                               const REAL *restrict src_gf_ptrs[NUM_INTERP_GFS], const int num_dst_pts, const REAL dst_x0x1x2[][3],
                                               REAL *restrict dst_data[NUM_INTERP_GFS]);
void Cart_to_xx_and_nearest_i0i1i2(const params_struct *restrict params, const REAL xCart[3], REAL xx[3], int Cart_to_i0i1i2[3]);
void Cart_to_xx_and_nearest_i0i1i2__rfm__SinhSymTP(const params_struct *restrict params, const REAL xCart[3], REAL xx[3], int Cart_to_i0i1i2[3]);
void xx_to_Cart__rfm__SinhSymTP(const params_struct *restrict params, const REAL xx[3], REAL xCart[3]);
void read_NRPYELL_binary(const char *restrict binary_path,
                         int *restrict NRPYELL_Nxx_plus_2NGHOSTS0, int *restrict NRPYELL_Nxx_plus_2NGHOSTS1, int *restrict NRPYELL_Nxx_plus_2NGHOSTS2,
                         int *restrict NRPYELL_NGHOSTS, int *restrict NRPYELL_TOTAL_PTS, int *restrict NRPYELL_NUM_FIELDS,
                         REAL *restrict NRPYELL_AMAX, REAL *restrict NRPYELL_bScale, REAL *restrict NRPYELL_SINHWAA,
                         REAL *restrict NRPYELL_dxx0, REAL *restrict NRPYELL_dxx1, REAL *restrict NRPYELL_dxx2,
                         REAL *restrict *NRPYELL_xx0, REAL *restrict *NRPYELL_xx1, REAL *restrict *NRPYELL_xx2, REAL *restrict *NRPYELL_fields);
