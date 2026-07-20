#ifndef INTERPOLATION_H
#define INTERPOLATION_H

enum {
  INTERP_SUCCESS,
  INTERP3D_GENERAL_NULL_PTRS,
  INTERP3D_GENERAL_INTERP_ORDER_GT_NXX123,
  INTERP3D_GENERAL_HORIZON_OUT_OF_BOUNDS
};

int interpolation_3d_general__uniform_src_grid(const int n_interp_ghosts, const REAL src_dxx0, const REAL src_dxx1, const REAL src_dxx2,
                                               const int src_Nxx_plus_2NGHOSTS0, const int src_Nxx_plus_2NGHOSTS1, const int src_Nxx_plus_2NGHOSTS2,
                                               const int NUM_INTERP_GFS, REAL *restrict src_x0x1x2[3],
                                               const REAL *restrict src_gf_ptrs[NUM_INTERP_GFS], const int num_dst_pts, const REAL dst_x0x1x2[][3],
                                               REAL *restrict dst_data[NUM_INTERP_GFS]);

#endif // INTERPOLATION_H
