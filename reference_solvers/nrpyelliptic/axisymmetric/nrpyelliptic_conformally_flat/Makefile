CC ?= gcc  # assigns the value CC to gcc only if environment variable CC is not already set

CFLAGS ?= -std=gnu99 -O2 -g -Wall
VALGRIND_CFLAGS = -std=gnu99 -O2 -g -Wall -Wno-unknown-pragmas
INCLUDEDIRS =
LDFLAGS ?= -lm
COMMON_HEADERS = BHaH_defines.h BHaH_function_prototypes.h NRPYELL_binary_format.h

# Check for OpenMP support
OPENMP_FLAG = -fopenmp
COMPILER_SUPPORTS_OPENMP := $(shell echo | $(CC) $(OPENMP_FLAG) -E - >/dev/null 2>&1 && echo YES || echo NO)

ifeq ($(COMPILER_SUPPORTS_OPENMP), YES)
    CFLAGS += $(OPENMP_FLAG)
    LDFLAGS += $(OPENMP_FLAG)
endif

OBJ_FILES = apply_bcs_inner_only.o apply_bcs_outerextrap_and_inner.o apply_bcs_outerradiation_and_inner.o auxevol_gfs_all_points.o auxevol_gfs_single_point.o bcstruct_set_up.o cfl_limited_timestep.o check_stop_conditions.o cmdline_input_and_parfile_parser.o commondata_struct_set_to_default.o compute_L2_norm_of_gridfunction.o compute_residual_all_points.o diagnostics.o diagnostics_nearest_1d_y_axis.o diagnostics_nearest_1d_z_axis.o diagnostics_nearest_2d_xy_plane.o diagnostics_nearest_2d_yz_plane.o ds_min_single_pt.o griddata_free.o initial_data.o initial_guess_single_point.o initialize_constant_auxevol.o interpolation_2d_general__uniform_src_grid.o main.o MoL_free_memory_non_y_n_gfs.o MoL_free_memory_y_n_gfs.o MoL_malloc_non_y_n_gfs.o MoL_malloc_y_n_gfs.o MoL_step_forward_in_time.o NRPYELL_Cart_to_xx__rfm__SinhSymTP.o NRPYELL_derivatives_all_points.o numerical_grid_params_Nxx_dxx_xx.o numerical_grids_and_timestep.o params_struct_set_to_default.o progress_indicator.o read_checkpoint.o read_NRPYELL_binary.o rfm_precompute_defines.o rfm_precompute_free.o rfm_precompute_malloc.o rhs_eval.o SinhSymTP/apply_bcs_outerradiation_and_inner__rfm__SinhSymTP.o SinhSymTP/bcstruct_set_up__rfm__SinhSymTP.o SinhSymTP/diagnostics_nearest_1d_y_axis__rfm__SinhSymTP.o SinhSymTP/diagnostics_nearest_1d_z_axis__rfm__SinhSymTP.o SinhSymTP/diagnostics_nearest_2d_xy_plane__rfm__SinhSymTP.o SinhSymTP/diagnostics_nearest_2d_yz_plane__rfm__SinhSymTP.o SinhSymTP/ds_min_single_pt__rfm__SinhSymTP.o SinhSymTP/numerical_grid_params_Nxx_dxx_xx__rfm__SinhSymTP.o SinhSymTP/rfm_precompute_defines__rfm__SinhSymTP.o SinhSymTP/rfm_precompute_free__rfm__SinhSymTP.o SinhSymTP/rfm_precompute_malloc__rfm__SinhSymTP.o SinhSymTP/xx_to_Cart__rfm__SinhSymTP.o variable_wavespeed_gfs_all_points.o write_checkpoint.o write_NRPYELL_binary.o xx_to_Cart.o

all: nrpyelliptic_conformally_flat

%.o: %.c $(COMMON_HEADERS)
	$(CC) $(CFLAGS) $(INCLUDEDIRS) -c $< -o $@

nrpyelliptic_conformally_flat: $(OBJ_FILES)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)


valgrind: clean
	$(MAKE) CFLAGS="$(VALGRIND_CFLAGS)" all
	valgrind --track-origins=yes --leak-check=full -s ./nrpyelliptic_conformally_flat

# Use $(RM) to be cross-platform compatible.
clean:
	$(RM) *.o */*.o *~ */*~ ./#* *.txt *.gp *.dat *.avi *.png nrpyelliptic_conformally_flat
