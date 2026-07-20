CC ?= gcc  # assigns the value CC to gcc only if environment variable CC is not already set

CFLAGS ?= -std=gnu99 -O2 -g -Wall -I.
CXXFLAGS ?= -I. -O2 -g -Wall -Wno-unknown-pragmas
VALGRIND_CFLAGS = -I. -std=gnu99 -O2 -g -Wall -Wno-unknown-pragmas
INCLUDEDIRS =
LDFLAGS ?= -lm
# Check for OpenMP support
OPENMP_FLAG = -fopenmp
COMPILER_SUPPORTS_OPENMP := $(shell echo | $(CC) $(OPENMP_FLAG) -E - >/dev/null 2>&1 && echo YES || echo NO)

ifeq ($(COMPILER_SUPPORTS_OPENMP), YES)
    CFLAGS += $(OPENMP_FLAG)
    LDFLAGS += $(OPENMP_FLAG)
endif

OBJ_FILES = apply_bcs_inner_only.o apply_bcs_inner_only_specific_gfs.o apply_bcs_outerextrap_and_inner.o apply_bcs_outerradiation_and_inner.o auxevol_gfs_set_to_constant.o bcstruct_set_up.o Cart_to_xx_and_nearest_i0i1i2.o cfl_limited_timestep.o cmdline_input_and_parfile_parser.o commondata_struct_set_to_default.o diagnostics/diagnostic_gfs_set.o diagnostics/diagnostics.o diagnostics/diagnostics_nearest.o diagnostics/diagnostics_nearest_1d_y_and_z_axes.o diagnostics/diagnostics_nearest_2d_xy_and_yz_planes.o diagnostics/diagnostics_nearest_grid_center.o diagnostics/diagnostics_volume_integration.o diagnostics/residual_H_compute_all_points.o diagnostics/SinhSymTP/diagnostics_nearest_1d_y_and_z_axes__rfm__SinhSymTP.o diagnostics/SinhSymTP/diagnostics_nearest_2d_xy_and_yz_planes__rfm__SinhSymTP.o diagnostics/SinhSymTP/diagnostics_nearest_grid_center__rfm__SinhSymTP.o diagnostics/SinhSymTP/residual_H_compute_all_points__rfm__SinhSymTP.o diagnostics/SinhSymTP/sqrt_detgammahat_d3xx_volume_element__rfm__SinhSymTP.o diagnostics/sqrt_detgammahat_d3xx_volume_element.o ds_min_single_pt.o griddata_free.o initial_data.o initial_guess_single_point.o interpolation_3d_general__uniform_src_grid.o main.o MoL/MoL_free_intermediate_stage_gfs.o MoL/MoL_malloc_intermediate_stage_gfs.o MoL/MoL_step_forward_in_time.o numerical_grid_params_Nxx_dxx_xx.o numerical_grids_and_timestep.o params_struct_set_to_default.o progress_indicator.o read_checkpoint.o rfm_precompute_defines.o rfm_precompute_free.o rfm_precompute_malloc.o rhs_eval.o SinhSymTP/apply_bcs_outerradiation_and_inner__rfm__SinhSymTP.o SinhSymTP/auxevol_gfs_set_to_constant__rfm__SinhSymTP.o SinhSymTP/bcstruct_set_up__rfm__SinhSymTP.o SinhSymTP/Cart_to_xx_and_nearest_i0i1i2__rfm__SinhSymTP.o SinhSymTP/ds_min_single_pt__rfm__SinhSymTP.o SinhSymTP/numerical_grid_params_Nxx_dxx_xx__rfm__SinhSymTP.o SinhSymTP/rfm_precompute_defines__rfm__SinhSymTP.o SinhSymTP/rfm_precompute_free__rfm__SinhSymTP.o SinhSymTP/rfm_precompute_malloc__rfm__SinhSymTP.o SinhSymTP/rhs_eval__rfm__SinhSymTP.o SinhSymTP/xx_to_Cart__rfm__SinhSymTP.o stop_conditions_check.o write_checkpoint.o write_NRPYELL_binary.o xx_to_Cart.o

all: nrpyelliptic_conformally_flat

%.o: %.c $(COMMON_HEADERS)
	$(CC) $(CFLAGS) $(INCLUDEDIRS) -c $< -o $@

nrpyelliptic_conformally_flat: $(OBJ_FILES)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

valgrind: clean
	$(MAKE) CFLAGS="$(VALGRIND_CFLAGS)" all
	valgrind --track-origins=yes --leak-check=full --show-leak-kinds=all -s ./nrpyelliptic_conformally_flat

# Use $(RM) to be cross-platform compatible.
clean:
	$(RM) *.o */*.o */*/*.o *~ */*~ */lib*.a ./#* *.txt *.gp *.dat *.avi *.png nrpyelliptic_conformally_flat
