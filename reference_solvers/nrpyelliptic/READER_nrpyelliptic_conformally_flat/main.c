#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
#include "NRPYELL_binary_format.h"
#include "interpolation.h"
#include <limits.h>

enum { NINTERP_GHOSTS = 4 }; // Four neighbors on either side: a fixed 9-point stencil.

typedef struct {
  const char *binary_path;
  const char *coords_path;
  const char *output_path;
} cli_options;

static void print_usage(FILE *stream, const char *program) {
  fprintf(stream,
          "Usage: %s --binary FILE --coords FILE --output FILE\n"
          "\n"
          "Interpolate the NRPyElliptic uu field at Cartesian coordinates.\n"
          "The coordinate file contains whitespace-separated x y z triples.\n"
          "The output is a text table with columns: x y z uu.\n",
          program);
}

static int parse_cli(const int argc, const char *argv[], cli_options *options) {
  memset(options, 0, sizeof(*options));
  if (argc == 2 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
    print_usage(stdout, argv[0]);
    return 1;
  }

  for (int i = 1; i < argc; i++) {
    const char **destination = NULL;
    if (strcmp(argv[i], "--binary") == 0)
      destination = &options->binary_path;
    else if (strcmp(argv[i], "--coords") == 0)
      destination = &options->coords_path;
    else if (strcmp(argv[i], "--output") == 0)
      destination = &options->output_path;
    else {
      fprintf(stderr, "Unknown argument: %s\n", argv[i]);
      return -1;
    }

    if (*destination != NULL) {
      fprintf(stderr, "Argument specified more than once: %s\n", argv[i]);
      return -1;
    }
    if (++i >= argc) {
      fprintf(stderr, "Missing value after %s\n", argv[i - 1]);
      return -1;
    }
    *destination = argv[i];
  }

  if (options->binary_path == NULL || options->coords_path == NULL || options->output_path == NULL) {
    fprintf(stderr, "--binary, --coords, and --output are all required.\n");
    return -1;
  }
  return 0;
}

static void free_reader_arrays(REAL *x_arr, REAL *y_arr, REAL *z_arr, REAL *NRPYELL_xx0, REAL *NRPYELL_xx1, REAL *NRPYELL_xx2,
                               REAL *NRPYELL_fields, REAL (*dst_x0x1x2)[3], REAL *dst_vals) {
  free(x_arr);
  free(y_arr);
  free(z_arr);
  free(NRPYELL_xx0);
  free(NRPYELL_xx1);
  free(NRPYELL_xx2);
  free(NRPYELL_fields);
  free(dst_x0x1x2);
  free(dst_vals);
}

int main(int argc, const char *argv[]) {
  cli_options options;
  const int parse_status = parse_cli(argc, argv, &options);
  if (parse_status > 0)
    return 0;
  if (parse_status < 0) {
    print_usage(stderr, argv[0]);
    return 1;
  }

  FILE *fp = fopen(options.coords_path, "r");
  if (fp == NULL) {
    fprintf(stderr, "Could not open coordinate file '%s': %s\n", options.coords_path, strerror(errno));
    return 1;
  }

  size_t cap = 1024;
  size_t n_pts = 0;
  REAL *x_arr = malloc(cap * sizeof(REAL));
  REAL *y_arr = malloc(cap * sizeof(REAL));
  REAL *z_arr = malloc(cap * sizeof(REAL));
  if (!x_arr || !y_arr || !z_arr) {
    perror("Memory allocation failed");
    fclose(fp);
    free(x_arr);
    free(y_arr);
    free(z_arr);
    return 1;
  }

  while (1) {
    if (n_pts == cap) {
      if (cap > SIZE_MAX / (2 * sizeof(REAL))) {
        fprintf(stderr, "Coordinate file is too large.\n");
        fclose(fp);
        free(x_arr);
        free(y_arr);
        free(z_arr);
        return 1;
      }
      cap *= 2;
      REAL *new_x = realloc(x_arr, cap * sizeof(REAL));
      if (new_x == NULL) {
        perror("Re-allocation failed");
        fclose(fp);
        free(x_arr);
        free(y_arr);
        free(z_arr);
        return 1;
      }
      x_arr = new_x;
      REAL *new_y = realloc(y_arr, cap * sizeof(REAL));
      if (new_y == NULL) {
        perror("Re-allocation failed");
        fclose(fp);
        free(x_arr);
        free(y_arr);
        free(z_arr);
        return 1;
      }
      y_arr = new_y;
      REAL *new_z = realloc(z_arr, cap * sizeof(REAL));
      if (new_z == NULL) {
        perror("Re-allocation failed");
        fclose(fp);
        free(x_arr);
        free(y_arr);
        free(z_arr);
        return 1;
      }
      z_arr = new_z;
    }
    REAL xv, yv, zv;
    int got = fscanf(fp, "%lf %lf %lf", &xv, &yv, &zv);
    if (got == EOF)
      break;
    if (got != 3) {
      fprintf(stderr, "Malformed coordinate triple %zu in %s\n", n_pts + 1, options.coords_path);
      fclose(fp);
      free(x_arr);
      free(y_arr);
      free(z_arr);
      return 1;
    }
    x_arr[n_pts] = xv;
    y_arr[n_pts] = yv;
    z_arr[n_pts] = zv;
    n_pts++;
  }
  fclose(fp);

  if (n_pts == 0) {
    fprintf(stderr, "Coordinate file '%s' contains no points.\n", options.coords_path);
    free(x_arr);
    free(y_arr);
    free(z_arr);
    return 1;
  }
  if (n_pts > INT_MAX) {
    fprintf(stderr, "Coordinate file contains too many points for the interpolation interface.\n");
    free(x_arr);
    free(y_arr);
    free(z_arr);
    return 1;
  }
  printf("Read %zu points from %s; writing results to %s\n", n_pts, options.coords_path, options.output_path);

  int NRPYELL_Nxx_plus_2NGHOSTS0, NRPYELL_Nxx_plus_2NGHOSTS1, NRPYELL_Nxx_plus_2NGHOSTS2;
  int NRPYELL_NGHOSTS, NRPYELL_TOTAL_PTS, NRPYELL_NUM_FIELDS;
  REAL NRPYELL_AMAX, NRPYELL_bScale, NRPYELL_SINHWAA;
  REAL NRPYELL_dxx0, NRPYELL_dxx1, NRPYELL_dxx2;
  REAL *NRPYELL_xx0 = NULL, *NRPYELL_xx1 = NULL, *NRPYELL_xx2 = NULL;
  REAL *NRPYELL_fields = NULL;

  read_NRPYELL_binary(options.binary_path, &NRPYELL_Nxx_plus_2NGHOSTS0, &NRPYELL_Nxx_plus_2NGHOSTS1, &NRPYELL_Nxx_plus_2NGHOSTS2, &NRPYELL_NGHOSTS,
                      &NRPYELL_TOTAL_PTS, &NRPYELL_NUM_FIELDS, &NRPYELL_AMAX, &NRPYELL_bScale, &NRPYELL_SINHWAA, &NRPYELL_dxx0,
                      &NRPYELL_dxx1, &NRPYELL_dxx2, &NRPYELL_xx0, &NRPYELL_xx1, &NRPYELL_xx2, &NRPYELL_fields);

  if (NRPYELL_NGHOSTS < NINTERP_GHOSTS) {
    fprintf(stderr, "Binary provides %d ghost cells, but the fixed 9-point stencil requires at least %d.\n", NRPYELL_NGHOSTS,
            NINTERP_GHOSTS);
    free_reader_arrays(x_arr, y_arr, z_arr, NRPYELL_xx0, NRPYELL_xx1, NRPYELL_xx2, NRPYELL_fields, NULL, NULL);
    return 1;
  }

  if (NRPYELL_NGHOSTS != NGHOSTS) {
    fprintf(stderr, "NRPYELL: Reader was compiled with NGHOSTS=%d but binary has NGHOSTS=%d.\n", NGHOSTS, NRPYELL_NGHOSTS);
    free_reader_arrays(x_arr, y_arr, z_arr, NRPYELL_xx0, NRPYELL_xx1, NRPYELL_xx2, NRPYELL_fields, NULL, NULL);
    return 1;
  }

  REAL(*dst_x0x1x2)[3] = malloc(n_pts * sizeof *dst_x0x1x2);
  REAL *dst_vals = malloc(n_pts * sizeof(REAL));
  if (!dst_x0x1x2 || !dst_vals) {
    perror("Allocation failed for interpolation arrays");
    free_reader_arrays(x_arr, y_arr, z_arr, NRPYELL_xx0, NRPYELL_xx1, NRPYELL_xx2, NRPYELL_fields, dst_x0x1x2, dst_vals);
    return 1;
  }

  params_struct params;
  memset(&params, 0, sizeof(params));
  params.AMAX = NRPYELL_AMAX;
  params.bScale = NRPYELL_bScale;
  params.SINHWAA = NRPYELL_SINHWAA;
  params.dxx0 = NRPYELL_dxx0;
  params.dxx1 = NRPYELL_dxx1;
  params.dxx2 = NRPYELL_dxx2;
  params.xxmin0 = NRPYELL_xx0[NRPYELL_NGHOSTS] - 0.5 * NRPYELL_dxx0;
  params.xxmin1 = NRPYELL_xx1[NRPYELL_NGHOSTS] - 0.5 * NRPYELL_dxx1;
  params.xxmin2 = NRPYELL_xx2[NRPYELL_NGHOSTS] - 0.5 * NRPYELL_dxx2;
  params.Nxx_plus_2NGHOSTS0 = NRPYELL_Nxx_plus_2NGHOSTS0;
  params.Nxx_plus_2NGHOSTS1 = NRPYELL_Nxx_plus_2NGHOSTS1;
  params.Nxx_plus_2NGHOSTS2 = NRPYELL_Nxx_plus_2NGHOSTS2;
  params.CoordSystem_hash = SINHSYMTP;

  for (size_t i = 0; i < n_pts; i++) {
    REAL xCart[3] = {x_arr[i], y_arr[i], z_arr[i]};
    int nearest_i0i1i2[3];
    Cart_to_xx_and_nearest_i0i1i2(&params, xCart, dst_x0x1x2[i], nearest_i0i1i2);
    // The analytic inverse map has an indeterminate polar angle on the z axis.
    if (!isfinite(dst_x0x1x2[i][1]))
      dst_x0x1x2[i][1] = (xCart[2] >= 0.0) ? 0.0 : M_PI;
  }

  REAL *src_x0x1x2[3] = {NRPYELL_xx0, NRPYELL_xx1, NRPYELL_xx2};
  const int NUM_INTERP_GFS = 1;
  const REAL *src_gf_ptrs[1] = {
      &NRPYELL_fields[(size_t)NRPYELL_SOL_UUGF * (size_t)NRPYELL_TOTAL_PTS],
  };
  REAL *dst_data[1] = {dst_vals};

  int err = interpolation_3d_general__uniform_src_grid(NINTERP_GHOSTS, NRPYELL_dxx0, NRPYELL_dxx1, NRPYELL_dxx2, NRPYELL_Nxx_plus_2NGHOSTS0,
                                                       NRPYELL_Nxx_plus_2NGHOSTS1, NRPYELL_Nxx_plus_2NGHOSTS2, NUM_INTERP_GFS, src_x0x1x2,
                                                       src_gf_ptrs, (int)n_pts, dst_x0x1x2, dst_data);
  if (err != INTERP_SUCCESS) {
    fprintf(stderr, "WARNING: Interpolation error code %d -- aborting\n", err);
    free_reader_arrays(x_arr, y_arr, z_arr, NRPYELL_xx0, NRPYELL_xx1, NRPYELL_xx2, NRPYELL_fields, dst_x0x1x2, dst_vals);
    return 1;
  }

  FILE *outf = fopen(options.output_path, "w");
  if (!outf) {
    fprintf(stderr, "Could not open output file '%s': %s\n", options.output_path, strerror(errno));
    free_reader_arrays(x_arr, y_arr, z_arr, NRPYELL_xx0, NRPYELL_xx1, NRPYELL_xx2, NRPYELL_fields, dst_x0x1x2, dst_vals);
    return 1;
  }
  fprintf(outf, "# x y z uu\n");
  for (size_t i = 0; i < n_pts; i++) {
    fprintf(outf, "%.17e %.17e %.17e %.17e\n", x_arr[i], y_arr[i], z_arr[i], dst_vals[i]);
  }
  fclose(outf);

  free_reader_arrays(x_arr, y_arr, z_arr, NRPYELL_xx0, NRPYELL_xx1, NRPYELL_xx2, NRPYELL_fields, dst_x0x1x2, dst_vals);
  return 0;
}
