#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
#include "NRPYELL_binary_format.h"

static void checked_fwrite(const void *restrict data, const size_t size, const size_t count, FILE *restrict fp, const char *restrict label) {
  if (fwrite(data, size, count, fp) != count) {
    fprintf(stderr, "NRPYELL: Error writing %s\n", label);
    fclose(fp);
    exit(1);
  }
}

/**
 * Writes enriched 3D NRPyElliptic data to NRPYELL_solution.bin.
 *
 * Binary v1 layout:
 * 1. Eight-byte magic string NRPYELL3.
 * 2. Seven integers: version, N0, N1, N2, NGHOSTS, total points, field count.
 * 3. Six REALs: AMAX, bScale, SINHWAA, dxx0, dxx1, dxx2.
 * 4. Three coordinate arrays xx0, xx1, xx2.
 * 5. NRPYELL_SOL_NUM_GFS field arrays, each of size total points.
 */
void write_NRPYELL_binary(const commondata_struct *restrict commondata, griddata_struct *restrict griddata) {

  params_struct *restrict params = &griddata[0].params;
  const rfm_struct *restrict rfmstruct = griddata[0].rfmstruct;
  REAL *restrict xx[3];
  for (int ww = 0; ww < 3; ww++)
    xx[ww] = griddata[0].xx[ww];

#include "set_CodeParameters.h"

  REAL *restrict y_n_gfs = griddata[0].gridfuncs.y_n_gfs;
  REAL *restrict auxevol_gfs = griddata[0].gridfuncs.auxevol_gfs;

  const int NRPYELL_VERSION = NRPYELL_BINARY_VERSION;
  const int NRPYELL_Nxx_plus_2NGHOSTS0 = Nxx_plus_2NGHOSTS0;
  const int NRPYELL_Nxx_plus_2NGHOSTS1 = Nxx_plus_2NGHOSTS1;
  const int NRPYELL_Nxx_plus_2NGHOSTS2 = Nxx_plus_2NGHOSTS2;
  const int NRPYELL_NGHOSTS = NGHOSTS;
  const int NRPYELL_TOTAL_PTS = Nxx_plus_2NGHOSTS0 * Nxx_plus_2NGHOSTS1 * Nxx_plus_2NGHOSTS2;
  const int NRPYELL_NUM_FIELDS = NRPYELL_SOL_NUM_GFS;

  const REAL NRPYELL_AMAX = params->AMAX;
  const REAL NRPYELL_bScale = params->bScale;
  const REAL NRPYELL_SINHWAA = params->SINHWAA;
  const REAL NRPYELL_dxx0 = params->dxx0;
  const REAL NRPYELL_dxx1 = params->dxx1;
  const REAL NRPYELL_dxx2 = params->dxx2;

  const size_t total_pts = (size_t)NRPYELL_TOTAL_PTS;
  const size_t total_fields = (size_t)NRPYELL_NUM_FIELDS * total_pts;
  REAL *restrict NRPYELL_fields = (REAL *restrict)malloc(sizeof(REAL) * total_fields);
  if (NRPYELL_fields == NULL) {
    perror("NRPYELL: Memory allocation failed for enriched 3D solution fields");
    exit(1);
  }

#pragma omp parallel for
  for (size_t i = 0; i < total_fields; i++)
    NRPYELL_fields[i] = NAN;

#pragma omp parallel for collapse(2)
  for (int i2 = 0; i2 < Nxx_plus_2NGHOSTS2; i2++) {
    for (int i1 = 0; i1 < Nxx_plus_2NGHOSTS1; i1++) {
      for (int i0 = 0; i0 < Nxx_plus_2NGHOSTS0; i0++) {
        const int idx3 = IDX3(i0, i1, i2);
        NRPYELL_fields[IDX4pt(NRPYELL_SOL_UUGF, idx3)] = y_n_gfs[IDX4(UUGF, i0, i1, i2)];
        NRPYELL_fields[IDX4pt(NRPYELL_SOL_PSI_BACKGROUNDGF, idx3)] = auxevol_gfs[IDX4(PSI_BACKGROUNDGF, i0, i1, i2)];
        NRPYELL_fields[IDX4pt(NRPYELL_SOL_ADD_TIMES_AUUGF, idx3)] = auxevol_gfs[IDX4(ADD_TIMES_AUUGF, i0, i1, i2)];
      }
    }
  }

  residual_H_compute_all_points(commondata, params, rfmstruct, auxevol_gfs, y_n_gfs, &NRPYELL_fields[IDX4pt(NRPYELL_SOL_RESIDUAL_HGF, 0)]);

  FILE *restrict fp = fopen("NRPYELL_solution.bin", "wb");
  if (fp == NULL) {
    perror("NRPYELL: Error opening NRPYELL_solution.bin for writing");
    free(NRPYELL_fields);
    exit(1);
  }

  checked_fwrite(NRPYELL_BINARY_MAGIC, sizeof(char), NRPYELL_BINARY_MAGIC_LEN, fp, "magic");
  checked_fwrite(&NRPYELL_VERSION, sizeof(NRPYELL_VERSION), 1, fp, "version");
  checked_fwrite(&NRPYELL_Nxx_plus_2NGHOSTS0, sizeof(NRPYELL_Nxx_plus_2NGHOSTS0), 1, fp, "Nxx_plus_2NGHOSTS0");
  checked_fwrite(&NRPYELL_Nxx_plus_2NGHOSTS1, sizeof(NRPYELL_Nxx_plus_2NGHOSTS1), 1, fp, "Nxx_plus_2NGHOSTS1");
  checked_fwrite(&NRPYELL_Nxx_plus_2NGHOSTS2, sizeof(NRPYELL_Nxx_plus_2NGHOSTS2), 1, fp, "Nxx_plus_2NGHOSTS2");
  checked_fwrite(&NRPYELL_NGHOSTS, sizeof(NRPYELL_NGHOSTS), 1, fp, "NGHOSTS");
  checked_fwrite(&NRPYELL_TOTAL_PTS, sizeof(NRPYELL_TOTAL_PTS), 1, fp, "TOTAL_PTS");
  checked_fwrite(&NRPYELL_NUM_FIELDS, sizeof(NRPYELL_NUM_FIELDS), 1, fp, "NUM_FIELDS");

  checked_fwrite(&NRPYELL_AMAX, sizeof(NRPYELL_AMAX), 1, fp, "AMAX");
  checked_fwrite(&NRPYELL_bScale, sizeof(NRPYELL_bScale), 1, fp, "bScale");
  checked_fwrite(&NRPYELL_SINHWAA, sizeof(NRPYELL_SINHWAA), 1, fp, "SINHWAA");
  checked_fwrite(&NRPYELL_dxx0, sizeof(NRPYELL_dxx0), 1, fp, "dxx0");
  checked_fwrite(&NRPYELL_dxx1, sizeof(NRPYELL_dxx1), 1, fp, "dxx1");
  checked_fwrite(&NRPYELL_dxx2, sizeof(NRPYELL_dxx2), 1, fp, "dxx2");

  checked_fwrite(xx[0], sizeof(REAL), (size_t)NRPYELL_Nxx_plus_2NGHOSTS0, fp, "xx0");
  checked_fwrite(xx[1], sizeof(REAL), (size_t)NRPYELL_Nxx_plus_2NGHOSTS1, fp, "xx1");
  checked_fwrite(xx[2], sizeof(REAL), (size_t)NRPYELL_Nxx_plus_2NGHOSTS2, fp, "xx2");
  checked_fwrite(NRPYELL_fields, sizeof(REAL), total_fields, fp, "solution fields");

  fclose(fp);
  free(NRPYELL_fields);
  printf("NRPYELL: FINISHED WRITING 3D enriched 'NRPYELL_solution.bin' with %d fields\n", NRPYELL_NUM_FIELDS);
}
