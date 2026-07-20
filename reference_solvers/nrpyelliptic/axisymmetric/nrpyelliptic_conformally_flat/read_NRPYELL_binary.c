#include "BHaH_defines.h"
#include "NRPYELL_binary_format.h"

static void checked_fread(void *restrict data, const size_t size, const size_t count, FILE *restrict fp, const char *restrict label) {
  if (fread(data, size, count, fp) != count) {
    fprintf(stderr, "NRPYELL: Error reading %s\n", label);
    fclose(fp);
    exit(1);
  }
}

/**
 * Reads enriched NRPyElliptic data from binary file NRPYELL_solution.bin.
 */
void read_NRPYELL_binary(int *restrict NRPYELL_Nxx_plus_2NGHOSTS0, int *restrict NRPYELL_Nxx_plus_2NGHOSTS1, int *restrict NRPYELL_Nxx_plus_2NGHOSTS2,
                         int *restrict NRPYELL_NGHOSTS, int *restrict NRPYELL_TOTAL_PTS, int *restrict NRPYELL_NUM_FIELDS,
                         REAL *restrict NRPYELL_AMAX, REAL *restrict NRPYELL_bScale, REAL *restrict NRPYELL_SINHWAA,
                         REAL *restrict NRPYELL_dxx0, REAL *restrict NRPYELL_dxx1, REAL *restrict NRPYELL_dxx2,
                         REAL *restrict *NRPYELL_xx0, REAL *restrict *NRPYELL_xx1, REAL *restrict *NRPYELL_xx2,
                         REAL *restrict *NRPYELL_fields) {

  FILE *restrict fp = fopen("NRPYELL_solution.bin", "rb");
  if (fp == NULL) {
    perror("NRPYELL: Error opening file 'NRPYELL_solution.bin' for reading");
    exit(1);
  }

  char magic[NRPYELL_BINARY_MAGIC_LEN];
  int version;
  checked_fread(magic, sizeof(char), NRPYELL_BINARY_MAGIC_LEN, fp, "magic");
  if (memcmp(magic, NRPYELL_BINARY_MAGIC, NRPYELL_BINARY_MAGIC_LEN) != 0) {
    fprintf(stderr, "NRPYELL: Unsupported binary magic. Expected enriched v2 NRPYELL_solution.bin.\n");
    fclose(fp);
    exit(1);
  }

  checked_fread(&version, sizeof(version), 1, fp, "version");
  if (version != NRPYELL_BINARY_VERSION) {
    fprintf(stderr, "NRPYELL: Unsupported binary version %d. Expected %d.\n", version, NRPYELL_BINARY_VERSION);
    fclose(fp);
    exit(1);
  }

  checked_fread(NRPYELL_Nxx_plus_2NGHOSTS0, sizeof(int), 1, fp, "Nxx_plus_2NGHOSTS0");
  checked_fread(NRPYELL_Nxx_plus_2NGHOSTS1, sizeof(int), 1, fp, "Nxx_plus_2NGHOSTS1");
  checked_fread(NRPYELL_Nxx_plus_2NGHOSTS2, sizeof(int), 1, fp, "Nxx_plus_2NGHOSTS2");
  checked_fread(NRPYELL_NGHOSTS, sizeof(int), 1, fp, "NGHOSTS");
  checked_fread(NRPYELL_TOTAL_PTS, sizeof(int), 1, fp, "TOTAL_PTS");
  checked_fread(NRPYELL_NUM_FIELDS, sizeof(int), 1, fp, "NUM_FIELDS");

  if (*NRPYELL_NUM_FIELDS != NRPYELL_SOL_NUM_GFS) {
    fprintf(stderr, "NRPYELL: Binary contains %d fields, but reader expects %d.\n", *NRPYELL_NUM_FIELDS, NRPYELL_SOL_NUM_GFS);
    fclose(fp);
    exit(1);
  }

  const size_t N0 = (size_t)*NRPYELL_Nxx_plus_2NGHOSTS0;
  const size_t N1 = (size_t)*NRPYELL_Nxx_plus_2NGHOSTS1;
  const size_t N2 = (size_t)*NRPYELL_Nxx_plus_2NGHOSTS2;
  const size_t total_pts = (size_t)*NRPYELL_TOTAL_PTS;
  if (N0 * N1 * N2 != total_pts) {
    fprintf(stderr, "NRPYELL: Grid dimensions imply %zu points, but binary reports %zu points.\n", N0 * N1 * N2, total_pts);
    fclose(fp);
    exit(1);
  }

  checked_fread(NRPYELL_AMAX, sizeof(REAL), 1, fp, "AMAX");
  checked_fread(NRPYELL_bScale, sizeof(REAL), 1, fp, "bScale");
  checked_fread(NRPYELL_SINHWAA, sizeof(REAL), 1, fp, "SINHWAA");
  checked_fread(NRPYELL_dxx0, sizeof(REAL), 1, fp, "dxx0");
  checked_fread(NRPYELL_dxx1, sizeof(REAL), 1, fp, "dxx1");
  checked_fread(NRPYELL_dxx2, sizeof(REAL), 1, fp, "dxx2");

  *NRPYELL_xx0 = (REAL *restrict)malloc(N0 * sizeof(REAL));
  *NRPYELL_xx1 = (REAL *restrict)malloc(N1 * sizeof(REAL));
  *NRPYELL_xx2 = (REAL *restrict)malloc(N2 * sizeof(REAL));
  const size_t total_fields = (size_t)(*NRPYELL_NUM_FIELDS) * total_pts;
  *NRPYELL_fields = (REAL *restrict)malloc(total_fields * sizeof(REAL));
  if (*NRPYELL_xx0 == NULL || *NRPYELL_xx1 == NULL || *NRPYELL_xx2 == NULL || *NRPYELL_fields == NULL) {
    perror("NRPYELL: Memory allocation failed while reading solution binary");
    fclose(fp);
    exit(1);
  }

  checked_fread(*NRPYELL_xx0, sizeof(REAL), N0, fp, "xx0");
  checked_fread(*NRPYELL_xx1, sizeof(REAL), N1, fp, "xx1");
  checked_fread(*NRPYELL_xx2, sizeof(REAL), N2, fp, "xx2");
  checked_fread(*NRPYELL_fields, sizeof(REAL), total_fields, fp, "solution fields");

  fclose(fp);
  printf("NRPYELL: FINISHED READING enriched 'NRPYELL_solution.bin'\n");
  printf("NRPYELL_Nxx_plus_2NGHOSTS0 = %d\n", *NRPYELL_Nxx_plus_2NGHOSTS0);
  printf("NRPYELL_Nxx_plus_2NGHOSTS1 = %d\n", *NRPYELL_Nxx_plus_2NGHOSTS1);
  printf("NRPYELL_Nxx_plus_2NGHOSTS2 = %d\n", *NRPYELL_Nxx_plus_2NGHOSTS2);
  printf("NRPYELL_NGHOSTS   = %d\n", *NRPYELL_NGHOSTS);
  printf("NRPYELL_TOTAL_PTS = %d\n", *NRPYELL_TOTAL_PTS);
  printf("NRPYELL_NUM_FIELDS = %d\n", *NRPYELL_NUM_FIELDS);
  printf("NRPYELL_AMAX    = %6.4e\n", *NRPYELL_AMAX);
  printf("NRPYELL_bScale  = %6.4e\n", *NRPYELL_bScale);
  printf("NRPYELL_SINHWAA = %6.4e\n", *NRPYELL_SINHWAA);
  printf("NRPYELL_dxx0 = %6.4e\n", *NRPYELL_dxx0);
  printf("NRPYELL_dxx1 = %6.4e\n", *NRPYELL_dxx1);
  printf("NRPYELL_dxx2 = %6.4e\n", *NRPYELL_dxx2);
} // END FUNCTION read_NRPYELL_binary
