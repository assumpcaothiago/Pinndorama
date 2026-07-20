#include "BHaH_defines.h"
#include "neural_net_weights.h"

#ifndef PINNDORAMA_GENERATED_PARAMETRIC_NN_HEADER
#error "neural_net_weights.h must be generated from a provenance-complete NPZ checkpoint"
#endif

static bool nearly_equal(const REAL a, const REAL b) {
  const REAL scale = MAX((REAL)1.0, MAX(fabs(a), fabs(b)));
  return fabs(a - b) <= (REAL)1.0e-12 * scale;
}

static void require_match(const char *restrict name, const REAL runtime_value, const REAL checkpoint_value) {
  if (!nearly_equal(runtime_value, checkpoint_value)) {
    fprintf(stderr, "ERROR: runtime %s=%.17e differs from NN checkpoint %s=%.17e\n", name, runtime_value, name, checkpoint_value);
    exit(1);
  }
}

/** Validate all immutable NN inputs once, before entering an OpenMP loop. */
void validate_parametric_nn_compatibility(const commondata_struct *restrict commondata, const params_struct *restrict params) {
  typedef char nn_requires_64_bit_REAL[(sizeof(REAL) == 8) ? 1 : -1];
  MAYBE_UNUSED nn_requires_64_bit_REAL dtype_check;

  if (strcmp(params->CoordSystemName, NN_COORDINATE_SYSTEM) != 0) {
    fprintf(stderr, "ERROR: NN checkpoint requires coordinate system %s, got %s\n", NN_COORDINATE_SYSTEM, params->CoordSystemName);
    exit(1);
  }
  if (!nearly_equal(params->AMAX, NN_CHECKPOINT_AMAX)) {
    fprintf(stderr, "ERROR: runtime AMAX=%.17e differs from NN checkpoint AMAX=%.17e\n", params->AMAX, NN_CHECKPOINT_AMAX);
    exit(1);
  }
  if (!nearly_equal(params->bScale, NN_CHECKPOINT_BSCALE)) {
    fprintf(stderr, "ERROR: runtime bScale=%.17e differs from NN checkpoint bScale=%.17e\n", params->bScale, NN_CHECKPOINT_BSCALE);
    exit(1);
  }
  if (!nearly_equal(params->SINHWAA, NN_CHECKPOINT_SINHWAA)) {
    fprintf(stderr, "ERROR: runtime SINHWAA=%.17e differs from NN checkpoint SINHWAA=%.17e\n", params->SINHWAA, NN_CHECKPOINT_SINHWAA);
    exit(1);
  }

  require_match("bare_mass_0", commondata->bare_mass_0, NN_SOURCE_BARE_MASS_0);
  require_match("bare_mass_1", commondata->bare_mass_1, NN_SOURCE_BARE_MASS_1);
  require_match("zPunc", commondata->zPunc, NN_SOURCE_ZPUNC);
  require_match("P0_x", commondata->P0_x, NN_SOURCE_P0_X);
  require_match("P0_y", commondata->P0_y, NN_SOURCE_P0_Y);
  require_match("P0_z", commondata->P0_z, NN_SOURCE_P0_Z);
  require_match("P1_x", commondata->P1_x, NN_SOURCE_P1_X);
  require_match("P1_y", commondata->P1_y, NN_SOURCE_P1_Y);
  require_match("P1_z", commondata->P1_z, NN_SOURCE_P1_Z);
  require_match("S0_x", commondata->S0_x, NN_SOURCE_S0_X);
  require_match("S0_y", commondata->S0_y, NN_SOURCE_S0_Y);
  require_match("S1_x", commondata->S1_x, NN_SOURCE_S1_X);
  require_match("S1_y", commondata->S1_y, NN_SOURCE_S1_Y);

  if (!nearly_equal(commondata->S0_z, commondata->S1_z)) {
    fprintf(stderr, "ERROR: NN warm start requires equal raw spins S0_z=S1_z; got %.17e and %.17e\n", commondata->S0_z, commondata->S1_z);
    exit(1);
  }
  if (commondata->S0_z < NN_EQUAL_SPIN_SZ_MIN || commondata->S0_z > NN_EQUAL_SPIN_SZ_MAX) {
    fprintf(stderr, "ERROR: equal_spin_sz=%.17e is outside NN checkpoint range [%.17e, %.17e]\n", commondata->S0_z, NN_EQUAL_SPIN_SZ_MIN,
            NN_EQUAL_SPIN_SZ_MAX);
    exit(1);
  }
  fprintf(stderr, "Using parametric NN warm start: checkpoint sha256=%s, equal_spin_sz=%.17e\n", NN_CHECKPOINT_SHA256, commondata->S0_z);
}

REAL eval_uNN(const REAL xx0, const REAL xx1, const REAL S) {
  REAL h0[NN_WIDTH];
  REAL h1[NN_WIDTH];
  REAL h2[NN_WIDTH];
  REAL h3[NN_WIDTH];

  const REAL input[NN_INPUT_DIM] = {xx0, xx1, S};

  for (int j = 0; j < NN_WIDTH; j++) {
    REAL value = NN_B0[j];
    for (int i = 0; i < NN_INPUT_DIM; i++)
      value += input[i] * NN_W0[i][j];
    h0[j] = tanh(value);
  }

  for (int j = 0; j < NN_WIDTH; j++) {
    REAL value = NN_B1[j];
    for (int i = 0; i < NN_WIDTH; i++)
      value += h0[i] * NN_W1[i][j];
    h1[j] = tanh(value);
  }

  for (int j = 0; j < NN_WIDTH; j++) {
    REAL value = NN_B2[j];
    for (int i = 0; i < NN_WIDTH; i++)
      value += h1[i] * NN_W2[i][j];
    h2[j] = tanh(value);
  }

  for (int j = 0; j < NN_WIDTH; j++) {
    REAL value = NN_B3[j];
    for (int i = 0; i < NN_WIDTH; i++)
      value += h2[i] * NN_W3[i][j];
    h3[j] = tanh(value);
  }

  REAL raw_out = NN_B4[0];
  for (int i = 0; i < NN_WIDTH; i++)
    raw_out += h3[i] * NN_W4[i][0];

  const REAL AA = NN_CHECKPOINT_AMAX * sinh(xx0 / NN_CHECKPOINT_SINHWAA) / sinh((REAL)1.0 / NN_CHECKPOINT_SINHWAA);
  const REAL cos_xx1 = cos(xx1);
  const REAL r_sph = sqrt(AA * AA + NN_CHECKPOINT_BSCALE * NN_CHECKPOINT_BSCALE * cos_xx1 * cos_xx1);
  return raw_out / sqrt((REAL)1.0 + r_sph * r_sph);
}

/**
 * Compute initial guess from the generated neural-network header at one point.
 */
void neural_net_guess_single_point(const commondata_struct *restrict commondata, const params_struct *restrict params, const REAL xx0, const REAL xx1,
                                   const REAL xx2, REAL *restrict uu_ID, REAL *restrict vv_ID) {
  MAYBE_UNUSED const REAL ignored_xx2 = xx2;
  const REAL S = commondata->S0_z;
  const REAL uNN = eval_uNN(xx0, xx1, S);
  *uu_ID = uNN;
  *vv_ID = commondata->eta_damping * uNN;
}
