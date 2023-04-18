// Generated by Futhark 0.23.0
// git: 7fa00324492db398bc13c1bd640e2747a34eeac2
#pragma once


// Headers
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialisation
struct futhark_context_config;
struct futhark_context_config *futhark_context_config_new(void);
void futhark_context_config_free(struct futhark_context_config *cfg);
void futhark_context_config_set_debugging(struct futhark_context_config *cfg, int flag);
void futhark_context_config_set_profiling(struct futhark_context_config *cfg, int flag);
void futhark_context_config_set_logging(struct futhark_context_config *cfg, int flag);
struct futhark_context;
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg);
void futhark_context_free(struct futhark_context *ctx);
int futhark_context_config_set_tuning_param(struct futhark_context_config *cfg, const char *param_name, size_t param_value);
int futhark_get_tuning_param_count(void);
const char *futhark_get_tuning_param_name(int);
const char *futhark_get_tuning_param_class(int);

// Arrays
struct futhark_f32_1d;
struct futhark_f32_1d *futhark_new_f32_1d(struct futhark_context *ctx, const float *data, int64_t dim0);
struct futhark_f32_1d *futhark_new_raw_f32_1d(struct futhark_context *ctx, const unsigned char *data, int64_t offset, int64_t dim0);
int futhark_free_f32_1d(struct futhark_context *ctx, struct futhark_f32_1d *arr);
int futhark_values_f32_1d(struct futhark_context *ctx, struct futhark_f32_1d *arr, float *data);
unsigned char *futhark_values_raw_f32_1d(struct futhark_context *ctx, struct futhark_f32_1d *arr);
const int64_t *futhark_shape_f32_1d(struct futhark_context *ctx, struct futhark_f32_1d *arr);
struct futhark_f32_2d;
struct futhark_f32_2d *futhark_new_f32_2d(struct futhark_context *ctx, const float *data, int64_t dim0, int64_t dim1);
struct futhark_f32_2d *futhark_new_raw_f32_2d(struct futhark_context *ctx, const unsigned char *data, int64_t offset, int64_t dim0, int64_t dim1);
int futhark_free_f32_2d(struct futhark_context *ctx, struct futhark_f32_2d *arr);
int futhark_values_f32_2d(struct futhark_context *ctx, struct futhark_f32_2d *arr, float *data);
unsigned char *futhark_values_raw_f32_2d(struct futhark_context *ctx, struct futhark_f32_2d *arr);
const int64_t *futhark_shape_f32_2d(struct futhark_context *ctx, struct futhark_f32_2d *arr);

// Opaque values
struct futhark_opaque_b0aee03a;
struct futhark_opaque_8f5f2e9d;
struct futhark_opaque_8ce68ead;
struct futhark_opaque_67daac7;
struct futhark_opaque_f270e689;
struct futhark_opaque_32531da2;
struct futhark_opaque_e9a0e68b;
struct futhark_opaque_c5da5e9b;
struct futhark_opaque_networkParameter;
struct futhark_opaque_particleType;
int futhark_free_opaque_b0aee03a(struct futhark_context *ctx, struct futhark_opaque_b0aee03a *obj);
int futhark_store_opaque_b0aee03a(struct futhark_context *ctx, const struct futhark_opaque_b0aee03a *obj, void **p, size_t *n);
struct futhark_opaque_b0aee03a *futhark_restore_opaque_b0aee03a(struct futhark_context *ctx, const void *p);
int futhark_project_opaque_b0aee03a_0(struct futhark_context *ctx, struct futhark_opaque_8f5f2e9d **out, const struct futhark_opaque_b0aee03a *obj);
int futhark_project_opaque_b0aee03a_1(struct futhark_context *ctx, struct futhark_f32_1d **out, const struct futhark_opaque_b0aee03a *obj);
int futhark_new_opaque_b0aee03a(struct futhark_context *ctx, struct futhark_opaque_b0aee03a **out, const struct futhark_opaque_8f5f2e9d *v0, const struct futhark_f32_1d *v1);
int futhark_free_opaque_8f5f2e9d(struct futhark_context *ctx, struct futhark_opaque_8f5f2e9d *obj);
int futhark_store_opaque_8f5f2e9d(struct futhark_context *ctx, const struct futhark_opaque_8f5f2e9d *obj, void **p, size_t *n);
struct futhark_opaque_8f5f2e9d *futhark_restore_opaque_8f5f2e9d(struct futhark_context *ctx, const void *p);
int futhark_project_opaque_8f5f2e9d_0(struct futhark_context *ctx, struct futhark_opaque_8ce68ead **out, const struct futhark_opaque_8f5f2e9d *obj);
int futhark_project_opaque_8f5f2e9d_1(struct futhark_context *ctx, struct futhark_opaque_32531da2 **out, const struct futhark_opaque_8f5f2e9d *obj);
int futhark_new_opaque_8f5f2e9d(struct futhark_context *ctx, struct futhark_opaque_8f5f2e9d **out, const struct futhark_opaque_8ce68ead *v0, const struct futhark_opaque_32531da2 *v1);
int futhark_free_opaque_8ce68ead(struct futhark_context *ctx, struct futhark_opaque_8ce68ead *obj);
int futhark_store_opaque_8ce68ead(struct futhark_context *ctx, const struct futhark_opaque_8ce68ead *obj, void **p, size_t *n);
struct futhark_opaque_8ce68ead *futhark_restore_opaque_8ce68ead(struct futhark_context *ctx, const void *p);
int futhark_project_opaque_8ce68ead_0(struct futhark_context *ctx, struct futhark_opaque_f270e689 **out, const struct futhark_opaque_8ce68ead *obj);
int futhark_project_opaque_8ce68ead_1(struct futhark_context *ctx, struct futhark_opaque_32531da2 **out, const struct futhark_opaque_8ce68ead *obj);
int futhark_new_opaque_8ce68ead(struct futhark_context *ctx, struct futhark_opaque_8ce68ead **out, const struct futhark_opaque_f270e689 *v0, const struct futhark_opaque_32531da2 *v1);
int futhark_free_opaque_67daac7(struct futhark_context *ctx, struct futhark_opaque_67daac7 *obj);
int futhark_store_opaque_67daac7(struct futhark_context *ctx, const struct futhark_opaque_67daac7 *obj, void **p, size_t *n);
struct futhark_opaque_67daac7 *futhark_restore_opaque_67daac7(struct futhark_context *ctx, const void *p);
int futhark_project_opaque_67daac7_0(struct futhark_context *ctx, struct futhark_opaque_c5da5e9b **out, const struct futhark_opaque_67daac7 *obj);
int futhark_project_opaque_67daac7_1(struct futhark_context *ctx, struct futhark_opaque_b0aee03a **out, const struct futhark_opaque_67daac7 *obj);
int futhark_new_opaque_67daac7(struct futhark_context *ctx, struct futhark_opaque_67daac7 **out, const struct futhark_opaque_c5da5e9b *v0, const struct futhark_opaque_b0aee03a *v1);
int futhark_free_opaque_f270e689(struct futhark_context *ctx, struct futhark_opaque_f270e689 *obj);
int futhark_store_opaque_f270e689(struct futhark_context *ctx, const struct futhark_opaque_f270e689 *obj, void **p, size_t *n);
struct futhark_opaque_f270e689 *futhark_restore_opaque_f270e689(struct futhark_context *ctx, const void *p);
int futhark_project_opaque_f270e689_0(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_f270e689 *obj);
int futhark_project_opaque_f270e689_1(struct futhark_context *ctx, struct futhark_f32_1d **out, const struct futhark_opaque_f270e689 *obj);
int futhark_new_opaque_f270e689(struct futhark_context *ctx, struct futhark_opaque_f270e689 **out, const struct futhark_f32_2d *v0, const struct futhark_f32_1d *v1);
int futhark_free_opaque_32531da2(struct futhark_context *ctx, struct futhark_opaque_32531da2 *obj);
int futhark_store_opaque_32531da2(struct futhark_context *ctx, const struct futhark_opaque_32531da2 *obj, void **p, size_t *n);
struct futhark_opaque_32531da2 *futhark_restore_opaque_32531da2(struct futhark_context *ctx, const void *p);
int futhark_project_opaque_32531da2_0(struct futhark_context *ctx, struct futhark_f32_2d **out, const struct futhark_opaque_32531da2 *obj);
int futhark_project_opaque_32531da2_1(struct futhark_context *ctx, struct futhark_f32_1d **out, const struct futhark_opaque_32531da2 *obj);
int futhark_new_opaque_32531da2(struct futhark_context *ctx, struct futhark_opaque_32531da2 **out, const struct futhark_f32_2d *v0, const struct futhark_f32_1d *v1);
int futhark_free_opaque_e9a0e68b(struct futhark_context *ctx, struct futhark_opaque_e9a0e68b *obj);
int futhark_store_opaque_e9a0e68b(struct futhark_context *ctx, const struct futhark_opaque_e9a0e68b *obj, void **p, size_t *n);
struct futhark_opaque_e9a0e68b *futhark_restore_opaque_e9a0e68b(struct futhark_context *ctx, const void *p);
int futhark_project_opaque_e9a0e68b_0(struct futhark_context *ctx, struct futhark_opaque_particleType **out, const struct futhark_opaque_e9a0e68b *obj);
int futhark_project_opaque_e9a0e68b_1(struct futhark_context *ctx, float *out, const struct futhark_opaque_e9a0e68b *obj);
int futhark_project_opaque_e9a0e68b_2(struct futhark_context *ctx, int32_t *out, const struct futhark_opaque_e9a0e68b *obj);
int futhark_new_opaque_e9a0e68b(struct futhark_context *ctx, struct futhark_opaque_e9a0e68b **out, const struct futhark_opaque_particleType *v0, const float v1, const int32_t v2);
int futhark_free_opaque_c5da5e9b(struct futhark_context *ctx, struct futhark_opaque_c5da5e9b *obj);
int futhark_store_opaque_c5da5e9b(struct futhark_context *ctx, const struct futhark_opaque_c5da5e9b *obj, void **p, size_t *n);
struct futhark_opaque_c5da5e9b *futhark_restore_opaque_c5da5e9b(struct futhark_context *ctx, const void *p);
int futhark_free_opaque_networkParameter(struct futhark_context *ctx, struct futhark_opaque_networkParameter *obj);
int futhark_store_opaque_networkParameter(struct futhark_context *ctx, const struct futhark_opaque_networkParameter *obj, void **p, size_t *n);
struct futhark_opaque_networkParameter *futhark_restore_opaque_networkParameter(struct futhark_context *ctx, const void *p);
int futhark_project_opaque_networkParameter_0(struct futhark_context *ctx, struct futhark_opaque_e9a0e68b **out, const struct futhark_opaque_networkParameter *obj);
int futhark_project_opaque_networkParameter_1(struct futhark_context *ctx, struct futhark_opaque_67daac7 **out, const struct futhark_opaque_networkParameter *obj);
int futhark_new_opaque_networkParameter(struct futhark_context *ctx, struct futhark_opaque_networkParameter **out, const struct futhark_opaque_e9a0e68b *v0, const struct futhark_opaque_67daac7 *v1);
int futhark_free_opaque_particleType(struct futhark_context *ctx, struct futhark_opaque_particleType *obj);
int futhark_store_opaque_particleType(struct futhark_context *ctx, const struct futhark_opaque_particleType *obj, void **p, size_t *n);
struct futhark_opaque_particleType *futhark_restore_opaque_particleType(struct futhark_context *ctx, const void *p);

// Entry points
int futhark_entry_hgoInteraction(struct futhark_context *ctx, struct futhark_f32_2d **out0, const float in0, const float in1, const float in2, const struct futhark_f32_2d *in3, const struct futhark_f32_2d *in4);
int futhark_entry_hgoPotential(struct futhark_context *ctx, float *out0, const float in0, const float in1, const float in2, const struct futhark_f32_1d *in3, const struct futhark_f32_1d *in4);
int futhark_entry_networkInteraction(struct futhark_context *ctx, struct futhark_f32_2d **out0, const struct futhark_opaque_networkParameter *in0, const struct futhark_f32_2d *in1, const struct futhark_f32_2d *in2);
int futhark_entry_networkPotential(struct futhark_context *ctx, float *out0, const struct futhark_opaque_networkParameter *in0, const struct futhark_f32_1d *in1, const struct futhark_f32_1d *in2);

// Miscellaneous
int futhark_context_sync(struct futhark_context *ctx);
void futhark_context_config_set_cache_file(struct futhark_context_config *cfg, const char *f);
char *futhark_context_report(struct futhark_context *ctx);
char *futhark_context_get_error(struct futhark_context *ctx);
void futhark_context_set_logging_file(struct futhark_context *ctx, FILE *f);
void futhark_context_pause_profiling(struct futhark_context *ctx);
void futhark_context_unpause_profiling(struct futhark_context *ctx);
int futhark_context_clear_caches(struct futhark_context *ctx);
#define FUTHARK_BACKEND_c
#define FUTHARK_SUCCESS 0
#define FUTHARK_PROGRAM_ERROR 2
#define FUTHARK_OUT_OF_MEMORY 3

#ifdef __cplusplus
}
#endif
