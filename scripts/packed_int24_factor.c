#include <stddef.h>
#include <stdint.h>
#include <limits.h>
#include <math.h>

/* Packed lower Cholesky factor, n >= 8.
 * wide: first eight complete columns (8*n-28 doubles).
 * diag: n doubles, including duplicate first-eight diagonal entries.
 * scales: n doubles; the first eight and last slots are zero.
 * low: 3*(n-8)*(n-9)/2 bytes of strict-lower entries in columns >= 8.
 * Buffers must be disjoint. No routine allocates or decodes a full factor.
 * Compile without fast-math/reassociation; nearbyint assumes FE_TONEAREST.
 * Returns 0 on success, j+1 for a nonpositive construction pivot,
 * -1 for invalid arguments, -2 for an invalid solve/product diagonal,
 * -3 for nonfinite arithmetic, and -4 for an unrepresentable scale/code.
 */

static int valid_size(size_t n) {
    return n >= 8 && n <= INT_MAX && n <= SIZE_MAX / 3 / n;
}

static size_t wide_offset(size_t n, size_t j) {
    return j*n - j*(j-1)/2;
}

static size_t low_offset(size_t n, size_t j) {
    return (j-8)*(2*n-j-9)/2;
}

static int32_t unpack24(const uint8_t *p) {
    uint32_t u = (uint32_t)p[0] | ((uint32_t)p[1] << 8)
               | ((uint32_t)p[2] << 16);
    return (u & UINT32_C(0x800000))
        ? (int32_t)u - INT32_C(16777216) : (int32_t)u;
}

static void pack24(uint8_t *p, int32_t code) {
    uint32_t u = (uint32_t)code & UINT32_C(0xffffff);
    p[0] = (uint8_t)u;
    p[1] = (uint8_t)(u >> 8);
    p[2] = (uint8_t)(u >> 16);
}

static double entry(size_t n, size_t i, size_t j,
                    const uint8_t *low, const double *wide,
                    const double *diag, const double *scales) {
    if (i == j) return diag[j];
    if (j < 8) return wide[wide_offset(n,j) + i-j];
    return scales[j] * (double)unpack24(low + 3*(low_offset(n,j)+i-j-1));
}

int int24_column(size_t n, size_t j, uint8_t *low, double *wide,
                 double *diag, double *scales, double *work,
                 double *epsilon, double *pivot) {
    if (!valid_size(n) || j >= n || !low || !wide || !diag || !scales
        || !work || !epsilon || !pivot) return -1;
    *epsilon = NAN;
    *pivot = NAN;
    for (size_t k=0; k<j; ++k) {
        double ljk = entry(n,j,k,low,wide,diag,scales);
        for (size_t i=j; i<n; ++i)
            work[i-j] -= entry(n,i,k,low,wide,diag,scales)*ljk;
    }
    *pivot = work[0];
    if (!isfinite(work[0])) return -3;
    if (!(work[0] > 0.0)) return (int)j+1;
    double diagonal = sqrt(work[0]);
    if (!(diagonal > 0.0) || !isfinite(diagonal)) return -3;
    diag[j] = diagonal;
    scales[j] = 0.0;
    if (j < 8) {
        double *column = wide + wide_offset(n,j);
        column[0] = diagonal;
        for (size_t i=j+1; i<n; ++i) {
            double v = work[i-j]/diagonal;
            if (!isfinite(v)) return -3;
            column[i-j] = v;
        }
    } else if (j+1 < n) {
        double maximum = 0.0;
        for (size_t i=j+1; i<n; ++i) {
            double v = work[i-j]/diagonal;
            if (!isfinite(v)) return -3;
            maximum = fmax(maximum,fabs(v));
        }
        double scale = maximum/8388607.0;
        if (!isfinite(scale) || (maximum > 0.0 && !(scale > 0.0))) return -4;
        scales[j] = scale;
        uint8_t *column = low + 3*low_offset(n,j);
        for (size_t i=j+1; i<n; ++i) {
            double code = scale == 0.0 ? 0.0 : nearbyint((work[i-j]/diagonal)/scale);
            if (!isfinite(code) || code < -8388607.0 || code > 8388607.0) return -4;
            pack24(column+3*(i-j-1),(int32_t)code);
        }
    }
    /* Same meaning as mixed_column_factor_first8.c: norm of the local
     * residual work minus the product of the STORED column and diagonal.
     * It includes diagonal/sqrt/division defects, not just quantization.
     * Earlier-column accumulation and this norm's rounding are separate.
     */
    double squares = 0.0;
    for (size_t i=j; i<n; ++i) {
        double defect = work[i-j] - diagonal*entry(n,i,j,low,wide,diag,scales);
        squares += defect*defect;
    }
    if (!isfinite(squares)) return -3;
    *epsilon = sqrt(squares);
    return 0;
}

static int valid_factor(size_t n, const double *diag, const double *scales) {
    for (size_t j=0; j<n; ++j) {
        if (!(diag[j] > 0.0) || !isfinite(diag[j])) return -2;
        if (!(scales[j] >= 0.0) || !isfinite(scales[j])) return -3;
    }
    return 0;
}

int int24_solve(size_t n, const uint8_t *low, const double *wide,
                const double *diag, const double *scales, double *rhs) {
    if (!valid_size(n) || !low || !wide || !diag || !scales || !rhs) return -1;
    int status = valid_factor(n,diag,scales);
    if (status) return status;
    for (size_t j=0; j<n; ++j) {
        rhs[j] /= diag[j];
        double value = rhs[j];
        for (size_t i=j+1; i<n; ++i)
            rhs[i] -= entry(n,i,j,low,wide,diag,scales)*value;
    }
    for (size_t j=n; j-- > 0;) {
        double value = rhs[j];
        for (size_t i=j+1; i<n; ++i)
            value -= entry(n,i,j,low,wide,diag,scales)*rhs[i];
        rhs[j] = value/diag[j];
        if (!isfinite(rhs[j])) return -3;
    }
    return 0;
}

int int24_product(size_t n, const uint8_t *low, const double *wide,
                  const double *diag, const double *scales,
                  const double *q, double *tmp, double *out) {
    if (!valid_size(n) || !low || !wide || !diag || !scales
        || !q || !tmp || !out) return -1;
    int status = valid_factor(n,diag,scales);
    if (status) return status;
    for (size_t j=0; j<n; ++j) {
        double value = diag[j]*q[j];
        for (size_t i=j+1; i<n; ++i)
            value += entry(n,i,j,low,wide,diag,scales)*q[i];
        tmp[j] = value;
        out[j] = 0.0;
    }
    for (size_t j=0; j<n; ++j) {
        out[j] += diag[j]*tmp[j];
        for (size_t i=j+1; i<n; ++i)
            out[i] += entry(n,i,j,low,wide,diag,scales)*tmp[j];
    }
    for (size_t i=0; i<n; ++i)
        if (!isfinite(out[i])) return -3;
    return 0;
}
