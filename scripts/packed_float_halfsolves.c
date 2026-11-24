#include <stddef.h>
#include <math.h>

static size_t start(size_t n, size_t j) { return j*n-j*(j-1)/2; }

/* No allocation: FP64 RHS and arithmetic, promoted FP32 coefficients. */
int packed_float_forward(size_t n, const float *packed, double *rhs) {
    for (size_t j=0; j<n; ++j) {
        size_t offset=start(n,j);
        double diagonal=(double)packed[offset];
        if (!(diagonal>0.0) || !isfinite(diagonal)) return (int)j+1;
        rhs[j]/=diagonal;
        double value=rhs[j];
        for (size_t i=j+1; i<n; ++i)
            rhs[i]-=(double)packed[offset+i-j]*value;
    }
    return 0;
}

int packed_float_backward(size_t n, const float *packed, double *rhs) {
    for (size_t j=n; j-->0;) {
        size_t offset=start(n,j);
        double diagonal=(double)packed[offset];
        if (!(diagonal>0.0) || !isfinite(diagonal)) return (int)j+1;
        double value=rhs[j];
        for (size_t i=j+1; i<n; ++i)
            value-=(double)packed[offset+i-j]*rhs[i];
        rhs[j]=value/diagonal;
    }
    return 0;
}
