#include <stddef.h>
#include <math.h>

static size_t start(size_t n, size_t j) { return j*n-j*(j-1)/2; }

/* work holds the original FP64 column A[j:n,j]. No dynamic allocation. */
int packed_float_column(size_t n, size_t j, float *packed,
                        double *work, double *raw_pivot) {
    for(size_t k=0;k<j;++k) {
        size_t offset=start(n,k);
        double ljk=(double)packed[offset+j-k];
        for(size_t i=j;i<n;++i)
            work[i-j]-=(double)packed[offset+i-k]*ljk;
    }
    *raw_pivot=work[0];
    if(!(work[0]>0.0) || !isfinite(work[0])) return (int)j+1;
    double diagonal=sqrt(work[0]);
    size_t offset=start(n,j);
    packed[offset]=(float)diagonal;
    if(!(packed[offset]>0.0f) || !isfinite(packed[offset])) return (int)j+1;
    for(size_t i=j+1;i<n;++i) packed[offset+i-j]=(float)(work[i-j]/diagonal);
    return 0;
}

/* FP64 solution/RHS, FP32 coefficients promoted before each operation. */
int packed_float_solve(size_t n, const float *packed, double *rhs) {
    for(size_t j=0;j<n;++j) {
        size_t offset=start(n,j);
        double diagonal=(double)packed[offset];
        if(!(diagonal>0.0) || !isfinite(diagonal)) return (int)j+1;
        rhs[j]/=diagonal;
        double value=rhs[j];
        for(size_t i=j+1;i<n;++i) rhs[i]-=(double)packed[offset+i-j]*value;
    }
    for(size_t j=n;j-->0;) {
        size_t offset=start(n,j);
        double value=rhs[j];
        for(size_t i=j+1;i<n;++i) value-=(double)packed[offset+i-j]*rhs[i];
        rhs[j]=value/(double)packed[offset];
    }
    return 0;
}
