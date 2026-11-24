#include <stddef.h>
#include <stdint.h>
#include <math.h>

/* FIRST8 only. Buffers must be disjoint, as in MixedFactor/StreamedFirst8.
 * Preserve the original arithmetic order and compile without contraction. */
static size_t packed_offset(size_t n, size_t first, size_t j) {
    size_t d=j-first;
    return d*(2*(n-first)-d+1)/2;
}

int first8_validate(size_t n, const uint8_t *high, const size_t *offsets) {
    size_t h=n<8?n:8;
    for(size_t j=0;j<n;++j) {
        if(high[j]!=(uint8_t)(j<h)) return -1;
        if(offsets[j]!=packed_offset(n,j<h?0:h,j)) return -2;
    }
    return 0;
}

int mixed_column(size_t n, size_t j, const uint8_t *high,
                 const size_t *offsets, float *restrict low,
                 double *restrict wide, double *restrict work,
                 double *epsilon, double *raw_pivot) {
    if(j==0) {
        int info=first8_validate(n,high,offsets);
        if(info) return info;
    }
    size_t h=n<8?n:8;
    size_t limit=j<h?j:h;
    for(size_t k=0;k<limit;++k) {
        const double *column=wide+packed_offset(n,0,k)+(j-k);
        double ljk=column[0];
        for(size_t i=0;i<n-j;++i) work[i]-=column[i]*ljk;
    }
    for(size_t k=h;k<j;++k) {
        const float *column=low+packed_offset(n,h,k)+(j-k);
        double ljk=(double)column[0];
        for(size_t i=0;i<n-j;++i) work[i]-=(double)column[i]*ljk;
    }
    *raw_pivot=work[0];
    if(!(work[0]>0.0) || !isfinite(work[0])) return (int)j+1;
    double diagonal=sqrt(work[0]);
    double squares=0.0;
    if(j<h) {
        double *column=wide+packed_offset(n,0,j);
        column[0]=diagonal;
        for(size_t i=1;i<n-j;++i) column[i]=work[i]/diagonal;
        double stored_diagonal=column[0];
        for(size_t i=0;i<n-j;++i) {
            double defect=work[i]-stored_diagonal*column[i];
            squares+=defect*defect;
        }
    } else {
        float *column=low+packed_offset(n,h,j);
        column[0]=(float)diagonal;
        for(size_t i=1;i<n-j;++i) column[i]=(float)(work[i]/diagonal);
        double stored_diagonal=(double)column[0];
        for(size_t i=0;i<n-j;++i) {
            double defect=work[i]-stored_diagonal*(double)column[i];
            squares+=defect*defect;
        }
    }
    *epsilon=sqrt(squares);
    return 0;
}

int mixed_solve(size_t n, const uint8_t *high, const size_t *offsets,
                const float *restrict low, const double *restrict wide,
                double *restrict rhs) {
    int info=first8_validate(n,high,offsets);
    if(info) return info;
    size_t h=n<8?n:8;
    for(size_t j=0;j<h;++j) {
        const double *column=wide+packed_offset(n,0,j);
        double diagonal=column[0];
        if(!(diagonal>0.0) || !isfinite(diagonal)) return (int)j+1;
        rhs[j]/=diagonal;
        double value=rhs[j];
        for(size_t i=j+1;i<n;++i) rhs[i]-=column[i-j]*value;
    }
    for(size_t j=h;j<n;++j) {
        const float *column=low+packed_offset(n,h,j);
        double diagonal=(double)column[0];
        if(!(diagonal>0.0) || !isfinite(diagonal)) return (int)j+1;
        rhs[j]/=diagonal;
        double value=rhs[j];
        for(size_t i=j+1;i<n;++i) rhs[i]-=(double)column[i-j]*value;
    }
    for(size_t j=n;j-->h;) {
        const float *column=low+packed_offset(n,h,j);
        double value=rhs[j];
        for(size_t i=j+1;i<n;++i) value-=(double)column[i-j]*rhs[i];
        rhs[j]=value/(double)column[0];
    }
    for(size_t j=h;j-->0;) {
        const double *column=wide+packed_offset(n,0,j);
        double value=rhs[j];
        for(size_t i=j+1;i<n;++i) value-=column[i-j]*rhs[i];
        rhs[j]=value/column[0];
    }
    return 0;
}
