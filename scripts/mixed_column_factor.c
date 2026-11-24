#include <stddef.h>
#include <stdint.h>
#include <math.h>

static double get(size_t j, size_t local, const uint8_t *high,
                  const size_t *offsets, const float *low, const double *wide) {
    return high[j] ? wide[offsets[j]+local] : (double)low[offsets[j]+local];
}
static void put(size_t j, size_t local, double value, const uint8_t *high,
                const size_t *offsets, float *low, double *wide) {
    if(high[j]) wide[offsets[j]+local]=value;
    else low[offsets[j]+local]=(float)value;
}

/* Rebuild from an original FP64 C[j:n,j] working column. */
int mixed_column(size_t n, size_t j, const uint8_t *high, const size_t *offsets,
                 float *low, double *wide, double *work,
                 double *epsilon, double *raw_pivot) {
    for(size_t k=0;k<j;++k) {
        double ljk=get(k,j-k,high,offsets,low,wide);
        for(size_t i=j;i<n;++i)
            work[i-j]-=get(k,i-k,high,offsets,low,wide)*ljk;
    }
    *raw_pivot=work[0];
    if(!(work[0]>0.0) || !isfinite(work[0])) return (int)j+1;
    double diagonal=sqrt(work[0]);
    put(j,0,diagonal,high,offsets,low,wide);
    for(size_t i=j+1;i<n;++i)
        put(j,i-j,work[i-j]/diagonal,high,offsets,low,wide);
    double stored_diagonal=get(j,0,high,offsets,low,wide);
    double squares=0.0;
    for(size_t i=j;i<n;++i) {
        double defect=work[i-j]-stored_diagonal*get(j,i-j,high,offsets,low,wide);
        squares+=defect*defect;
    }
    *epsilon=sqrt(squares);
    return 0;
}

int mixed_solve(size_t n, const uint8_t *high, const size_t *offsets,
                const float *low, const double *wide, double *rhs) {
    for(size_t j=0;j<n;++j) {
        double diagonal=get(j,0,high,offsets,low,wide);
        if(!(diagonal>0.0) || !isfinite(diagonal)) return (int)j+1;
        rhs[j]/=diagonal;
        double value=rhs[j];
        for(size_t i=j+1;i<n;++i)
            rhs[i]-=get(j,i-j,high,offsets,low,wide)*value;
    }
    for(size_t j=n;j-->0;) {
        double value=rhs[j];
        for(size_t i=j+1;i<n;++i)
            value-=get(j,i-j,high,offsets,low,wide)*rhs[i];
        rhs[j]=value/get(j,0,high,offsets,low,wide);
    }
    return 0;
}
