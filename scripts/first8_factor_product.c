#include <stddef.h>
#include <stdint.h>
/* Caller supplies disjoint vector, temporary and output arrays. */
int first8_factor_product(size_t n, const uint8_t *high, const size_t *offsets,
                         const float *low, const double *wide,
                         const double *vector, double *temporary, double *out) {
    const size_t h=n<8?n:8;
    for(size_t j=0;j<n;++j) {
        if(high[j]!=(uint8_t)(j<h)) return -1;
        size_t first=j<h?0:h,d=j-first;
        if(offsets[j]!=d*(2*(n-first)-d+1)/2) return -2;
    }
    for(size_t j=0;j<n;++j) {
        double sum=0.0;
        if(j<h) {const double *c=wide+offsets[j];for(size_t i=j;i<n;++i)sum+=c[i-j]*vector[i];}
        else {const float *c=low+offsets[j];for(size_t i=j;i<n;++i)sum+=(double)c[i-j]*vector[i];}
        temporary[j]=sum;out[j]=0.0;
    }
    for(size_t j=0;j<n;++j) {
        double value=temporary[j];
        if(j<h) {const double *c=wide+offsets[j];for(size_t i=j;i<n;++i)out[i]+=c[i-j]*value;}
        else {const float *c=low+offsets[j];for(size_t i=j;i<n;++i)out[i]+=(double)c[i-j]*value;}
    }
    return 0;
}
