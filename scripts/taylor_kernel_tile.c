#include <stddef.h>
/* Separate input/output tiles: both charged. No fast-math reassociation. */
void taylor_tile(const double *g, double *h, const double *d,
                 size_t n, size_t start, size_t rows, int degree) {
    size_t len=n*rows;
    for(size_t a=0;a<len;++a) h[a]=1.0;
    for(int k=degree;k>=1;--k) {
        double inv=1.0/k;
        for(size_t a=0;a<len;++a) h[a]=1.0+g[a]*inv*h[a];
    }
    for(size_t i=0;i<rows;++i)
        for(size_t j=0;j<n;++j) h[i*n+j]*=d[start+i]*d[j];
}
