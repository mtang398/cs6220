#include <math.h>
#include <stddef.h>
/* Dense oracle coordinate flips. Q/L row-major n*n; U,V,R,T row-major n*d.
   Caller holds all arrays; no heap allocations. Scale fixed per column. */
size_t calibrated_rounding(size_t n,size_t d,const double *L,double *Q,const double *scales,const double *U,const double *V,double *R,double *T){
 size_t flips=0;
 for(int sweep=0;sweep<2;sweep++){
  for(size_t jj=8;jj+1<n;jj++){
   size_t j=sweep==0?jj:n+6-jj;double scale=scales[j];if(!(scale>0))continue;
   for(size_t i=j+1;i<n;i++){
    double x=L[i*n+j]/scale,lo=floor(x),hi=ceil(x);if(lo==hi)continue;
    double old=nearbyint(Q[i*n+j]/scale),other=fabs(old-lo)<.5?hi:lo;
    double delta=(other-old)*scale,linear=0,quadratic=0;
    for(size_t k=0;k<d;k++){double u=U[i*d+k],v=V[j*d+k];linear+=R[i*d+k]*v+T[j*d+k]*u;quadratic+=u*u+v*v;}
    if(2*delta*linear+delta*delta*quadratic<0){
     Q[i*n+j]=other*scale;flips++;
     for(size_t k=0;k<d;k++){R[i*d+k]+=delta*V[j*d+k];T[j*d+k]+=delta*U[i*d+k];}
    }
   }
  }
 }
 return flips;
}
