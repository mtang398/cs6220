"""SPD diagnostic of information in an early CG prefix; not a kernel benchmark."""
import json
from pathlib import Path
from hashlib import sha256
from time import perf_counter
import numpy as np
from scipy.linalg import solve
from threadpoolctl import threadpool_limits
root=Path(__file__).resolve().parents[1];out=root/'results/prefix-information-diagnostic.json'
if out.exists():raise FileExistsError(out)
start=perf_counter()
def cg(A,b):
 x=np.zeros_like(b);r=b.copy();p=r.copy();rr=r@r;history=[];vectors=[]
 for it in range(1,401):
  ap=A@p;alpha=rr/(p@ap);x+=alpha*p;r-=alpha*ap
  new=r@r;beta=new/rr
  history.append([float(alpha),float(beta),float(np.sqrt(new))])
  if it<=10:vectors.append(p.copy())
  if np.linalg.norm(b-A@x)<=1e-8*np.linalg.norm(b):break
  p=r+beta*p;rr=new
 return dict(iterations=it,true_residual=float(np.linalg.norm(b-A@x)),first10=history[:10]),vectors
with threadpool_limits(limits=1):
 n=120;b=np.zeros(n);b[0]=1.;J=np.arange(56,n);R=np.arange(56)
 results=[];prefixes=[]
 for tail_diagonal in [2.001,10.001]:
  diagonal=np.full(n,2.001);diagonal[20:]=tail_diagonal
  A=np.diag(diagonal)-np.diag(np.ones(n-1),1)-np.diag(np.ones(n-1),-1)
  base,P=cg(A,b)
  D=A[np.ix_(R,R)]-A[np.ix_(R,J)]@solve(A[np.ix_(J,J)],A[np.ix_(J,R)],assume_a='pos')
  reduced,_=cg(D,b[R])
  prefixes.append(np.column_stack(P))
  results.append(dict(tail_diagonal=tail_diagonal,smallest_eigenvalue=float(np.linalg.eigvalsh(A)[0]),baseline=base,eliminate_last64=reduced))
 scalar_gap=float(np.max(np.abs(np.array(results[0]['baseline']['first10'])-np.array(results[1]['baseline']['first10']))))
 vector_gap=float(np.max(np.abs(prefixes[0]-prefixes[1])))
 assert scalar_gap==0 and vector_gap==0
 assert all(x['smallest_eigenvalue']>0 for x in results)
 assert all(x[y]['true_residual']<=1e-8 for x in results for y in ['baseline','eliminate_last64'])
result=dict(scope='Independent SPD information diagnostic only; matrices are not Gaussian kernel instances, dense arrays are validation-only, and no native timing or resource-cap claim is made.',protocol=dict(n=n,rhs='first coordinate',identical_first_rows=20,prefix_length=10,eliminated_coordinates=J.tolist(),rtol=1e-8),cases=results,first10_scalar_gap=scalar_gap,first10_direction_gap=vector_gap,source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),elapsed_wall_seconds=perf_counter()-start)
out.write_text(json.dumps(result,indent=2)+'\n')
print([(r['baseline']['iterations'],r['eliminate_last64']['iterations']) for r in results])
