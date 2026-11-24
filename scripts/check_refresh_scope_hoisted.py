"""Frozen 15-case scope, applying the common accounting hoist to every method."""
import hashlib,inspect,json,textwrap,sys
from pathlib import Path
import check_refresh_regularization_scope as scope
import check_kernel_accounting_hoist as hoist
ROOT=scope.ROOT
TEXT=textwrap.dedent(inspect.getsource(scope.main))
REPLACEMENTS=[("ROOT/'results/refresh-scope'","ROOT/'results/refresh-scope-hoisted'"),("for method in ['adaptive','periodic8']:","for method in METHODS:"),("def invoke(method):","def invoke(method):\n        cap=8 if tracemalloc.is_tracing() else 7"),("cpu_cap=7","cpu_cap=cap"),("return row","audit_prior_work(row,args.dataset,args.lam)\n        return row")]
MODIFIED=TEXT
for before,after in REPLACEMENTS:
    assert before in MODIFIED
    MODIFIED=MODIFIED.replace(before,after)
MODIFIED=MODIFIED.replace("audits=[],runs=[],failures=[])","audits=[],runs=[],failures=[],active_implementation=implementation_metadata())")
assert 'active_implementation=implementation_metadata()' in MODIFIED

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
CACHE={}
def audit_prior_work(row,dataset,lam):
    key=(dataset,lam)
    if key not in CACHE:
        path=ROOT/f'results/refresh-scope/{dataset.lower()}-lambda{lam:g}.json'
        prior=json.loads(path.read_text());assert all(sha(ROOT/p)==h for p,h in prior['sources'].items())
        assert sha(ROOT/'data/prepared'/scope.DATASETS[dataset])==prior['data_sha256']
        CACHE[key]=(prior,path)
    prior,path=CACHE[key];old=[r for r in prior['runs'] if r['method']==row['method']]
    fields=['iterations','converged','kernel_stats','memory']
    checks={field:all(row.get(field)==r.get(field) for r in old) for field in fields}
    checks['refresh_steps']=all(row.get('refresh_steps')==r.get('refresh_steps') for r in old)
    checks['original_relative_residual']=all(row.get('original_relative_residual',row.get('true_relative_residual'))==r.get('original_relative_residual',r.get('true_relative_residual')) for r in old)
    row['old_scope_work_identity']=checks
    row['old_scope_result']=str(path.relative_to(ROOT));row['old_scope_result_sha256']=sha(path)
    assert all(checks.values()),checks

def implementation_metadata():
    return dict(shared_kernel_patch='All five methods use common hoisted SharedKernel.product, restored in finally.',before=hoist.BEFORE,after=hoist.AFTER,original_product_sha256=hashlib.sha256(hoist.TEXT.encode()).hexdigest(),hoisted_product_sha256=hashlib.sha256(hoist.MODIFIED.encode()).hexdigest(),execution='Private compilation of prior scope main: change destination, audit all five methods, trace cap8/native cap7, exact prior-work checks, and this metadata. Original module/source untouched.',trace_audit_methods=scope.METHODS,scope='Larger-case extension only; n512 and held-out cases pending. Two prior pilot cases reused separately; no absolute cross-batch runtime claim.')

def main():
    namespace=dict(scope.main.__globals__);namespace.update(audit_prior_work=audit_prior_work,implementation_metadata=implementation_metadata)
    exec(compile(MODIFIED,'<scope-with-common-accounting-hoist>','exec'),namespace)
    with hoist.variant('hoisted'):namespace['main']()
if __name__=='__main__':main()
