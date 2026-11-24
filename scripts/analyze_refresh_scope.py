"""Descriptive paired timing summaries; censored solves are never completed costs."""
import json
from pathlib import Path
from hashlib import sha256
import math

ROOT=Path(__file__).resolve().parents[1]
METHODS=['adaptive','periodic8','uniform600','packed24','original_exact416']
STATIC=METHODS[2:]
EXPECTED=[(name,lam) for name in ['smooth','random','mixed','Concrete','Airfoil'] for lam in [.1,.001,1e-5]]

def main():
    folder=ROOT/'results/refresh-scope'
    result={'scope':'Larger-case seed0 development extension; not original n512 or held-out confirmation.',
            'interpretation':'Two repeats are descriptive, not a statistical significance claim. Best-static-per-case is an oracle diagnostic, not an implemented selector. No failed/capped solve is treated as a completed runtime.',
            'sources':{},'cases':[],'missing':[],'invalid':[]}
    for name,lam in EXPECTED:
        p=folder/f'{name.lower()}-lambda{lam:g}.json'
        if not p.exists():result['missing'].append([name,lam]);continue
        data=json.loads(p.read_text());result['sources'][str(p.relative_to(ROOT))]=sha256(p.read_bytes()).hexdigest()
        if data['status']!='complete':result['invalid'].append({'case':[name,lam],'status':data['status']});continue
        assert data['dataset']==name and data['regularization']==lam
        assert data['input_hashes_unchanged'] and data['source_hashes_unchanged']
        for path,digest in data['sources'].items():assert sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
        case={'dataset':name,'regularization':lam,'methods':{},'audits_pass':all(a['within_1MiB'] and a['metrics']['named_allocation_within_1MiB'] for a in data['audits'])}
        for method in METHODS:
            rows=sorted([r for r in data['runs'] if r['method']==method],key=lambda r:r['repeat'])
            assert [r['repeat'] for r in rows]==[0,1]
            complete=all(r.get('converged',False) and r['verified_original_relative_residual']<=1e-8 and r['named_allocation_within_1MiB'] for r in rows)
            entry={'completed_valid_pair':complete,'statuses':[r['status'] for r in rows],
                   'observed_seconds':[r['total_seconds'] for r in rows],
                   'iterations':[r.get('iterations') for r in rows],
                   'refresh_counts':[len(r.get('refresh_steps',[])) for r in rows]}
            entry['mean_complete_seconds']=sum(entry['observed_seconds'])/2 if complete else None
            case['methods'][method]=entry
        adaptive=case['methods']['adaptive']['mean_complete_seconds']
        fixed=case['methods']['periodic8']['mean_complete_seconds']
        static={m:case['methods'][m]['mean_complete_seconds'] for m in STATIC}
        case['adaptive_over_periodic8']=adaptive/fixed if adaptive is not None and fixed is not None else None
        if adaptive is not None and all(v is not None for v in static.values()):
            best=min(static,key=static.get)
            case['best_static_method']=best
            case['adaptive_over_best_static']=adaptive/static[best]
        else:case['adaptive_over_best_static']=None
        result['cases'].append(case)
    result['all_cases_present']=not result['missing'] and not result['invalid'] and len(result['cases'])==15
    result['fixed_method_totals']={}
    if result['all_cases_present']:
        for method in METHODS:
            entries=[c['methods'][method]['mean_complete_seconds'] for c in result['cases']]
            result['fixed_method_totals'][method]={'complete_cases':sum(v is not None for v in entries),
                'sum_of_case_mean_seconds':sum(entries) if all(v is not None for v in entries) else None}
        for metric in ['adaptive_over_periodic8','adaptive_over_best_static']:
            ratios=[c[metric] for c in result['cases']]
            result[metric]={'valid_cases':sum(v is not None for v in ratios),
                'geometric_mean_full_scope':math.exp(sum(math.log(v) for v in ratios)/15) if all(v is not None for v in ratios) else None}
    out=folder/'analysis.json';out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['sources','cases']},indent=2))
if __name__=='__main__':main()
