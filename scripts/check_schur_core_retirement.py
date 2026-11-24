"""Exact-factor headroom for retiring a nested principal core, no timing claim."""
import json
import resource
from pathlib import Path
from hashlib import sha256
from time import perf_counter, process_time
import numpy as np
from threadpoolctl import threadpool_limits
import check_pair_rotation_schur as original
from check_schur_inplace_frontier_v2 import SharedKernel, Budget, ROOT


class Coordinates:
    def __init__(self, n, selected):
        self.m = len(selected)
        self.permutation = np.r_[selected, np.setdiff1d(np.arange(n), selected)]

    def to_rotated(self, v):
        return v[self.permutation].copy()

    def to_original(self, v):
        result = np.empty_like(v)
        result[self.permutation] = v
        return result

    def matrix(self, A):
        return A[np.ix_(self.permutation, self.permutation)]


def main():
    output = ROOT/'results/schur-core-retirement.json'
    if output.exists():
        raise FileExistsError(output)
    resource.setrlimit(resource.RLIMIT_CPU, (55,60))
    start, cpu = perf_counter(), process_time()
    paths = [Path(__file__).resolve(), Path(original.__file__).resolve(),
             Path(original.previous.__file__).resolve(), ROOT/'scripts/check_schur_inplace_frontier_v2.py']
    record = dict(status='running', sources={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in paths},
        protocol=dict(large_core=592, small_core=416, checkpoint=10, seed=0, regularization=1e-5,
            mechanism='Keep full x at checkpoint, release last176 principal coordinates, resume exact Schur CG in enlarged tail.',
            question='Can smaller-core applications become cheap enough to offset the changed remaining iteration count?',
            scope='Exact dense factors including infeasible592 FP64 are free headroom inputs. No native memory/time or adaptive selection claim. Fixed checkpoint is a diagnostic, not a justified decision rule.',
            stopping='Literal original residual1e-8 at every iterate, same repaired pair.solve for all controls.',
            cost='Report required small/large per-action cost ratio ignoring transition cost. Kernel query counts alone exclude cache/tile benefits.',
            wall_cap=90,cpu_cap=55),cases=[],failures=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-start,elapsed_cpu_seconds=process_time()-cpu)
        tmp=output.with_suffix('.json.tmp');tmp.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');tmp.replace(output)
    save()
    cases=[('smooth','synthetic-n2560-smooth-s0-lambda0.001.npz'),
           ('mixed','synthetic-n2560-mixed-s0-lambda0.001.npz'),
           ('Concrete','concrete-observed-s0-lambda1e-05.npz'),
           ('Airfoil','airfoil-observed-s0-lambda1e-05.npz')]
    with threadpool_limits(limits=1):
        try:
            for name,filename in cases:
                path=ROOT/'data/prepared'/filename
                with np.load(path) as data:
                    X,b,bw=data['X_train'],data['y_train'],float(data['bandwidth'])
                kernel=SharedKernel(X,bw,Budget([X,b]));J=kernel.farthest(592)
                A=kernel.block(X,X);A.flat[::len(b)+1]+=1e-5
                large=Coordinates(len(b),J);small=Coordinates(len(b),J[:416])
                AL,bL=large.matrix(A),large.to_rotated(b)
                opL=original.previous.build(AL,bL,np.arange(592))
                base,saved,_=original.solve(A,b,large,AL,bL,opL,capture_prefix=True,deadline=cpu+50)
                case=dict(dataset=name,data_sha256=sha256(path.read_bytes()).hexdigest(),large_upfront=base)
                record['cases'].append(case);save()
                if 10 not in saved:
                    case['status']='checkpoint_not_reached';save();continue
                x=saved[10]
                restart,_,_=original.solve(A,b,large,AL,bL,opL,x0=x,deadline=cpu+50)
                AS,bS=small.matrix(A),small.to_rotated(b)
                opS=original.previous.build(AS,bS,np.arange(416))
                static,_,_=original.solve(A,b,small,AS,bS,opS,deadline=cpu+50)
                retired,_,_=original.solve(A,b,small,AS,bS,opS,x0=x,deadline=cpu+50)
                required=(base['iterations']-10)/retired['iterations'] if retired['iterations'] else None
                case.update(status='complete',large_restart=restart,small_upfront=static,retired=retired,
                    retirement_total_iterations=10+retired['iterations'],
                    maximum_small_to_large_action_cost_ratio_for_win_ignoring_transition=required,
                    exact_entry_ratio_small_to_large=(len(b)**2-416**2)/(len(b)**2-592**2))
                save()
                print(json.dumps(dict(dataset=name,large=base['iterations'],small=static['iterations'],
                    retired_total=10+retired['iterations'],required_action_ratio=required)),flush=True)
        except Exception as error:
            record['status']='failed';record['failures'].append(repr(error));save();raise
    record['status']='complete';save()


if __name__=='__main__':
    main()
