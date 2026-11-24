"""Batched fixed-landmark Schur RCD, with an established momentum control.

Every block uses one outer-indexed kernel query, one local Cholesky solve,
and one global residual update. No sequential block prefitting, adaptive
landmarks, full Schur matrix, or dense inverse occurs in performance runs.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import resource
import sys
from time import perf_counter, process_time

import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from nystrom_project.numerics import GaussianKernel, initialize_rpcholesky
from check_projected_balancing import pcg
from check_block_pcg_native import initialize, NativeOperator, memory_bounds
from check_block_pcg_width2 import CountedInverse


class LandmarkSchur:
    def __init__(self, kernel, b, lam, rank=16):
        self.kernel, self.b, self.lam = kernel, b, lam
        self.n, self.rank = kernel.n, rank
        base = initialize_rpcholesky(kernel, rank, lam,
            np.random.default_rng(np.random.SeedSequence([0, 6220, 1])))
        self.permutation = np.r_[base.indices, np.setdiff1d(np.arange(kernel.n), base.indices)]
        self.indices = self.permutation[:rank]
        self.remaining = self.permutation[rank:]
        core = base.columns[self.indices].copy()
        core.flat[::rank+1] += lam
        self.L = linalg.cholesky((core+core.T)*.5, lower=True, check_finite=False)
        self.B = linalg.solve_triangular(self.L, base.columns[self.remaining].T,
            lower=True, check_finite=False).T.copy()
        del base, core
        self.top_rhs = linalg.solve_triangular(self.L, b[self.indices], lower=True, check_finite=False)
        self.q = b[self.remaining]-self.B@self.top_rhs
        self.diagonal = 1.+lam-np.einsum('ij,ij->i', self.B, self.B)
        self.m = len(self.remaining)
        if np.any(self.diagonal <= 0):
            raise ArithmeticError('nonpositive Schur diagonal')

    def batch(self, J):
        columns = self.kernel.entries(self.remaining, self.remaining[J])
        columns -= self.B@self.B[J].T
        columns[J, np.arange(len(J))] += self.lam
        return columns

    def reconstruct(self, tail):
        full = np.empty(self.n)
        full[self.remaining] = tail
        full[self.indices] = linalg.solve_triangular(self.L.T,
            self.top_rhs-self.B.T@tail, lower=False, check_finite=False)
        return full

    def verify(self, tail):
        full = self.reconstruct(tail)
        residual = self.b-self.kernel.matvec(full)-self.lam*full
        return full, residual

    @property
    def coarse_bytes(self):
        return self.permutation.nbytes+self.L.nbytes+self.B.nbytes


def block_update(x, r, J, columns, previous, momentum):
    """Minimize in the current block direction, optionally plus previous step.

    The current direction is z=E_J S_JJ^-1 r_J. The second minimization
    spans only [z,p_previous], not all b coordinates plus p_previous.
    Its Gram is normalized to avoid scale-induced ill conditioning.
    """
    local = columns[J, :]
    local = (local+local.T)*.5
    factor = linalg.cho_factor(local, lower=True, check_finite=False)
    coefficients = linalg.cho_solve(factor, r[J], check_finite=False)
    image = columns@coefficients
    curvature = float(coefficients@(local@coefficients))
    numerator = float(coefficients@r[J])
    if curvature <= 0 or not np.isfinite(curvature+numerator):
        raise ArithmeticError('nonpositive coordinate-block curvature')
    if not momentum:
        x[J] += coefficients
        r -= image
        return None, dict(predicted_energy_decrease=numerator-.5*curvature,
            two_direction_rank=1, two_direction_min_eigenvalue=None)
    if momentum=='joint':
        reorthogonalized=False
        if previous is None:
            p=np.zeros_like(r);p[J]=coefficients
            ap=image
            gain=numerator-.5*curvature
            joint_rank=len(J)
        else:
            p,ap=previous
            cross=ap[J].copy()
            old_curvature=float(p@ap)
            w=linalg.cho_solve(factor,cross,check_finite=False)
            # Reuse the previous pair buffers for h=d-E_J w and Sh=Sd-Cw.
            # Their dot product avoids subtracting two nearly equal energies.
            p[J]-=w
            ap-=columns@w
            if np.linalg.norm(ap[J])>64*np.finfo(float).eps*max(np.linalg.norm(cross),1e-300):
                correction=linalg.cho_solve(factor,ap[J],check_finite=False)
                p[J]-=correction
                ap-=columns@correction
                reorthogonalized=True
            denominator=float(p@ap)
            tolerance=128*np.finfo(float).eps*max(abs(old_curvature),1.)
            if denominator < -tolerance:
                raise ArithmeticError('negative joint block/history Schur pivot')
            if denominator <= tolerance:
                beta=0.
                joint_rank=len(J)
            else:
                beta=float(p@r)/denominator
                joint_rank=len(J)+1
            gain=numerator-.5*curvature+.5*beta*beta*max(denominator,0.)
            p*=beta;p[J]+=coefficients
            ap*=beta;ap+=image
        x+=p;r-=ap
        actual_curvature=float(p@ap)
        if actual_curvature<=0:
            raise ArithmeticError('nonpositive joint block/history curvature')
        scale=np.sqrt(actual_curvature)
        p/=scale;ap/=scale
        return (p,ap),dict(predicted_energy_decrease=float(gain),
            two_direction_rank=2 if joint_rank>len(J) else 1,
            two_direction_min_eigenvalue=None,joint_subspace_rank=joint_rank,
            joint_reorthogonalized=reorthogonalized)
    scale = np.sqrt(curvature)
    normalized = coefficients/scale
    image /= scale
    if previous is None:
        p = np.zeros_like(r)
        p[J] = normalized
        ap = image
        step_scale = numerator/scale
        # The first block is still solved by its exact line minimizer.
        p *= step_scale
        ap *= step_scale
        gain = .5*step_scale*step_scale
        rank, minimum = 1, None
    else:
        p, ap = previous
        cross = float(normalized@ap[J])
        old_curvature = float(p@ap)
        h = np.array([numerator/scale, float(p@r)])
        G = np.array([[1., cross], [cross, old_curvature]])
        eigenvalues = linalg.eigvalsh(G, check_finite=False)
        minimum = float(eigenvalues[0])
        tolerance = 128*np.finfo(float).eps*max(float(eigenvalues[-1]), 1.)
        if eigenvalues[0] < -tolerance:
            raise ArithmeticError('indefinite two-direction Gram')
        if eigenvalues[0] <= tolerance:
            values, vectors = linalg.eigh(G, check_finite=False)
            keep = values > tolerance
            weights = vectors[:, keep]@((vectors[:, keep].T@h)/values[keep])
            rank = int(keep.sum())
        else:
            determinant = old_curvature-cross*cross
            weights = np.array([(old_curvature*h[0]-cross*h[1])/determinant,
                                (h[1]-cross*h[0])/determinant])
            rank = 2
        gain = float(h@weights-.5*weights@(G@weights))
        if gain < -128*np.finfo(float).eps*max(float(h@h), 1.):
            raise ArithmeticError('two-direction predicted energy increased')
        p *= weights[1]
        p[J] += weights[0]*normalized
        ap *= weights[1]
        image *= weights[0]
        ap += image
    x += p
    r -= ap
    step_curvature = float(p@ap)
    if step_curvature <= 0:
        raise ArithmeticError('nonpositive combined-step curvature')
    scale = np.sqrt(step_curvature)
    p /= scale
    ap /= scale
    return (p, ap), dict(predicted_energy_decrease=float(gain),
        two_direction_rank=rank, two_direction_min_eigenvalue=minimum)


def coordinate_solve(model, selection, momentum, *, width=16, rtol=1e-8,
                     max_blocks=50000, cpu_seconds=8., global_cpu_deadline=float('inf'),
                     audit=None):
    start, cpu_start = perf_counter(), process_time()
    x = np.zeros(model.m)
    r = model.q.copy()
    norm_b = np.linalg.norm(model.b)
    rng = np.random.default_rng(np.random.SeedSequence([0, 6220, 83]))
    previous = None
    progress, reductions = [], 0
    joint_reorthogonalizations=0
    total_gain = 0.
    maximum_selected_residual=0.
    query_hash = sha256()
    replacements = verification_calls = 0
    last_verified = -1
    true_relative = float(np.linalg.norm(r)/norm_b)
    timings = dict(selection_seconds=0., query_seconds=0., update_seconds=0., verification_seconds=0.)
    status = 'max_blocks'
    breakdown = None
    block = 0
    def verify(iteration):
        nonlocal verification_calls, last_verified, true_relative
        begin = perf_counter()
        full, residual = model.verify(x)
        verification_calls += 1
        last_verified = iteration
        true_relative = float(np.linalg.norm(residual)/norm_b)
        progress.append(dict(block=iteration, true_relative_residual=true_relative,
            recursive_relative_residual=float(np.linalg.norm(r)/norm_b),
            recursive_gap=float(np.linalg.norm(residual[model.remaining]-r)/norm_b)))
        timings['verification_seconds'] += perf_counter()-begin
        return full, residual[model.remaining].copy()
    for block in range(1, max_blocks+1):
        if block % 25 == 1 and (process_time()-cpu_start >= cpu_seconds or process_time() >= global_cpu_deadline):
            status = 'cpu_screen_cap'; block -= 1; break
        begin = perf_counter()
        if selection == 'uniform':
            J = np.sort(rng.choice(model.m, size=min(width, model.m), replace=False))
        elif selection == 'greedy':
            scores = r*r/model.diagonal
            J = np.sort(np.argpartition(scores, -min(width, model.m))[-min(width, model.m):])
            del scores
        else:
            raise ValueError(selection)
        timings['selection_seconds'] += perf_counter()-begin
        query_hash.update(model.remaining[J].tobytes())
        begin = perf_counter()
        columns = model.batch(J)
        timings['query_seconds'] += perf_counter()-begin
        if audit is not None:
            audit('before', block, x, r, J, columns, previous)
        begin = perf_counter()
        try:
            previous, step = block_update(x, r, J, columns, previous, momentum)
        except (ArithmeticError, np.linalg.LinAlgError) as error:
            breakdown = str(error)
            status = 'numerical_failure'
            del columns
            break
        timings['update_seconds'] += perf_counter()-begin
        total_gain += step['predicted_energy_decrease']
        joint_reorthogonalizations+=int(step.get('joint_reorthogonalized',False))
        maximum_selected_residual=max(maximum_selected_residual,float(np.linalg.norm(r[J])/norm_b))
        if momentum and block > 1 and step['two_direction_rank'] < 2:
            reductions += 1
        if audit is not None:
            audit('after', block, x, r, J, columns, previous)
        # A verification never retains the queried n_tail-by-width columns.
        del columns
        suspected = np.linalg.norm(r) <= rtol*norm_b
        if block % 200 == 0 or suspected or block == max_blocks:
            verified_full, true_r = verify(block)
            del verified_full
            if true_relative <= rtol:
                status = 'converged'; break
            if suspected:
                r = true_r
                previous = None
                replacements += 1
            del true_r
    if last_verified != block:
        full, _ = verify(block)
    else:
        full = model.reconstruct(x)
    return full, dict(status=status, converged=true_relative <= rtol,
        true_relative_residual=true_relative, blocks=block, breakdown=breakdown,
        solve_seconds=perf_counter()-start, solve_cpu_seconds=process_time()-cpu_start,
        verification_matvecs=verification_calls, residual_replacements=replacements,
        dependent_two_direction_steps=reductions, predicted_energy_decrease_sum=total_gain,
        joint_reorthogonalizations=joint_reorthogonalizations,
        coordinate_trace_sha256=query_hash.hexdigest(), selection=selection, momentum=momentum,
        history_kind='joint_block_history' if momentum=='joint' else ('two_direction' if momentum else 'none'),
        max_selected_block_residual_after_update=maximum_selected_residual,
        unchanged_iterate_landmark_swap_constraints_preserved=momentum is not True,
        width=width, retained_displacement_pairs=int(bool(momentum)), query_columns_released_before_verification=True,
        timings=timings, progress=progress)


class DenseSchur:
    """Independent tiny SPD validation only."""
    def __init__(self, A, b, indices):
        self.A, self.b, self.n = A, b, len(b)
        self.indices = indices
        self.remaining = np.setdiff1d(np.arange(len(b)), indices)
        self.m = len(self.remaining)
        self.A11 = A[np.ix_(indices, indices)]
        self.A12 = A[np.ix_(indices, self.remaining)]
        self.S = A[np.ix_(self.remaining, self.remaining)]-self.A12.T@linalg.solve(self.A11, self.A12, assume_a='pos')
        self.q = b[self.remaining]-self.A12.T@linalg.solve(self.A11, b[indices], assume_a='pos')
        self.diagonal = np.diag(self.S).copy()
    def batch(self, J):
        return self.S[:, J].copy()
    def reconstruct(self, tail):
        x = np.empty(self.n)
        x[self.remaining] = tail
        x[self.indices] = linalg.solve(self.A11, self.b[self.indices]-self.A12@tail, assume_a='pos')
        return x
    def verify(self, tail):
        full = self.reconstruct(tail)
        return full, self.b-self.A@full


def small_checks():
    rng = np.random.default_rng(622085)
    n = 29
    F = rng.normal(size=(n, 11))
    A = F@F.T+1.5*np.eye(n)
    b = rng.normal(size=n)
    model = DenseSchur(A, b, np.array([7, 2, 18, 4, 23]))
    exact = linalg.solve(A, b, assume_a='pos')
    target = exact[model.remaining]
    checks = []
    for selection in ['uniform','greedy']:
        for momentum in [False,True,'joint']:
            saved = {}
            errors = dict(max_direct_galerkin_relative_error=0., max_recursive_schur_residual_error=0.,
                max_energy_increase_relative=0., max_original_vs_schur_residual_error=0.,
                max_selected_coordinate_residual=0., max_previous_displacement_residual=0.,
                max_direct_galerkin_scaled_absolute_error=0.,early_direct_galerkin_relative_error=0.)
            def audit(stage, block, x, r, J, C, previous):
                if stage == 'before':
                    old_x = x.copy()
                    a = linalg.solve(model.S[np.ix_(J,J)], r[J], assume_a='pos')
                    z = np.zeros(model.m); z[J] = a
                    if momentum=='joint':
                        E=np.eye(model.m)[:,J]
                        D=E if previous is None else np.column_stack((E,previous[0]))
                    else:
                        D = z[:,None] if previous is None or not momentum else np.column_stack((z, previous[0]))
                    # Column scaling preserves the subspace but prevents the
                    # independent reference's rank threshold from discarding
                    # a tiny current correction beside a normalized old step.
                    D /= np.linalg.norm(D, axis=0)
                    G = D.T@model.S@D
                    weights = linalg.pinvh(G, rtol=1e-13)@(D.T@r)
                    saved.update(x=old_x, expected=D@weights,
                        old_p=None if previous is None else previous[0].copy(),
                        energy=.5*float((old_x-target)@model.S@(old_x-target)))
                else:
                    change=x-saved['x']
                    errors['max_direct_galerkin_relative_error']=max(errors['max_direct_galerkin_relative_error'],
                        float(np.linalg.norm(change-saved['expected'])/max(np.linalg.norm(saved['expected']),1e-10)))
                    errors['max_direct_galerkin_scaled_absolute_error']=max(errors['max_direct_galerkin_scaled_absolute_error'],
                        float(np.linalg.norm(change-saved['expected'])/max(np.linalg.norm(saved['expected']),1.)))
                    if np.linalg.norm(saved['expected'])>1e-6:
                        errors['early_direct_galerkin_relative_error']=max(errors['early_direct_galerkin_relative_error'],
                            float(np.linalg.norm(change-saved['expected'])/np.linalg.norm(saved['expected'])))
                    errors['max_recursive_schur_residual_error']=max(errors['max_recursive_schur_residual_error'],
                        float(np.linalg.norm(model.q-model.S@x-r)/np.linalg.norm(b)))
                    energy=.5*float((x-target)@model.S@(x-target))
                    errors['max_energy_increase_relative']=max(errors['max_energy_increase_relative'],
                        (energy-saved['energy'])/max(saved['energy'],1e-20))
                    _, residual=model.verify(x)
                    errors['max_original_vs_schur_residual_error']=max(errors['max_original_vs_schur_residual_error'],
                        float(np.linalg.norm(residual[model.remaining]-r)/np.linalg.norm(b)))
                    errors['max_selected_coordinate_residual']=max(errors['max_selected_coordinate_residual'],
                        float(np.linalg.norm(r[J])/np.linalg.norm(b)))
                    if saved['old_p'] is not None:
                        old_p=saved['old_p']
                        errors['max_previous_displacement_residual']=max(errors['max_previous_displacement_residual'],
                            abs(float(old_p@r))/(np.linalg.norm(old_p)*np.linalg.norm(b)))
            full, run=coordinate_solve(model, selection, momentum, width=6, rtol=1e-10,
                max_blocks=3000, cpu_seconds=5., audit=audit)
            solution_error=float(np.linalg.norm(full-exact)/np.linalg.norm(exact))
            assert run['converged'],run
            assert solution_error<1e-9,(run,solution_error)
            assert errors['max_direct_galerkin_relative_error']<2e-5,errors
            assert errors['max_direct_galerkin_scaled_absolute_error']<1e-11,errors
            assert errors['max_recursive_schur_residual_error']<1e-11,errors
            assert errors['max_original_vs_schur_residual_error']<1e-11,errors
            assert errors['max_energy_increase_relative']<1e-6,errors
            if momentum=='joint':
                assert errors['max_selected_coordinate_residual']<1e-11,errors
                assert errors['max_previous_displacement_residual']<1e-11,errors
            checks.append(dict(selection=selection,momentum=momentum,blocks=run['blocks'],
                solution_relative_error=solution_error,**errors))
    # Native Schur columns and recovery, checked against independent dense A.
    from scipy.spatial.distance import cdist
    X=rng.normal(size=(31,4));width=.9;lam=.03
    kernel=GaussianKernel(X,width,block_size=7)
    K=np.exp(-cdist(X,X,'sqeuclidean')/(2*width*width))
    matrix=K+lam*np.eye(len(X));rhs=rng.normal(size=len(X))
    native=LandmarkSchur(kernel,rhs,lam,rank=5)
    independent=DenseSchur(matrix,rhs,native.indices)
    J=np.array([1,7,13,20])
    column_error=float(np.linalg.norm(native.batch(J)-independent.S[:,J])/np.linalg.norm(independent.S[:,J]))
    tail=rng.normal(size=native.m)
    observed,residual=native.verify(tail)
    exact_full=independent.reconstruct(tail)
    recovery_error=float(np.linalg.norm(observed-exact_full)/np.linalg.norm(exact_full))
    assert column_error<1e-12 and recovery_error<1e-12
    return dict(spd_iterations=checks,native_schur_batch_relative_error=column_error,
        native_original_recovery_relative_error=recovery_error,
        native_full_residual_relative_error=float(np.linalg.norm(residual-(rhs-matrix@observed))/np.linalg.norm(rhs)))


def resource_views(n,k=16,width=16,momentum=True):
    m=n-k
    coarse=8*(k*k+m*k+n)
    history=2*m*8 if momentum else 0
    # b, q, diagonal, x, r, score/image, verified residual/reconstruction
    # allowances are explicit; the exact factor payload is separate.
    persistent_solver=8*(n+4*m+k)+history
    active_work=8*(2*m*width+6*m+8*width*width+4*k*width)
    coarse_history_column_allowance=coarse+persistent_solver+active_work
    strong=max(rank for rank in range(1,min(n,96)+1)
        if memory_bounds(n,rank,False)['factor_plus_live_array_bound_bytes']<=coarse_history_column_allowance)
    return dict(exact_coarse_L_B_permutation_bytes=coarse,history_pair_bytes=history,
        solver_persistent_array_allowance_bytes=persistent_solver,
        batch_column_bytes=m*width*8,batch_coarse_product_temporary_bytes=m*width*8,
        active_block_workspace_allowance_bytes=active_work,
        coarse_history_column_live_allowance_bytes=coarse_history_column_allowance,
        scalar_strong_reference_rank=strong,
        full_verification_kernel_tile_bytes=min(128,n)*n*8,
        query_column_kernel_array_bytes=m*width*8,
        joint_projected_history_buffers='h and Sh overwrite the existing previous-displacement/image pair; C@w uses one m-vector temporary included in active workspace',
        verification_extra_full_vector_allowance_bytes=4*n*8,
        resource_view_note='The strong scalar rank matches this conservative solver/column allowance only. Kernel tiles, factorization setup, borrowed input features, and process/library work are separately reported; this is not exact total peak matching.')


def forbidden_dense():
    raise AssertionError('dense kernel forbidden in performance solve')


class CpuBudgetReached(RuntimeError):
    pass


class BudgetOperator(NativeOperator):
    def __init__(self, kernel, lam, deadline):
        super().__init__(kernel, lam)
        self.deadline=deadline
    def __matmul__(self, vector):
        if process_time()>=self.deadline:
            raise CpuBudgetReached('global CPU guard during scalar control')
        return super().__matmul__(vector)


def scalar_control(X,b,bw,lam,rank,depth,deadline):
    wall,cpu=perf_counter(),process_time()
    kernel=GaussianKernel(X,bw);kernel.dense=forbidden_dense
    state=initialize(kernel,rank,depth,lam)
    setup=perf_counter()-wall
    op=BudgetOperator(kernel,lam,deadline);inv=CountedInverse(state)
    try:
        run,_=pcg(op,b,inv,maxiter=2500)
        run['status']='converged' if run['converged'] else 'iteration_cap'
    except CpuBudgetReached:
        run=dict(status='global_cpu_cap',converged=False,iterations=None,
            true_relative_residual=None,breakdown=None,
            note='Interrupted scalar control has no final residual measurement; no convergence claim.')
    run.update(method='scalar_power_pcg',rank=rank,power_passes=depth,
        allocation='same_rank16' if rank==16 else 'larger_strong_control',
        total_seconds=perf_counter()-wall,total_cpu_seconds=process_time()-cpu,
        setup_seconds=setup,factor_state_bytes=state.storage_bytes,
        solver_array_allowance=memory_bounds(len(b),rank,False),
        kernel_tile_bytes=min(128,len(b))*len(b)*8,kernel_stats=dict(kernel.stats),
        full_operator_calls=op.calls,inverse_rhs=inv.rhs,
        setup_factorization_array_bound_bytes=8*(10*len(b)*rank+12*rank*rank+8*rank),
        kernel_X_copy_bytes=kernel.X.nbytes)
    return run


def main():
    output=ROOT/'results/batched-schur-rcd-diagnostic.json'
    if output.exists():raise FileExistsError(output)
    started,cpu_start=perf_counter(),process_time()
    resource.setrlimit(resource.RLIMIT_CPU,(119,120))
    record=dict(status='running',scope='Single-repeat matrix-free headroom screen of established fixed-landmark block corrections; no adaptive landmarks or novelty claim',
        protocol=dict(datasets=['synthetic-n640-smooth','concrete-observed'],regularizations=[.001,1e-5],seed=0,
            landmark_rank=16,block_size=16,history_pairs=[0,1],history_kind=['none','two_direction','joint_block_history'],selection=['uniform','greedy'],
            uniform='without replacement inside each block, fresh block draws',
            greedy='one-shot top current residual squared / original Schur diagonal; no sequential prefitting',
            momentum='exact energy minimization in current block correction and previous combined displacement; normalized2x2 Gram, numerical dependence truncation only',
            joint_history='project previous pair against coordinate block with existing Cholesky, optional local reorthogonalization, denominator h^T Sh; exact minimization in all16 coordinates plus previous displacement, no extra kernel query',
            feasibility='two_direction baseline generally violates selected residual zeros and cannot support unchanged-iterate landmark swaps; joint_history enforces selected residual and old-displacement orthogonality',
            verification_interval=200,rtol=1e-8,max_blocks=50000,max_pcg_iterations=2500,
            coordinate_cpu_screen='min(8,max(1,2*fastest converged stronger scalar CPU seconds)) per variant, after setup; timeouts are censored cost screens',
            blas_threads=1,global_cpu_cap_seconds=120,kernel_block_size=128,
            scalar_controls='power2/4/8 at rank16 and larger rank under explicit coarse/history/column-work allowance, not claimed exact total-peak match',
            execution_order='per case: larger scalar controls, coordinate variants, rank16 scalar controls; expensive same-rank baselines cannot prevent the coordinate comparison on the current case',
            landmark_seed_sequence=[0,6220,1],coordinate_seed_sequence=[0,6220,83],power_seed_sequence=[0,6220,3],
            performance_dense_kernel=False,substantial_cpu_lock='/tmp/cs6220-project-benchmark.lock'),
        primary_source='https://arxiv.org/pdf/2506.09394',
        source_sha256={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in [Path(__file__).resolve(),
            ROOT/'scripts/check_projected_balancing.py', ROOT/'scripts/check_block_pcg_native.py',
            ROOT/'scripts/check_block_pcg_width2.py', ROOT/'src/nystrom_project/numerics.py',
            ROOT/'src/nystrom_project/preconditioners.py']},cases=[])
    def save():
        record.update(elapsed_wall_seconds=perf_counter()-started,elapsed_cpu_seconds=process_time()-cpu_start)
        output.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    with threadpool_limits(limits=1):
        record['independent_small_checks']=small_checks();save()
        # Lower-cost regularization first gives both datasets coverage before
        # expensive small-regularization controls can consume the budget.
        for lam in [.001,1e-5]:
            for name in ['synthetic-n640-smooth','concrete-observed']:
                if process_time()-cpu_start>110:record['status']='cpu_budget';save();return
                data_path=ROOT/f'data/prepared/{name}-s0-lambda0.001.npz'
                data=np.load(data_path);X,b=data['X_train'],data['y_train'];bw=float(data['bandwidth'])
                n=len(b);views=resource_views(n);strong_rank=views['scalar_strong_reference_rank']
                case=dict(dataset=name,regularization=lam,n=n,data_sha256=sha256(data_path.read_bytes()).hexdigest(),
                    resource_views=views,shared_original_X_bytes=X.nbytes,shared_original_b_bytes=b.nbytes,
                    controls=[],coordinates=[])
                record['cases'].append(case)
                # Strong controls first determine the predeclared cost screen.
                for rank in [strong_rank]:
                    for depth in [2,4,8]:
                        if process_time()-cpu_start>108:record['status']='cpu_budget';save();return
                        run=scalar_control(X,b,bw,lam,rank,depth,cpu_start+117.)
                        case['controls'].append(run);save()
                        print(json.dumps(dict(dataset=name,lam=lam,method='scalar',rank=rank,depth=depth,
                            seconds=run['total_seconds'],iterations=run['iterations'],converged=run['converged'])),flush=True)
                        if run['status']=='global_cpu_cap':record['status']='cpu_budget';save();return
                strong=[v['total_cpu_seconds'] for v in case['controls'] if v['rank']==strong_rank and v['converged']]
                cap=min(8.,max(1.,2*min(strong))) if strong else 8.
                case['coordinate_cpu_screen_seconds']=cap
                for selection,momentum in [('uniform',False),('greedy',False),('uniform',True),('greedy',True),('uniform','joint'),('greedy','joint')]:
                    if process_time()-cpu_start>116:record['status']='cpu_budget';save();return
                    wall,cpu=perf_counter(),process_time()
                    kernel=GaussianKernel(X,bw);kernel.dense=forbidden_dense
                    model=LandmarkSchur(kernel,b,lam)
                    setup=perf_counter()-wall
                    full,run=coordinate_solve(model,selection,momentum,cpu_seconds=cap,
                        global_cpu_deadline=cpu_start+117.)
                    stats=dict(kernel.stats)
                    run.update(method='batched_schur_rcd',total_seconds=perf_counter()-wall,
                        total_cpu_seconds=process_time()-cpu,setup_seconds=setup,
                        kernel_stats=stats,landmark_indices=model.indices.tolist(),
                        coarse_state_bytes=model.coarse_bytes,resource_views=resource_views(n,momentum=momentum),
                        kernel_X_copy_bytes=kernel.X.nbytes,
                        kernel_query_feature_copy_bytes=(model.m+16)*X.shape[1]*8,
                        setup_rp_columns_and_factor_array_bound_bytes=8*(8*n*16+12*16*16),
                        exact_entry_equivalent_full_passes=(stats.get('entry_evaluations',0)+n*stats['column_evaluations']+n*n*stats['matvecs'])/(n*n),
                        coordinate_cpu_screen_seconds=cap)
                    case['coordinates'].append(run);save()
                    print(json.dumps(dict(dataset=name,lam=lam,selection=selection,momentum=momentum,
                        blocks=run['blocks'],seconds=run['total_seconds'],converged=run['converged'],
                        relative_residual=run['true_relative_residual'],status=run['status'])),flush=True)
                    del model,kernel,full
                for depth in [2,4,8]:
                    if process_time()-cpu_start>108:record['status']='cpu_budget';save();return
                    run=scalar_control(X,b,bw,lam,16,depth,cpu_start+117.)
                    case['controls'].append(run);save()
                    print(json.dumps(dict(dataset=name,lam=lam,method='scalar',rank=16,depth=depth,
                        seconds=run['total_seconds'],iterations=run['iterations'],converged=run['converged'])),flush=True)
                    if run['status']=='global_cpu_cap':record['status']='cpu_budget';save();return
        record['status']='complete';save()


if __name__=='__main__':main()
