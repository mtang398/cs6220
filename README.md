# Updating Nyström Preconditioners with Solver Residuals

CS 6220 research project. The submitted proposal is preserved in `proposal/`.

On 2026-11-24, the user authorized changing the representation under a fixed
total memory budget, including cores, factors, history and workspace. The
current method revision therefore allows the effective rank to change. It
still requires useful information from the current solve and reproducible
complete-cost gains; a stronger static representation alone is not success.

**Current research question:** Can information from the current linear solve
guide the construction and application of a kernel preconditioner so that the
total solution time decreases under a fixed memory budget? The budget includes
setup and application workspaces as well as retained factors and solver history.

The submitted proposal asked this question for a fixed number of actual kernel
columns. Its implementation and negative results remain preserved below.

The initial proposal solves `(K + lambda I) alpha = y` for a Gaussian kernel.
Randomly pivoted
Cholesky selects the initial columns. At a PCG checkpoint, sample candidate columns
using squared entries of the current residual, evaluate every possible removal,
and accept a replacement only when the residual quadratic form with the inverse
Nyström preconditioner decreases. Accepted updates restart PCG from its current
iterate. Rejected updates keep the existing recurrence.

The first experiments ask whether these exchanges improve subsequent error
reduction and whether that improvement pays for column selection, factorization,
and restarting. All solver operations use CPUs.

## Current results

The research goal remains active. A useful advantage from current-solve
adaptation has not yet been established. A compressed static representation
now gives an end-to-end improvement on two difficult development cases.
The production code passes 45 tests and 29 subtests; the newer diagnostic
solvers have separate numerical and allocation checks. [The revision record](docs/method-revision.json) tracks the
execution allowance, decisions, and evidence.

The [compressed-core comparison](results/compressed-core-schur-odd-pilot.json)
uses a packed FP32 factor for a 600-coordinate principal block, with FP64
triangular arithmetic and regenerated exact core products for refinement.
Three refinement corrections per Schur action and four for constructing the
reduced right-hand side and reconstructing the solution pass all eight runs.
At regularization 1e-5, the two-repeat mean total time is 1.94 seconds for
smooth targets and 1.82 seconds for mixed targets, respectively 1.42 and
1.43 times faster than the faster of two contemporaneous exact
416-coordinate controls. At .001 the mean improvement is only about 1–2%.
The candidate's full traced allocation peak is 989,720 bytes; named arrays
plus the 65,536-byte reserve total 1,043,544 bytes. Both fit 1 MiB, with
immutable inputs reported separately. These are static improvements on seed0
at n=2048, not evidence of adaptive benefit or full-scope validation.

The [coupling-sensitive error diagnostic](results/core-coupling-certificate.json)
uses the interaction between core and remaining coordinates to decide whether
one correction can replace three. On frozen exact trajectories, it admits
five of 39 and eight of 36 steps in the difficult smooth and mixed cases,
where the previous bound admitted none. These calculations use dense matrices
and the future trajectory length. The subsequent
[native adaptive comparison](results/compressed-core-adaptive-pilot.json)
charges six probes and uses an online accumulated-error budget without a
future iteration count. All four adaptive runs meet the original 1e-8
residual threshold and the full traced allocation peak is 992,264 bytes.
However, they take 1.150/2.111/1.153/2.000 seconds on smooth-soft,
smooth-hard, mixed-soft and mixed-hard cases, versus
1.007/1.872/1.017/1.782 seconds for the fastest accurate fixed-order control.
Probes cost .066–.068 seconds; the hard adaptive cases also require one
extra outer iteration. A common .5e-8 recursive threshold makes fixed two
corrections accurate on both hard cases. This rules out the tested adaptive
selector as an improvement over those controls; it does not erase the
static compression benefit.

The [mixed-column precision diagnostic](results/core-precision-allocation.json)
then rebuilt 592-coordinate factors with eight FP64 columns and FP32 storage
elsewhere. Four earlier core solves determine the candidate precision mask;
later frozen-trajectory directions evaluate it. On smooth, mixed, Concrete
and Airfoil cases at regularization 1e-5, its median action-error ratios to
matrix-only allocation are .668, 1.402, .507 and .406. One correction with
the selected factor therefore has measurable accuracy headroom in three
cases, but does not uniformly match two corrections with an FP32 factor.
Construction and mixed solves pass independent checks, including bitwise
agreement with the original all-FP32 arithmetic. The diagnostic uses dense
references and free exact trajectories; its factor storage excludes metadata
and selection workspace. The [actual-trajectory comparison](results/core-precision-trajectories.json)
then tests the first four observed H2 actions followed by a full factor rebuild
and H1 actions, without restarting. Storing the first eight columns in FP64
from the start passes both synthetic cases with less work than current-solve
selection. On Concrete, current selection reaches an original relative residual
of 8.14e-9, while FIRST8, matrix-only and initial-RHS H1 masks fail the 1e-8
threshold. Its analytically counted kernel entries are 18.22 million, versus
22.17 million for uniform600 H2, including factor/rebuild, boundary and
verification queries. These counts omit native arithmetic cost and do not
establish elapsed-time benefit. All six Airfoil variants fail with tail
residuals around 1.77e-7–3.56e-7 despite landmark residuals near 3.4e-10.
The [Airfoil error decomposition](results/airfoil-core-error-diagnosis.json)
identifies approximate core-action bias: exact-core and preformed exact-Schur
controls converge, while the difference between action and reconstruction
operators explains the large tail residual. Initial boundary-RHS error and
fixed-operator recursive drift are much smaller. The [common-correction comparison](results/core-precision-outer-correction.json)
then applies full original-residual correction equally to every method; all
24 cases converge with at most one additional correction cycle. A static
FIRST8 factor with the same four-step H2 prefix matches the adaptive method's
action/core-product counts on smooth, mixed and Concrete, and improves on it
on Airfoil (23/43 versus 28/48). It also avoids the factor rebuild. The earlier
Concrete advantage therefore does not establish a benefit from learning the
precision mask: the extra early action accuracy explains it. This closes the
tested solve-dependent precision-allocation route. Static compression,
precision scheduling and ordinary refinement remain useful controls;
the adaptive research goal remains unachieved.

The [native static comparison](results/native-first8-schur.json) validates
streamed FIRST8 factors and full original-residual correction under the 1 MiB
working allowance. All 16 attempts converge. Single-run times for exact416,
FIRST8 H1, FIRST8 with the four-step H2 prefix, and uniform600 H2 are,
respectively: 2.903/2.037/2.068/2.042 seconds on smooth; 2.557/1.820/1.816/1.823
on mixed; .198/.262/.230/.239 on Concrete; and .320/.480/.432/.534 on Airfoil.
Maximum whole-solve traced allocation is 1,001,194 bytes; named arrays plus
reserve peak at 1,044,288 bytes. Mixed factorization takes about .040 seconds,
versus .019 seconds for uniform600. Thus compressed cores improve the larger
synthetic systems, while exact416 remains the stronger measured real-data
control. These are static, single-repeat development measurements.

A later [FIRST8 implementation check](results/first8-specialization.json)
removes generic column-type/offset lookups from the numerical inner loops.
Factor, pivot, local-error and solution bits match the earlier implementation.
Across five cyclic primitive measurements at each of two shifts, factorization
is 2.42–2.56 times faster and scalar triangular solves are 1.48–1.51 times
faster. These measurements exclude kernel construction and full solver work;
they do not revise the end-to-end timings above. The [block-action component gate](results/schur-block-action-cost.json)
uses that same helper for scalar and block actions. Under the declared full
block-workspace allowance, n2048 permits a 552-coordinate block core versus
592 for scalar; both real-data block cores retain 592 coordinates. Width-two
actions cost 1.051, 1.108 and 1.131 times scalar592 on smooth2048, Concrete and
Airfoil. The maximum traced allocation across the component phases is
1,003,158 bytes. Setup, QR/deflation and verification phases are audited, but
full block convergence and end-to-end timing remain untested. The next gate
compares standard fixed geometric-partition and random-auxiliary block methods
against the strongest native scalar controls.

The [larger-system CPU screen](results/larger-system-static-scope.json) adds
2048-training-point smooth, random and mixed systems using the same generator.
It preserves all earlier cases. At the 1 MiB working allowance, 10 of 18 attempts
converged, six 448-coordinate cores left no kernel row tile, and two plain-CG
runs hit the CPU cap. The feasible 384-coordinate core spends only 1.23–4.08%
of total time on setup (2.07% in aggregate). Skipping construction therefore
offers little savings; repeated kernel applications dominate. These are single
timing observations and static controls, not adaptive performance evidence.

Two kernel-product screens found a measurable application-cost opportunity.
[FP32 generation with FP64 accumulation](results/kernel-mixed-accumulation.json)
is 1.74–1.98 times faster than the fastest matched FP64 generator in these
standalone measurements, including tile conversion within the same 128 KiB
tile allowance. On reference solutions at regularization .001, its product
error is roughly 0.00008–0.00042 times the right-hand-side norm. It therefore
cannot replace exact final verification. Subsequent checks rejected this fast
Gram-based generator because its rounding depends on query shape. A symmetric,
block-consistent alternative computes distances in FP64 and exponentials in
FP32, retaining a 1.40–1.53-fold product speedup. These
microbenchmarks alone establish neither full-solver memory feasibility nor speedup;
single-precision inner Gaussian products with double-precision outer products
already have direct prior art recorded in the literature notes.

The [native iterative-refinement pilot](results/mixed-precision-refinement-pilot.json)
then tested 24 runs on 2048-point smooth and mixed systems under a 1 MiB
algorithm-owned working allowance. Eighteen converged and six reached their
CPU limit. The adaptive method took 1.55–1.56 seconds on the easier systems,
versus 1.12–1.22 seconds for the exact 416-coordinate control. Both harder
adaptive runs reached the CPU limit; the exact control finished in 2.66–2.88
seconds. Fixed inner tolerances also beat the adaptive rule. Repeated inner
iterations consumed the cheaper-product savings. Source, operator-count and
allocation checks found no material accounting error: candidate peak including
reserve was 1,041,344 bytes and the exact control was 1,043,824 bytes; immutable
input arrays are reported separately. These single-repeat development results
reject this candidate, without establishing a general limitation of mixed
precision. A compiled Taylor approximation has now passed the standalone cost screen
described next; an adaptive solver using it has not been demonstrated.

The [compiled Taylor cost screen](results/taylor-kernel-cost.json) evaluates
Gaussian kernel expansions using global degrees and two charged FP64 tiles.
Degrees 8–17 satisfy the predeclared trace or diagonal-corrected row error bound
at shifts .001 and .00001. Across the four tested sizes, these selected actions
are 1.06–1.72 times faster than the fastest matched FP64 Gaussian action; the
smallest advantage is on Airfoil. Five cyclic repetitions measure standalone
application time, with extra retained context reported separately. This is
neither a full-memory solver comparison nor an adaptive performance result.
The [numerical audit](results/taylor-kernel-accuracy.json) finds symmetric
matrices and successful Cholesky at shift .00001 for the tested degrees. The
analytic directionwise error bound exceeds the observed error by factors
6–710 on the prepared reference vectors. Its conservatism may prevent useful
degree reduction; a complete adaptive-versus-fixed comparison remains needed.

The [current-correction degree oracle](results/taylor-degree-oracle-v2.json)
subsequently evaluated 103 runs across smooth512, mixed512 and Concrete824 at
two shifts. All selection probes, dense factors and exact inner solves were
free. Even under these favorable conditions, a fixed degree chosen from the
global error bound needed fewer corrections and a smaller cumulative degree
than certificate-based adaptation in all six cases. Adaptive corrections were
4/3/4/3/4/4 versus fixed 3/2/2/2/3/3, respectively. A selector using the actual
error instead of its bound also required more corrections. This rejects native
integration of the tested repeated-correction design. It does not measure
inner-CG work or prove that every variable-accuracy method is slower. Version 2
includes measured inner residuals in the acceptance test; the original record
is preserved, and all selected degree sequences remained unchanged.

A [direct Krylov trajectory replay](results/taylor-krylov-cost-oracle.json)
then removes the nested inverse solves: it asks which Taylor degree would meet
a cumulative residual-gap allowance along exact PCG directions. In six cases,
free access to exact step lengths and future iteration counts gives 1.06–1.26x
action speedup over exact Gaussian products, and 1.07–1.13x over the least common
fixed degree satisfying the same trajectory budget. These are weighted
standalone action timings, not observed inexact-PCG runtimes. The next diagnostic
must remove this future knowledge and check actual convergence. The mechanism
is established inexact CG with a fixed preconditioner; it adapts operator
accuracy and does not yet establish a new preconditioning method.

The [actual inexact-CG diagnostic](results/taylor-inexact-cg-diagnostic.json)
completed 24 runs with final original-system residuals below 1e-8. It removes
future iteration-count and step-length knowledge. The adaptive mean degrees
are approximately 15.2/16.1, 15.4/16.0 and 19.9/21.6, compared with fixed degrees
15/16, 15/16 and 20/22. Thus the degree advantage largely disappears. Fixed
controls use a stricter recursive stopping threshold, so slightly lower
adaptive iteration counts are not evidence of faster convergence. Dense
matrices, factors and all-degree tail arrays remain free in this diagnostic.
A [memory-lean selection screen](results/taylor-selection-cost.json) separately
measures stable-tail binary search without an all-degree cache. Its overhead
materially reduces the standalone Taylor speedup; complete pricing of the
observed degree sequences is still needed. Analytic truncation bounds exclude
roundoff, which sometimes exceeds those bounds despite successful final checks.

The [completed degree pricing](results/taylor-inexact-cg-pricing.json) covers
all 32 dataset/degree combinations used by those 24 runs. Including a standalone
selector-cost estimate, adaptation is 12.6–31.3% slower than the strongest
priced exact/static control in five of six settings; the remaining estimated
win is only 1.5%. The adaptive stopping rule is less strict than the controls,
which favors its counts. These weighted estimates omit preconditioner and
setup work and are not native end-to-end timings. A further
[preconditioner-energy bound diagnostic](results/taylor-energy-cg-diagnostic.json)
updates the current direction's quadratic form through the PCG recurrence to
improve the step-length bound. All six runs converge, but mean degree decreases
by at most 0.103. An independent dense energy audit agrees within 5.5e-13
relative error. These results do not justify native integration of this Taylor
accuracy-selection route; a different mechanism is needed.

A distinct [fixed-block-cache oracle](results/block-cache-oracle.json) asks
whether the current direction can avoid exact kernel queries by applying
retained low-rank off-diagonal blocks. On four 2048-point smooth/mixed settings,
a core320 plus six rank6 blocks uses 660,000 retained bytes, less than the
693,888-byte core416 control. The oracle grants free SVD construction, true
block errors, exact step lengths and future iteration counts. It enumerates
all 64 symmetric fallback masks. No approximate block is usable on any step
under the prescribed residual-gap budget, so zero kernel entries are saved.
The smaller core then requires 24–67% more kernel entries than core416, before
cache arithmetic and setup costs. This closes the tested cache design, without
claiming that all hierarchical representations fail. Dense oracle workspace
is excluded, and no native total-memory or timing advantage is asserted.

A [large-core exchange oracle](results/large-core-exchange-oracle.json)
rechecks actual-column changes in a materially different regime: n=2048 with
an exact packed416-coordinate core, rather than the earlier rank16/32 Nyström
factors. None of 32 current-residual or matrix-selected exchanges beats
uninterrupted continuation, even with free selection and rebuilding. A paired
[exact-energy diagnostic](results/large-core-exchange-energy.json) then chooses
the outgoing coordinate to maximize the immediate energy decrease. Its total
iteration counts are 29/75/30/68, exactly matching an unchanged-core coordinate
correction plus restart, versus 27/73/29/67 without intervention. The energy
identity is checked numerically. Large core size does not make these constraint
changes benign, and no persistent benefit from retaining the new column is
observed. Fast factor updates are therefore not justified by these selectors.

A [spectral-floor feedback diagnostic](results/spectral-floor-feedback.json)
then holds the Nyström basis fixed and changes only the complement scaling.
The current direction and its already available matrix product provide a
curvature-matching scalar without another kernel action. All 60 comparisons
reach the original 1e-8 tolerance, but adaptive iterations are
56/527/51/218/135/1323 versus 40/338/38/169/88/748 for the best of seven fixed
scales. Freezing the first directional estimate or selecting from the initial
right-hand side also loses. This is a dense diagnostic with identical
one-direction flexible recurrences, not a full-memory timing claim. Direct
spectral cluster-placement prior work further limits novelty. The scalar
feedback route is not promoted.

The next [representation headroom screen](results/larger-core-refinement-headroom.json)
uses FP32 packed triangular storage to make a larger exact-elimination core
possible in principle. Exact core600 trajectories take 20/40, 22/45 and 22/36
iterations on 2048-point smooth/random/mixed systems, compared with
27/73, 31/84 and 29/67 for core416. Even charging three regenerated core
products per outer action gives 7–35% fewer kernel entries. These are exact
dense trajectories with free setup, not compressed-core solve timings. An
initial reduced-only endpoint attempt failed its final check; the retained
screen checks the original full residual at every step.
The [streamed factor primitive](results/packed-float-factor.json) constructs
all four tested FP32 factors without invalid pivots and solves using FP64
accumulation. However, its setup trace reaches 1,168,284 bytes, exceeding 1 MiB
despite a smaller named-array count. That discrepancy is under investigation.
Refinement reaches a residual floor comparable to the FP64 direct reference;
the requested 1e-12 core residual is generally unattainable with those residual
products. Neither memory feasibility nor an adaptive benefit is established.

The [allocation repair](results/packed-float-allocation.json) isolates the
excess setup memory to ctypes argument-wrapper retention. Passing addresses
of caller-owned contiguous arrays reduces the traced setup peak from
1,161,910 to 788,260 bytes. Factor, selected-index and single-solve hashes are
identical; original sources and measurements remain preserved. This establishes
setup feasibility, not full-solver memory feasibility.
A [core-error feedback replay](results/core-error-feedback.json) then tests
whether error directions from preceding outer right-hand sides improve the
compressed core inverse. All 128 core solves meet the diagnostic 1e-9 residual
target. Rolling BFGS corrections require 29/29 refinement products plus 16
additional base-error products per setting; base refinement needs 32/29,
and static initial/random corrections need 29/28 and 32/27. Even waiving the
extra products does not establish an advantage over static correction. That
feedback variant is closed; the complete static compressed-core solver still
requires its native memory and timing comparison.

The [first complete compressed-core pilot](results/compressed-core-schur-pilot.json)
fits the 1 MiB working allowance: the full-solve traced peak is 990,464 bytes,
and the named peak including reserve is 1,043,544 bytes. Fixed four-correction
core solves reach the original tolerance in all four cases. Two corrections
and tolerance-based stopping fail the harder-shift endpoints. Fixed4 appears
1.34–1.37x faster on the hard cases than the paired exact wrapper, but that
wrapper verifies every four iterations. A stronger exact baseline needs only
suspected/final verification; this comparison is pending. No adaptive benefit
has been established.
An exact-arithmetic identity supplies the next static controls: after j core
refinement corrections, H_j=C^{-1/2}[I-(I-C^{1/2}H0C^{1/2})^{j+1}]C^{-1/2}.
For odd j, H_j<=C^{-1}, so the corresponding Schur matrix is at least as
positive definite as the exact Schur matrix. One and three corrections are
therefore being checked with accurate boundary solves and stronger baselines.

The newest [precision/rank oracle](results/quantized-spectral-headroom.json)
uses exact eigenvectors to separate quantization error from approximation rank.
All 42 solves reach the first original-residual tolerance crossing. With a
128 KiB retained-factor allowance, FP16 and int8 retain more columns, but
quantization can destroy useful convergence: the hard mixed case takes 58
iterations with FP16 versus one with FP64 at the same rank. A smaller FP32
factor within the same byte allowance takes seven iterations, making it an
essential control for any selective-precision method. Exact eigendecomposition,
decoded factors and workspaces are excluded from this oracle; it establishes
headroom, not an adaptive or memory-feasible method.

A subsequent [selective-precision oracle](results/selective-precision-oracle.json)
promoted eight columns after the first iteration, reducing rank in advance to
keep the same retained-byte allowance. All 42 solves verified. Adding the first
residual to the selector produced exactly the same iteration counts as using
the initial right-hand side alone in all six cases. Large reductions in the
reference-action discrepancy did not establish a benefit from new solve
information. This promotion criterion is closed.

The [reference-action correction oracle](results/reference-action-correction-oracle.json)
uses a positive definite block-BFGS update to preserve high-precision inverse
actions on the right-hand side and one plain-CG residual. All 42 branches
converged. Its one-step mixed-case recovery is also obtained by a single
Galerkin warm start, so that result does not establish a benefit from persistent
repair. After the same warm start, repair saves a few iterations in some cases,
but does not produce a broad advantage over static precision choices. The hard
Concrete case still needs 314 iterations versus 266 for the existing static
int8 control. Dense construction and reference actions remain uncharged.

The [compression-error deflation oracle](results/compression-error-deflation-oracle.json)
then acquired one direction from the difference between high-precision and
compressed inverse actions on the current residual. All 66 branches converged.
Persistently preserving that direction improved five of six cases over using
it only for a Galerkin warm start. However, a static correction using the two
largest kernel eigenvectors was better in all six cases. This separates a useful
effect of persistent correction from the unsupported benefit of current-residual
direction selection. The latter selector is closed. The static oracle uses the
exact eigenvector-image identity; reference construction and compression costs
are still excluded from both sides.

The latest mechanism screen compares a smaller eliminated core plus block-Jacobi
preconditioning of its Schur complement with a larger core alone. After matching
packed factor storage, the 128-coordinate core with geometric blocks requires
38–167% more kernel-entry work than the 160-coordinate core across six seed-0
cases (smooth, mixed, Concrete; two shifts). The 64-coordinate alternatives are
worse. All 18 matched comparisons reach the original residual tolerance. This
is a dense oracle that ignores setup costs and exceeds the working-memory
allowance, so it establishes neither runtime nor memory feasibility. The earlier
18-run screen underused factor storage and is preserved separately. Geometric
blocks provide no headroom here.
See [the matched record](results/schur-block-matched-factors.json).

An [exact-error partition oracle](results/schur-cancellation-oracle-v2.json) then
regrouped variables to capture signed cancellations after 20 iterations, keeping
every block size fixed. Its objective improved by 62–97%, but complete convergence
lost to the uninterrupted old partition in all six cases. All 24 branch solves
and six newly computed larger-core controls stopped at their first literal
original-residual crossing of the tolerance. Exact-error grouping required
1.417–2.807 times the larger core's optimistic action-plus-verification entry
count. These counts omit partition construction, reconstruction work and one
initial Schur product for each restarted branch; including them cannot rescue
the losing adaptive branch. This objective is retired. The initial v1 record used
a stricter reduced-system stopping test; it is preserved but superseded for
this comparison by v2's original-system stopping test.

The latest fixed-memory comparison covers all five development workloads and
three regularizations at seed 0. Of 60 attempted runs, 53 converged, three
Airfoil configurations exceeded the memory allowance, and four plain-CG runs
reached their iteration or CPU limit. At the 1 MiB working allowance, fixed
320-coordinate elimination totals 2.382 seconds; choosing the fastest feasible
configuration separately for every case totals 2.314 seconds. This single-repeat
comparison leaves only 2.85% aggregate time for selection overhead.

A new shared-action implementation lets nested elimination sets use one maximum
Cholesky factor and the same generated kernel blocks. Independent numerical
checks and allocation traces pass; two synchronized candidate actions query
exactly `n² - m_max²` kernel entries. Extra arithmetic, solver states and smaller
tiles still cost time. The tested 1 MiB candidate races are not advanced because
320-coordinate elimination takes fewer iterations than both 256-coordinate
elimination and plain CG in every case. Smaller-budget selection headroom is
being evaluated. Racing solvers and batching their products are
established approaches; the shared implementation is not yet a research result.

A subsequent 36-run screen at 256 and 512 KiB finds different winners. At
256 KiB on the weakly regularized mixed target, plain CG takes 0.121 seconds
and 96-coordinate elimination takes 0.584 seconds. Elimination remains much
faster on the smooth target. All 18 distinct allocation audits pass. These are
single observations; the hard Concrete case is censored for all candidates at
256 KiB, so its complete-cost comparison remains unresolved. A shared-race
pilot now completes all six cases at the original true residual tolerance.
It selects plain CG for the difficult mixed case and Schur96 otherwise. The
mixed-case time improves from 0.590 to 0.164 seconds, but aggregate time is
8.400 seconds versus 7.610 for fixed Schur96 using the same representation.
The 10.4% aggregate slowdown prevents promoting this prototype. Four censored
plain-CG controls remain in the results.

Packed Cholesky storage passes an independent feasibility check and permits
larger static cores without storing the unused triangle. Upper-packed leading
prefixes are contiguous, so nested candidates can still share one factor.
The subsequent 36-attempt static comparison confirms a stronger baseline:
at 256 KiB, the weakly regularized smooth case takes 0.369 seconds with
160 coordinates, and Concrete takes 3.709 seconds with 128 coordinates.
At 1 MiB, Concrete takes 0.214 seconds with 448 coordinates. These are single
observations, and infeasible configurations remain recorded. The packed shared
solver passes independent numerical and memory checks. Its subsequent 36-row
complete-cost pilot also loses overall: the 128-coordinate race totals about
5.42 seconds versus 4.97 for its fixed counterpart; the 160-coordinate race
totals about 5.56 seconds versus 3.63 for its fixed counterpart. All twelve
races reach the true tolerance, while four plain controls remain censored.
The current racing approach is closed. Its operators remain useful controls,
and the broader research objective requires a different mechanism.

Under the revised memory constraint, a streamed Nyström implementation can
reuse residual projections accumulated during the main kernel product. In two
single-repeat smooth-target comparisons, this reduced total time from
0.233 to 0.207 seconds and from 1.767 to 1.630 seconds. The factors are identical
within each pair, all solves meet the true residual tolerance, and the small
iteration differences caused by finite precision are recorded. This is a local
implementation benefit; stronger static representations still need comparison.

Implicit Schur-CG provides such a control: it stores a principal-block Cholesky
factor and recomputes cross-kernel products. Its initial runs show substantial
headroom from using a larger eliminated set. A source audit found that the old
true-residual check briefly retains two kernel tiles, invalidating its claimed
1 MiB peak allowance. Corrected controls explicitly release each tile and count
the setup, selection and verification workspaces.

The corrected implementation completed 42 solves comparing growth from 64 to
192 eliminated coordinates with fixed and restarted controls. Current-residual
f/P selection lost to both upfront farthest-point selection and delayed
matrix-only selection in all six workload/regularization cases. Using its final
selected set upfront also lost in all six cases, so the failure is not explained
only by restarting. This selection rule is rejected. The largest named-array
allowance was 1,048,448 bytes; a separate smooth/m192 allocation trace peaked at
885,114 bytes. Common immutable inputs are reported separately and are identical
across competing methods. No method is yet promoted as the research result.

A further static control assembles one Fortran-order principal block, factors
it in place, and selects kernel tile sizes from the remaining memory. All 12
tested solves with 192, 256 and 320 coordinates fit the same allowance, with
separate allocation traces. At 320 coordinates and `lambda=1e-5`, smooth and
Concrete take 0.079 and 0.424 seconds in single observations. This is a much
stronger control for future methods. The earlier 192-coordinate limit came
from conservative setup accounting; it was not the largest feasible factor.
The maximum named storage plus reserve is 1,043,768 bytes, and the largest
independent allocation trace is 985,620 bytes. These figures exclude the same
immutable inputs for every competing method; full-process RSS is also recorded.

Simply retaining the completed Krylov prefix with an unchanged operator does
not supply a new exact-arithmetic advantage: CG already optimizes over that
space. Reorthogonalization can repair rounding errors, and augmentation can
preserve information across a restart, but these are established mechanisms.
The next experimental method has not yet been selected. The failed residual
admission rule is closed, and the broader research objective remains active.

The method revision has separated several causes of failure. The original
inverse quadratic-form score predicts poor subsequent progress. Preserving a
short FCG history after a preconditioner change can also severely slow convergence;
an independent transition replay found no implementation error. Coarse-space
balancing improves the lower-rank starting solver, but generally loses to using
the same storage for more kernel columns. These diagnostics are preserved under
`results/`, with their scripts in `scripts/`.

One revision learns a rank-k Nyström factor from the first ten search
directions, then freezes it and restarts. This uses a generalized sketch rather
than a set of k actual columns. A separate variant selects k actual columns from
the learned sketch using QR pivoting. Both are compared with fixed RPCholesky,
Gaussian-sketch QR selection, static QR selection, and a matched restart.
The stronger two-pass Gaussian Nyström control removes its apparent advantage
in the first complete-cost comparisons. Further fixed-nonzero Schur-FSAI pattern
experiments also show no consistent benefit from solve-aware selection. The
direct block-correction test reduces kernel-entry queries, but its matrix-free
implementation remains slower than fixed preconditioners within the same
retained-storage allowance. Sequential greedy block selection is established
prior work; the additional bounded history does not give a consistent time
advantage. Global FSAI neighbor searches also fail to improve the solve
consistently. Allocating a fixed total number of sparse-factor entries across
all rows gives a stronger negative result: residual-weighted allocation takes
more iterations than matrix-only allocation in all eight tested cases.
An exact two-block oracle also fails: one residual-selected coupling update
loses all 16 comparisons against a matched restart; periodic updates lose
15 of 16. Every update decreases the exact current inverse-action error,
and all solves reach the true residual tolerance. The oracle's preconditioners
share exact eigenvectors, so noncommutation is not necessary for this failure.
A difficult mode with a small residual component can still become costly
when its preconditioning is removed.

Further controls also retain the old search direction or bound the relative
spectral change of a rank-preserving update; neither gives a useful advantage.
A full-prefix projection substantially improves the unchanged solver, while
column swaps add at most one iteration of improvement in the tested cases.
The stable projection oracle retains the entire prefix and uses dense matrix
geometry. Its inexpensive streamed approximation loses the required
orthogonality, so the oracle improvement is not a practical result.

A two-pass reconstruction of four Ritz vectors reproduces the original
trajectory exactly and yields valid spectral corrections. After including
replay and matching retained storage, both spectral and balanced corrections
lose to Gaussian-power controls in all eight tested cases. Additional fixed
subspace iterations also strengthen the baseline: on the mixed target at
`lambda=1e-5`, rank-32 power Nyström needs 162, 30 and 18 iterations with
two, four and eight setup passes. Installing the same refined factors after
a twenty-step prefix does not recover those upfront gains. These are offline
iteration comparisons, with all setup passes recorded; they are not time claims.

Selecting columns before PCG with the established target-dependent f/P-greedy
criterion also loses to matrix-only selection in all ten tested cases. Giving
both methods a coordinate Galerkin initial guess does not reverse that result.
The coordinate pilot uses exhaustive projected/prefitting selection; it should
not be read as a performance verdict on the cheaper published GBCD algorithm.

Recomputing the full prefix's operator images in a shared kernel block pass
also fails an optimistic complete-cost screen. Even with omitted continuation
overhead and favorable batch timings, it loses all six comparisons with matched
static controls. The independently checked width-two block-CG control does
reduce kernel passes in all 30 offline comparisons, while increasing total
right-hand-side arithmetic. Native CPU timings and full storage comparisons
are required before interpreting that established solver's practical benefit.

These methods remain reproducible diagnostics; none is promoted as the new
research method. The next mechanism must address subsequent convergence,
including the difficult directions that a single residual may underweight.
The full original dataset, budget, and regularization scope remains the
validation target. Controlled mixed-eigenspace targets are now implemented
alongside smooth and random targets. Their kernel and split are identical;
training-only spectral construction is checked independently, and solver
inputs contain no eigenvectors. The first mixed-target execution check includes
21 converged baseline solves: preconditioned runs across all three
regularizations and all three rank budgets, plus unpreconditioned CG.
Unpreconditioned CG is substantially faster on the most weakly regularized
mixed case, so it remains an essential control. Confirmation seeds remain unused.

### Initial implementation

The [research record](docs/initial-results.pdf) and its
[LaTeX source](docs/initial-results.tex) describe the first experiments.
The initial implementation passed 19 tests. Eight methods completed the initial
execution check; the four-method comparison contains 240 completed solves
across four workloads, five seeds, and three timing repeats.

At `k=16`, `lambda=0.001`, checkpoint interval 20, and four candidates, residual
exchanges were slower than fixed preconditioning in all 20 paired cases.
The median paired time ratios were 2.24 for smooth synthetic targets, 2.36 for
random synthetic targets, 2.89 for Concrete, and 2.34 for Airfoil. Exchange work
accounted for a median 5.0% of adaptive solve time. Restarted fixed PCG also used
more iterations than uninterrupted fixed PCG in every case.

At 60 inspected checkpoints, the exchange score decreased, but the new
preconditioner gave a larger squared energy error after both 5 and 10 further
iterations. Both diagnostic branches restarted from the same iterate.
Subsequent branch, frozen-set, and transition diagnostics separate the effects
of the score, selected columns, restarting, and inherited solver history.

The first timing batch reached its execution limit after 237 completed records;
the remaining three were completed with identical source and settings in
`results/initial-001-continuation/`. The interrupted attempt remains preserved.
Use `results/initial-analysis.json` for the complete paired comparison.

## Files

- `proposal/`: unchanged submitted LaTeX, PDF, and bibliography.
- `docs/instructor-feedback.txt`: instructor feedback and recommended reference.
- `docs/research-plan.txt`: the original detailed research plan.
- `docs/iterativecur-notes.txt`: reading notes and comparison with IterativeCUR.
- `docs/experiment-protocol.txt`: execution, comparison, and interpretation rules.
- `docs/method-revision.json`: active research goal, decisions, and execution record.
- `docs/hierarchical-preconditioner-hypothesis.tex`: exact two-block diagnostic,
  derivation, observed negative results, and resource accounting.
- `docs/initial-results.tex` and `.pdf`: first results and the next experiment.
- `src/nystrom_project/`: numerical method, solver baselines, data preparation,
  paired diagnostics, and experiment runner.
- `configs/`: exact experiment settings.
- `tests/`: independent numerical and solver correctness checks.
- `data/manifest.json`: dataset sources, licenses, and checksums.
- `results/`: raw histories, exchange decisions, run metadata, and summaries.
- `ccfa.yaml`: project status and evidence links.

## Environment

Python 3.13 and a single BLAS thread are used for the initial experiments. On this
WSL machine, `.venv` points to `/home/maosen/.cache/cs6220-project-venv`; installing
the environment on the Linux filesystem avoids slow package access on `/mnt/c`.
The project and results stay in this directory. `requirements-lock.txt` records
the installed versions. To create a separate environment on another machine:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[test]'
```

For the exact recorded package versions, run
`.venv/bin/python -m pip install -r requirements-lock.txt` from this directory.

Run commands below from `project/`:

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests
.venv/bin/python -m nystrom_project.datasets --root .
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m nystrom_project.diagnostics \
  --root . --config configs/diagnostic.json --output results/diagnostic-NEW.json
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m nystrom_project.benchmark \
  --root . --config configs/pilot.json --output results/pilot-NEW
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m nystrom_project.benchmark \
  --root . --config configs/initial.json --output results/initial-NEW
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m nystrom_project.sweep \
  --root . --config configs/revision-learning-medium.json --output results/revision-NEW
```

Choose a new output name for each execution; existing measurements are preserved.
For repeated subprocess runs on this WSL machine, use the absolute Linux path
`/home/maosen/.cache/cs6220-project-venv/bin/python` in place of `.venv/bin/python`
to avoid repeatedly resolving the environment through the Windows mount.
The runner stores source hashes, settings, software versions, all iteration
histories, and all exchange decisions. Dense Cholesky references are constructed
outside the timed workers. Every timed worker applies the same blockwise kernel
operator and includes setup, updates, restarts, and residual verification.

Regenerate the current combined analysis and report:

```bash
.venv/bin/python scripts/summarize_results.py --root . \
  --campaign results/initial-001 --continuation results/initial-001-continuation \
  --diagnostic results/diagnostic-001.json --output results/initial-analysis.json
bash scripts/build-results.sh
```

The build uses Tectonic from `PATH` or the existing course workspace cache.
`CS6220_TECTONIC` can select a different executable.

## Instructor reference

[IterativeCUR: Large Rank-Adaptive Approximation From a Small Recycled Sketch](https://arxiv.org/abs/2509.21963v2)
uses a matrix approximation residual to grow a CUR approximation. The submitted
proposal instead used a linear-system residual to exchange columns at a fixed
column budget; those initial exchanges failed the complete-cost comparison.
The current user-authorized scope allows representation changes under a fixed
total memory budget, while retaining the requirement for useful current-solve
feedback. The proposal remains preserved as submitted.

In v2, Section 3.3 separates additional Gaussian vectors used for error evaluation
from those used for column selection. Its bound concerns matrix approximation
error in Frobenius norm, not subsequent linear-solve time. Reusing a solver
residual as a selection direction therefore does not inherit that independent
error estimate or its stopping guarantee. Sections 3.2–3.4 motivate incremental
reuse of matrix information, but the corresponding benefit must be established
under our implicit-kernel access, full-memory, and setup-inclusive cost model.

### Native block Schur controls: complete-cost result

The fixed Gaussian and geometric auxiliary block-CG controls converge in all eight cases but lose to the strongest scalar control in each of the four development systems. Gaussian/geometric versus best scalar times (seconds) are smooth 2.448/2.438 versus 2.033; mixed 2.221/2.309 versus 1.850; Concrete 0.291/0.283 versus 0.207; Airfoil 0.585/0.584 versus 0.321. These are one-repeat development measurements at seed 0 and regularization 1e-5, including setup and original-residual corrections. All 20 timed rows converge. Full allocation audits pass at traced peaks 1,007,086 and 1,000,339 bytes.

There are no numerical search-rank losses. The only auxiliary convergence event occurs at iteration 43 of 44 in smooth Gaussian block CG. Thus the observed trajectories do not support early release of block-vector memory to grow the principal core. This conditional follow-on is not advanced. Shared kernel generation does not compensate for the smaller core and additional actions/checks. These findings do not establish adaptive preconditioning benefits; the research goal remains active and unachieved. Source and full results: `scripts/check_native_block_schur.py`, `results/native-block-schur.json`. Whole-command charge: 41.58 wall seconds, 33.75 CPU seconds.

### Pair rotations before exact Schur elimination

A dense diagnostic changes the coarse basis using disjoint core–tail coordinate pairs, preserving all couplings in the transformed Schur system. Four seed-0 systems (smooth/mixed, n=2048, shifts 0.001 and 1e-5), core416 and checkpoint10 were tested. All24 baseline/branch solves meet the literal original residual tolerance1e-8.

| Method | Smooth .001 | Smooth 1e-5 | Mixed .001 | Mixed 1e-5 |
|---|---:|---:|---:|---:|
| Unchanged uninterrupted |27|73|29|67|
| Unchanged restart |28|75|29|68|
| Fixed45 degrees upfront |22|67|24|60|
| Fixed45 degrees after10 |25|70|26|64|
| Exact remaining-error alignment after10 |33|104|35|95|
| Recent displacement alignment after10 |32|94|35|92|

Counts include the10-step prefix for delayed methods. Pairing, reference error and rebuilding are free diagnostic resources; these are iteration counts, not CPU speedups or memory-feasible runs. The unpaired coordinates retain65.0–74.3% of the remaining error norm, so the oracle does not automatically capture the entire error. The exact-error and current-displacement alignment rules both fail against the controls. This rejects these rules, not all sparse orthogonal coarse spaces. The fixed geometry result does not establish adaptive preconditioning. No native integration is advanced. Source/results: `scripts/check_pair_rotation_schur.py`, `results/pair-rotation-schur.json`. Successful command15.25 wall seconds; an initial duplicate metadata keyword caused a compile-only failure costing0.67 seconds, preserved separately. Total campaign charge15.92 seconds.

### Preserving local constants during triplet adaptation

The next diagnostic preserves one constant vector on each of208 disjoint triples and changes only the second coarse direction. The initial contrast is the dominant direction of the local kernel restricted to the mean-orthogonal plane. The exact Schur system retains all other directions and couplings. In the same four n2048 development cases, all16 newly solved baseline/branch runs reach original relative residual1e-8.

| Method | Smooth .001 | Smooth 1e-5 | Mixed .001 | Mixed 1e-5 |
|---|---:|---:|---:|---:|
| Fixed triplet basis, uninterrupted |28|88|30|82|
| Fixed triplet basis, restart |30|91|32|83|
| Current-displacement contrast |30|93|32|86|
| Exact-error contrast oracle |31|97|32|85|

Delayed counts include the10-step prefix. Every local constant column is unchanged, no zero-candidate fallback is used, and the oracle removes the local fine-coordinate error to roundoff. Neither contrast update improves the matched restart. Previously verified coordinate416 and fixed-pair-sum controls are stronger than the static triplet basis too. Thus retaining local constants does not rescue these alignment rules; the tested formulation is closed without angle, prefix or anchor-count sweeps. This does not rule out all adaptive aggregation. Dense reference error and rebuilds are free in this diagnostic, so it establishes no runtime or memory advantage.

Results/source: `results/triplet-rotation-schur.json`, `scripts/check_triplet_rotation_schur.py`. Command cost14.37wall/9.88CPU seconds. Related algebra establishes a possible streamed action with n²−3g² kernel entries, independently checked numerically; this is an implementation possibility, not an adaptive result or a measured speedup. The scientific goal remains active and unachieved.

### Positive correction of compressed-core inverse error

The next method reuses the first four core-refinement calculations. For z=H0t, r=t−Cz and delta=H0r, it retains S=[z] and W=[delta] and forms T=SᵀCS using the already computed Cz. The correction H1+W T^-1 Wᵀ satisfies H1≤Hcorr≤C^-1 in exact arithmetic. This is established symmetric two-level machinery; useful same-solve acquisition and complete-cost gains were the tested questions.

All28 dense trajectories meet original relative residual1e-8. The candidate uses core568 to leave space for two rank-four archives, versus static core592; complete native allocation remains untested.

| Method | Smooth2048 | Mixed2048 | Concrete824 | Airfoil1202 |
|---|---:|---:|---:|---:|
| Core568 unchanged |45|42|21|27|
| Core568 matched restart |46|42|22|28|
| Core568 current correction |46|42|23|28|
| Core592 H1 |42|38|20|27|
| Core592 H2 prefix then H1 |43|38|18|23|

Counts include all inner iterations across common original-residual repair cycles; the new-operator restart also pays an extra Schur action. Current acquisition uses no additional C products, retains rank4 in every case, and has Gram condition numbers10.7–999.7. Both real-data candidates still need an outer repair. Random upfront/delayed controls are included in the full results. No adaptive gain or native performance result is established. A possible next discriminator is a free-information best coupled rank-four inverse-error correction, to distinguish poor sample directions from insufficient representational capacity.

Source/results: `scripts/check_core_error_subspace.py`, `results/core-error-subspace.json`. Tiny Gram/correction identities pass at1.36e-15/5.30e-16; inverse-error ordering holds to roundoff. Whole command5.98wall/7.18CPU seconds.

### Coupled core-error oracle

A follow-up uses the complete positive inverse error F=C^-1−H1=ZZᵀ. The top four right singular vectors of BᵀZ define a correction minimizing spectral/Frobenius error of the Schur-gap approximation. This is matrix-error optimality, not optimal iteration count. All spectral construction and complete error information are free diagnostic resources.

| Method | Smooth2048 | Mixed2048 | Concrete824 | Airfoil1202 |
|---|---:|---:|---:|---:|
| Core568 unchanged |45|42|21|27|
| Rank4 coupled oracle upfront |45|42|21|25|
| Matched restart |46|42|22|28|
| Rank4 coupled oracle after4 |46|42|22|26|
| Exact core568 reference |45|42|19|22|
| Best tested core592 H1/prefix control |42|38|18|23|

All20 new solves converge, and the eight repeated unchanged/restart controls match the previous solution hashes bitwise. Rank4 leaves42.4%,42.4%,22.7%,16.7% of the Schur-gap spectral norm respectively; it still requires original-residual repair on both real datasets. It saves two iterations only on Airfoil, remaining weaker than the stronger static controls in every case. The exact-core reference preserves the common H4 boundary and grants free exact core actions; its storage is not feasible under the candidate budget. Exact core accuracy does not change synthetic iteration counts, supporting the matrix-block-size explanation there.

These results close the tested rank4 core-error repair route without a rank/prefix sweep. They are not an impossibility theorem for arbitrary low-rank corrections. Source/results: `scripts/check_core_error_oracle.py`, `results/core-error-oracle.json`. Whole command4.57wall/6.77CPU seconds. No native/adaptive success is established.

### Retiring part of a principal core

An exact-factor diagnostic tests whether a592-coordinate core can be reduced to its416-coordinate prefix after10 iterations, freeing working memory for faster kernel tiles. All16 baseline/branch solves reach the original residual tolerance.

| Case | Core592 throughout | Core416 throughout |592 then416 total | Required small/large action-cost interval |
|---|---:|---:|---:|---|
| Smooth2048 |40|73|67|greater than0.625 and less than0.526|
| Mixed2048 |36|66|61|greater than0.667 and less than0.510|
| Concrete824 |18|38|29|greater than0.526 and less than0.421|
| Airfoil1202 |21|27|26|greater than0.909 and less than0.688|

For prefix k, large static count N_L, small static count N_S and post-switch count N_T, beating both endpoints requires k/(N_S-N_T)<c_small/c_large<(N_L-k)/N_T. All four intervals are empty. This grants free setup and transition and assumes fixed application cost for each core size; it is not a native timing measurement or an impossibility result for other switch points. Dense exact592 factors also exceed the practical memory allowance. The tested policy is not advanced to native implementation. Source/results: `scripts/check_schur_core_retirement.py`, `results/schur-core-retirement.json`. Command3.98wall/6.76CPU seconds.

### SPD nested HSS representation gate

A static diagnostic retains nested off-diagonal couplings using bottom-up SPD scaling and projection. It selects the largest uniform rank fitting a provisional component planner at 1 MiB; dense construction and a dense inverse are free diagnostic resources. This does not establish a compact implementation or complete memory feasibility.

| Case | Rank | Iterations at stop | Original relative residual | Stop |
|---|---:|---:|---:|---|
| Smooth2048 |10|120|2.3277|CPU cap|
| Mixed2048 |10|101|0.6117|CPU cap|
| Concrete824 |19|2000|4.8601e-4|Iteration cap|
| Airfoil1202 |13|919|0.7820|CPU cap|

None reaches the target 1e-8. The full-rank tiny identity check has relative error 8.80e-15; the compressed tiny matrix is positive definite and nested-basis error is 3.00e-15. Independent algebra review found no material scaling/projection or nesting error. These results close this tested configuration, not HSS as a whole. No adaptive allocation mechanism or end-to-end gain is established. The nominal kernel-entry count excludes extra products used to inspect true residuals.

Source/results: `scripts/check_scaled_hss_headroom.py`, `results/scaled-hss-headroom.json`. Whole command 30.93 wall / 33.46 CPU seconds. The project goal remains active and unachieved.

### Why the tested HSS representation failed

Dense generalized eigenanalysis of the same four matrices distinguishes a few outliers from broad spectral distortion. The condition numbers are 1.24e7 (both synthetic cases), 5.38e5 (Concrete), and 9.26e6 (Airfoil). Even free replacement of 32 exact eigenmodes, choosing the retained contiguous spectral window with the lowest condition number, leaves 1.36e6, 1.36e6, 2.53e4, and 1.55e5 respectively. Reaching condition number 100 in this spectral replacement model requires 860, 860, 274, and 333 replacements.

These are spectral diagnostics, not CG iteration lower bounds or evidence against every low-rank update. They give no reason to invest in a small learned spectral correction on top of the tested HSS matrix. The full-rank original operator remains unchanged; data hashes and reconstructed approximation errors match the preceding run. Source/results: `scripts/check_hss_spectrum.py`, `results/hss-spectrum.json`. Whole command 17.74 wall / 19.05 CPU seconds.

### Current core-energy accuracy selector

A distinct selector removes the six random acquisition probes used earlier. It reuses e=t−C H0t and delta=H0e from the first refinement, and compares sqrt(eᵀdelta/(pᵀDp−tᵀH1t)) with 0.1 times the inner target divided by the recursive residual norm. It selects H3 when the proxy exceeds this threshold, otherwise H1. The threshold was fixed before testing. This is a directional energy heuristic, not an action-error certificate.

All20 solves reach original relative residual1e-8; eight repeated static controls match earlier solution hashes bitwise. Selector iterations are42/38/18/22 (smooth/mixed/Concrete/Airfoil), including25/22/12/16 H3 actions. Analytical kernel-entry costs exceed the best tested static controls by9.41%,9.10%,38.85%,7.47%, respectively. This metric includes setup, core/cross products and residual repairs; it excludes scalar arithmetic and triangular-solve work, so it is not a timing result. No native promotion or threshold sweep follows.

Independent review found no count/order mismatch on executed paths. The unused nonpositive-curvature score branch could overflow during serialization; all recorded scores are finite and all solves converge. Source/results: `scripts/check_core_energy_selector.py`, `results/core-energy-selector.json`. Command5.11 wall /6.48 CPU seconds.

### Removing the first core correction

The prior H1/H3 selectors cannot eliminate H1's first core matrix product. A separate fixed-H0 diagnostic removes it using the same FIRST8 factor, H4 boundary treatment, and original-residual repairs. All four resulting Schur matrices pass Cholesky; their smallest eigenvalues are approximately1e-5.

| Case | First-cycle iterations | First-cycle original residual | Total iterations after repairs | Kernel entries |
|---|---:|---:|---:|---:|
| Smooth2048 |42|7.32e-6|61|253306024|
| Mixed2048 |38|4.00e-6|56|234086824|
| Concrete824 |18|2.69e-5|30|21790824|
| Airfoil1202 |22|1.08e-4|39|58478640|

Every final original relative residual is below1e-8. The analytical kernel-entry count nevertheless exceeds the best tested H1/prefix control in all four cases. Residual repair, not loss of positive definiteness, defeats this fixed-H0 cost saving. This is a dense diagnostic, not native runtime or adaptive-method evidence. Any further skipping rule needs a mechanism addressing repair cost, with fixed H0 retained as a comparator. Source/results: `scripts/check_unrefined_first8_core.py`, `results/unrefined-first8-core.json`. Command2.68 wall /5.61 CPU seconds.

### Reusing first-cycle directions during residual repair

Original-residual repair changes the reduced right-hand side, so first-cycle history can in principle help even when the Schur operator is unchanged. A dense diagnostic retains all first-cycle directions and their Schur images, orthonormalizes the directions by SVD, and compares no reuse, a Galerkin warm start, and a Galerkin warm start followed by deflated CG. Archive storage, SVD, and projection work are granted free in the analytical kernel-count comparison.

| Case | No archive | Warm start | Deflated | Deflated kernel entries / best H1-prefix control |
|---|---:|---:|---:|---:|
| Smooth2048 |61|59|58|1.298|
| Mixed2048 |56|53|52|1.290|
| Concrete824 |30|26|25|1.117|
| Airfoil1202 |39|37|36|1.214|

All12 solves meet original relative residual1e-8. The first-cycle metadata match the preceding H0 run exactly, and all four no-reuse solution hashes match bitwise. Full archives retain42/38/18/22 directions and require978432/885248/66816/214720 bytes for Q and SQ alone, in addition to the core and other workspace. The tested policy does not justify implementing a compressed archive.

Independent review verified the projected-direction recurrence and SVD image reconstruction. Concrete has relative reconstructed-Gram asymmetry3.10e-7; others are at most6.23e-9. This is a floating-point archive experiment, not an exact optimality bound against every subspace or stopping policy. Source/results: `scripts/check_first_cycle_archive_oracle.py`, `results/first-cycle-archive-oracle.json`. Command4.15 wall /5.98 CPU seconds.

### Signed H0 correction without core products

Unlike the earlier positive H1 correction, a signed H0 correction can avoid the first core matrix product entirely. With C=L(I+E)Lᵀ and eigenpairs(theta,V) of E, the pilot retains the four largest absolute values of −theta/(1+theta), giving Hcorr=H0+W diag(signs)Wᵀ. This truncates whitened inverse error; it does not optimize the coupled Schur error or CG iterations. Complete eigensystem construction is free in this diagnostic.

At core584, unchanged versus corrected totals are63→62,57→57,31→31,39→39 for smooth/mixed/Concrete/Airfoil. All8 solves meet original relative residual1e-8. An independent audit checks that every corrected Schur matrix is SPD, with minimum eigenvalue approximately1e-5; SPD of Hcorr alone would not establish this. Rank4 leaves48.3%,48.3%,40.7%,64.5% of whitened inverse-error spectral norm. The factor, metadata, and correction occupy725832 bytes versus726272 for the core592 control; full working-memory feasibility is untested. This spectral-selection policy is not advanced to native implementation. Source/results: `scripts/check_signed_h0_correction.py`, `results/signed-h0-correction.json`, and `results/signed-h0-spd.json`. Combined commands8.17 wall /11.28 CPU seconds.

The earlier Concrete archive ambiguity was also checked using directly recomputed SQ for the identical retained Q. Image change is7.29e-7 relative and Gram asymmetry falls to2.24e-10. The solver still takes25 iterations and passes original tolerance, even granting all18 refresh actions free. The negative decision is unchanged. Source/result: `scripts/check_concrete_archive_image_refresh.py`, `results/concrete-archive-image-refresh.json`. Command1.84 wall /4.16 CPU seconds.

### Correcting the coupled Schur error

A follow-up targets G=Bᵀ(C^-1−H0)B directly, retaining its four eigenpairs of largest absolute eigenvalue and subtracting that signed rank-four approximation from S0. This is optimal rank-four truncation of G in spectral/Frobenius norm, not iteration count. Dense G construction and eigendecomposition are free. The core size is the largest integer fitting the same retained-factor/metadata allowance after adding four tail vectors and signs; full working-memory feasibility is not established.

| Case | Core | Uncorrected iterations | Corrected iterations | Corrected entries / best H1-prefix control |
|---|---:|---:|---:|---:|
| Smooth2048 |572|65|65|1.448|
| Mixed2048 |572|61|59|1.454|
| Concrete824 |588|30|26|0.9292|
| Airfoil1202 |583|39|37|1.241|

All8 endpoints pass original relative residual1e-8; all corrected Schur matrices pass Cholesky. Concrete saves7.08% in analytical kernel-entry count, while the other three cases remain worse. This is a limited static free-information opportunity, not evidence of a feasible learned method or full-scope success. Acquisition, working storage, and correction arithmetic still need to be paid. Independent review found no material algebra/count issue. Source/results: `scripts/check_coupled_h0_schur_correction.py`, `results/coupled-h0-schur-correction.json`. Command6.91 wall /7.72 CPU seconds.

### Paying for the prefix and isolating restart cost

The coupled-H0 correction was next installed after four H1 actions. Those core products are charged, while the optimal G directions are still supplied free; this does not implement or fully price their acquisition. A restart variant recomputes the new Schur residual and charges that action. A matched no-restart variant retains the recurrence as a changing-operator heuristic and relies on the same literal original-residual acceptance.

| Case | Same-core prefix only | Prefix + correction + restart | Prefix + correction, no restart | Core592 prefix only |
|---|---:|---:|---:|---:|
| Smooth2048 |63|65|63|60|
| Mixed2048 |57|60|56|52|
| Concrete824 |25|27|25|25|
| Airfoil1202 |34|38|31|34|

All24 runs converge; four repeated prefix-only controls match solution hashes bitwise. Without restart, correction helps Mixed and Airfoil but still loses the strongest available static kernel-entry cost in all four cases. On Concrete, the corrected no-restart method ties the same-core prefix-only control and costs more than the core592 prefix-only control. Thus the prior free-upfront7.08% advantage does not establish a paid correction benefit. No schedule sweep or learned-rank4 implementation follows this tested route. Per-row oracle metadata refers to the candidate core; actual m and factor_builds identify each control.

Sources/results: `scripts/check_paid_h0_correction_prefix.py`, `results/paid-h0-correction-prefix.json`, `scripts/check_no_restart_h0_correction_prefix.py`, `results/no-restart-h0-correction-prefix.json`. Combined13.41 wall /16.50 CPU seconds.

### HODLR compression of the Cholesky factor

A different representation compresses lower sibling blocks of the exact Cholesky factor, retaining exact leaf triangles. The resulting factor remains nonsingular because its positive diagonal is unchanged; its product with its transpose is therefore SPD. Two predeclared allocations are tested: the largest uniform rank fitting a component planner, and greedy increments ranked by next singular-value squared per byte. This is a factor-error heuristic, not an optimal inverse-error allocation. Full dense construction, SVDs and factor applications are granted free; the planner does not establish native peak memory.

| Case | Uniform rank | Uniform stop/residual | Greedy stop/residual |
|---|---:|---|---|
| Smooth2048 |5|CPU cap /21.83|CPU cap /24.89|
| Mixed2048 |5|CPU cap /17.52|CPU cap /62.25|
| Concrete824 |23|2000 iterations /0.2069|2000 iterations /1.49e-4|
| Airfoil1202 |13|CPU cap /5.339|CPU cap /12.29|

None of8 runs reaches original relative residual1e-8. Greedy allocation lowers relative factor Frobenius error to0.0698/0.0698/0.0233/0.0184, but this does not recover convergence. Full-rank reconstruction on a tiny check agrees to1.06e-15. These tests close the two specified allocations, not all hierarchical factorizations. No adaptive-rank or compact native implementation follows. Existing HODLR and mixed-precision prior work is recorded in the literature notes; no new format claim is made. Source/results: `scripts/check_hodlr_cholesky_headroom.py`, `results/hodlr-cholesky-headroom.json`. Command47.07 wall /49.74 CPU seconds.

### Local inverse scaling of triangular blocks

To distinguish raw factor error from inverse-factor error, the next gate approximates L22^-1 L21 by truncated SVD, then stores L22 U Sigma Vᵀ. The exact child factor is used independently at each block. In a two-block factor with unchanged diagonal children, the lower-left block of Ltilde^-1 L−I is exactly L22^-1(L21−L21approx). That identity passes a tiny check at2.53e-15; full-rank reconstruction passes at1.24e-15.

All8 uniform/greedy runs still fail the original1e-8 target within the same caps. Concrete reaches2000 iterations with residual0.02014/0.06710; the other cases reach their CPU caps. The measured global quantity ||Ltilde^-1 L−I||F/sqrt(n) is80.21 on the synthetic uniform factors and256.83 on the greedy factors; Concrete is4.26/8.94. Local inverse scaling therefore does not prevent large global error when many blocks are compressed independently. Independent algebra review found no material implementation issue. This closes the tested local criterion, not all triangular-factor representations. Source/results: `scripts/check_scaled_cholesky_factor_headroom.py`, `results/scaled-cholesky-factor-headroom.json`. Command50.16 wall /52.20 CPU seconds.

### Direct inverse-Cholesky compression

To avoid inverting an already compressed factor, this gate compresses Z=L^-1 directly and applies ZtildeᵀZtilde through two matrix-vector products. Each off-diagonal block is approximated after right multiplication by R, where RRᵀ=A[J,J] for its column indices J. This minimizes that block's weighted Frobenius contribution ||deltaZ L[J,:]||F; contributions from different blocks can interact. Exact positive diagonal entries preserve SPD. Construction and dense applications remain free diagnostic resources.

Concrete reaches original relative residual1e-8 in1688 uniform-rank iterations and738 greedy-allocation iterations. The other six cases hit their CPU caps without convergence. Global normalized ||Ztilde L−I||F/sqrt(n) is19.58/15.99 on the synthetic cases and2.45/1.46 on Concrete. Direct inverse-factor compression changes the failure but gives no competitive static representation at the tested allocation. Full-rank reconstruction and the single-block weighting identity check at1.11e-15 and2.13e-14. Independent algebra review passed; native memory and learned-rank claims remain unestablished. Source/results: `scripts/check_inverse_cholesky_headroom.py`, `results/inverse-cholesky-headroom.json`. Command44.13 wall /46.98 CPU seconds.

### Complementary principal-core multipreconditioning

A separate batching gate uses two disjoint272-coordinate cores, selected by alternating the first544 farthest-point indices. Each defines an SPD block-elimination preconditioner. MPCG(1) forms two directions from the current residual, conjugates only against the preceding block, and takes a local Galerkin step. It is compared with their arithmetic mean and a single384-core preconditioner. The two exact packed factors require594048 bytes versus591360 for the single factor; full solver memory remains unaudited. MPCG is established prior work, not a new algorithm claim.

| Case | Average steps | MPCG(1) steps | Single384 steps |
|---|---:|---:|---:|
| Smooth2048 |180|223|88|
| Mixed2048 |197|230|78|
| Concrete824 |267|446|50|
| Airfoil1202 |131|349|29|

All12 endpoints meet original relative residual1e-8. Even granting one kernel traversal per direction block and symmetry-fused dual crosspasses, MPCG(1) loses the average in all four cases; both dual modes lose the larger single core. Independent review found no recurrence/count defect. This closes the specified truncated policy, not all multipreconditioning. No native implementation follows. Source/results: `scripts/check_complementary_core_mpcg.py`, `results/complementary-core-mpcg.json`. Successfulrun10.60 wall /13.24 CPU seconds; initial compile-only syntax failure0.31 wall /0.08 CPU seconds is preserved as `results/complementary-core-mpcg-initial-source.txt`. Total charge10.91 wall seconds.

### Full-history complementary-core control

To separate the two-core representation from last-block truncation, full-history MPCG retains every A-normalized direction and reorthogonalizes twice against all prior blocks. Storage and orthogonalization work are granted free. All four runs converge in104/110/92/49 blocks, still more than the single384 control's88/78/50/29. Optimistic kernel-entry counts also remain higher in every case. Retained history alone occupies6815744/7208960/2425856/1884736 bytes; A-orthogonality error is at most1.87e-9 and no direction is dropped. No feasible-history implementation follows. Source/results: `scripts/check_complementary_core_full_history.py`, `results/complementary-core-mpcg-full-history.json`. Command3.60 wall /6.69 CPU seconds.

An independent comparison audit confirmed that this rejection does not require a per-case best-static oracle: one single384 control wins in all four cases. Likewise one fixed H2-prefix control already beats the earlier energy selector on all four. Query counts remain development cost metrics, not native-time lower bounds. Independent mechanism review did not identify a qualified new route from the current evidence; the scientific goal remains active and unachieved.

### Full-coverage aggregation

All coordinates are partitioned into 200 balanced geometric groups. Each retains its mean and one contrast, giving coarse dimension 400. After ten iterations, the contrast is replaced by the centered current original residual or last full-iterate displacement. Exact transformations, construction and refactorization are free diagnostic resources.

| Case | Static uninterrupted | Unchanged restart | Current residual | Last displacement |
|---|---:|---:|---:|---:|
| Smooth2048 |111|114|127|126|
| Mixed2048 |101|103|115|108|
| Concrete824 |81|86|148|131|
| Airfoil1202 |55|59|71|75|

All 24 solves, including initial-target and exact-error controls, meet original relative residual 1e-8. The exact remaining-error oracle needs zero further fine iterations: full coverage can represent that error, but obtaining it is not implemented. Raw residual and displacement updates lose unchanged restart in all four cases. Tiny orthogonality and transformed-matrix errors are 1.71e-15 and 1.69e-16. This closes the two tested update policies, not adaptive aggregation in general. Native cost and peak memory remain unestablished. Source/result: `scripts/check_full_aggregation_schur.py`, `results/full-aggregation-schur.json`. Command 11.32 wall / 13.56 CPU seconds.

### Preconditioned residual for full-coverage aggregation

Before local centering, apply the old block inverse to the current original residual: u=C^-1 r_c, v=(r_f-Bᵀu)/lambda, z_c=u-C^-1 Bv, z_f=v. At the reconstructed checkpoint r_c is approximately zero; this adds a coarse harmonic extension absent from the raw residual. Each group then replaces its contrast by its centered restriction of z. Group means, checkpoint, coarse dimension and stopping rule are unchanged.

Totals for Smooth/Mixed/Concrete/Airfoil are118/109/109/62, improving the raw-residual update's127/115/148/71 but losing unchanged restart's114/103/86/59. All16 endpoints converge at original relative residual1e-8. Four uninterrupted and eight repeated branch controls match previous solution hashes. The explicit tiny block-inverse check passes; independent algebra review confirms the update is not redundant with raw residual. Dense construction and rebuilding remain free, and native memory/cost feasibility is not established. This particular update is not advanced. Source/result: `scripts/check_preconditioned_aggregation_schur.py`, `results/preconditioned-aggregation-schur.json`. Command8.70 wall /11.10 CPU seconds.

The error-projection diagnostic also separates capture from convergence: on Concrete, raw-residual aggregation leaves46.0% of the Euclidean remaining-error norm outside the coarse space, versus66.0% for preconditioned-residual aggregation, yet takes148 rather than109 iterations. Smooth and Airfoil show the same ordering. Better Euclidean error capture alone is therefore insufficient for selecting these replacements.

### Preserving the dominant local mode during aggregation updates

This gate uses133 groups and coarse dimension399. The static space keeps each group's mean and two dominant centered local-kernel eigenvectors. At iteration10, only the second contrast is replaced, preserving the mean and dominant eigenvector. Raw and old-block-preconditioned residuals are projected orthogonal to those retained directions.

| Case | Static uninterrupted | Unchanged restart | Raw residual | Preconditioned residual |
|---|---:|---:|---:|---:|
| Smooth2048 |106|111|118|110|
| Mixed2048 |98|100|107|105|
| Concrete824 |80|85|108|107|
| Airfoil1202 |50|53|60|59|

All20 solves meet original relative residual1e-8. Local retained-space projector defects are below1e-12; the exact-error oracle needs zero further fine iterations. Preserving the dominant local mode does not resolve the deterioration across these cases. The single iteration saved against restarted Smooth is insufficient to advance the method; construction and rebuilding are still free, and equal coarse dimension is not a full working-memory comparison. Source/result: `scripts/check_preserved_mode_aggregation_schur.py`, `results/preserved-mode-aggregation-schur.json`. Command10.16 wall /11.30 CPU seconds.

Independent source/result review found no rotation, indexing, sign or projection defect. All updated branches lose the uninterrupted static control; no group-size or checkpoint sweep follows this tested policy.

### Large local blocks with free factor recomputation

A separate capacity check applies exact original-system block Jacobi with balanced geometric groups of at most320 coordinates. A proposed sequential implementation would reconstruct one factor at a time; the diagnostic caches all factors and grants construction and application work free. This differs from the earlier small retained blocks in the Schur complement. It does not implement BLOKS, which targets another system.

| Case | Candidate stop | Original relative residual | Exact416 Schur iterations |
|---|---|---:|---:|
| Smooth2048 |CPU cap,1503 steps|0.021285|73|
| Mixed2048 |CPU cap,1339 steps|0.015460|66|
| Concrete824 |2000 steps|0.012016|38|
| Airfoil1202 |2000 steps|0.122508|27|

All four controls satisfy1e-8; no candidate does. Recorded candidate entry counts are work spent before stopping, not completed-solve costs. Diagnostic CPU caps include literal full-residual checks, which are excluded from the analytical action count except the final check. No native timing or peak-memory conclusion follows. The tested local-block representation is not advanced to selective recomputation. Source/result: `scripts/check_streamed_block_jacobi_headroom.py`, `results/streamed-block-jacobi-headroom.json`. Command17.88 wall /21.47 CPU seconds.

Independent audit confirmed the solver and comparison. Even granting free reuse of diagonal blocks in each matrix action, already-spent analytical entries exceed completed-control entries by18.02/17.72/43.82/58.58 times. This does not bound other partitions, overlapping blocks, residual-selected subsets or multiplicative sweeps. The component memory planner is incomplete and is not an implementation peak-memory bound.

### Same-matrix native representation comparison

To test whether current-solve information might select between strong static representations, keep X and kernel bandwidth bitwise identical and vary smooth/random/mixed right-hand sides at n2048, lambda1e-5, seed0. Two repetitions reverse method order and include native setup, iterations and original-residual corrections. The specialized FIRST8 helper is used.

| RHS | Exact416 mean seconds | FIRST8_H1 mean seconds | Exact / FIRST8 |
|---|---:|---:|---:|
| smooth |2.7669|1.9827|1.396|
| random |3.3525|2.6163|1.281|
| mixed |2.5908|1.7991|1.440|

All12 solves meet original relative residual1e-8; FIRST8 wins every paired repetition. No RHS-dependent winner reversal appears, so this result does not justify adding a solve-informed selector between these representations. It extends the static compressed-core evidence to the large random RHS without changing the original evaluation scope. Timing repetitions are not independent dataset seeds; confirmation seeds remain untouched. Existing solver allocation policies are unchanged; no new whole-solve allocation audit was run. Source/result: `scripts/check_same_matrix_representation_frontier.py`, `results/same-matrix-representation-frontier.json`. Command33.67 wall /35.74 CPU seconds.

### Residual-weighted Fourier-feature resampling

This gate changes representation to P=lambda*I+ZZᵀ with416 random Fourier features. At step10, sample from832 candidate frequencies using either uniform probabilities, initial-target correlations or current-residual correlations. Weighted distributions mix50% uniform mass; importance weights preserve the finite-pool covariance in expectation. All resampling controls use identical categorical random numbers and the same starting iterate. Setup, scoring and rebuilding are free diagnostic work.

| Case | Unchanged restart | Uniform resampling | Current-residual resampling |
|---|---:|---:|---:|
| Smooth2048 |562|CPU cap|CPU cap|
| Mixed2048 |585|CPU cap|CPU cap|
| Concrete824 |306|428|477|
| Airfoil1202 |241|274|275|

Counts include the initial10 steps. All six synthetic resampling branches, including initial-target controls, hit the CPU cap; their iteration counts are not convergence costs. The two real-data current-residual branches lose matched uniform resampling. Independent review verifies the Gaussian frequency scaling, Woodbury inverse and importance normalization. Resampling can duplicate frequencies, so comparison only to the original non-resampled416 features would confound feedback and resampling. This specified policy is not advanced. No native memory or timing claim is made. Source/result: `scripts/check_residual_fourier_preconditioning.py`, `results/residual-fourier-preconditioning.json`. Command35.45 wall /37.86 CPU seconds.

### Exact original-system PCG with compressed block inverse

Using the compressed core only in a fixed SPD preconditioner, while retaining exact original-system products, removes the approximate-Schur outer-correction issue. With H equal to the stored core inverse, apply u=Hr_S, v=(r_R-Bᵀu)/lambda, z_S=u-HBv, z_R=v. A one-step-refined H variant is also SPD in all four cases, as checked by the whitened core spectrum.

All eight solves converge. Both H variants take42/37/18/21 iterations on Smooth/Mixed/Concrete/Airfoil. Direct H0 application costs1.364/1.325/1.026/1.052 times the saved best static Schur analytical entry count; H1 adds core products without reducing iterations. Independent algebra/accounting review passed. This is a dense cost diagnostic, not a native memory/runtime result.

An exact congruence formulation may avoid the extra cross passes: T=[[I,-HB],[0,I]], G=TᵀAT. For transformed direction(w,v), form t=Bv,d=Dv, a=C(w-Ht)+t, and the lower result d+Bᵀ(w-Ht-Ha). H0 then needs one C product plus the two cross passes and D product, totaling n² kernel entries. This fusion is derived but not implemented or validated yet; no speedup or novelty is claimed.

Source/result: `scripts/check_exact_outer_mixed_core.py`, `results/exact-outer-mixed-core.json`. Command2.41 wall /5.72 CPU seconds.

### Exact congruence fusion validated

The fused action is implemented in a dense diagnostic. All four original-system residuals reach1e-8 in42/37/18/21 iterations. Analytical kernel-entry ratios against the saved strongest compressed-Schur controls are0.98495/0.95871/0.76693/0.73459. Initial RHS transformation, final solution reconstruction and final original verification are charged; intermediate diagnostic reconstructions and verifications are free. These comparisons do not establish native speed or superiority to exact416 on real data.

A tiny-check assertion originally required the full floating-point trajectories to match within1e-6. It failed at1.435e-4 late in CG, while fused action and RHS errors were2.33e-16 and1.38e-16. The original source is retained. The corrected check retains action/RHS identities, early-iterate agreement, independent dense-reference endpoint checks, and original-residual acceptance. Independent review confirms that late floating-point trajectory equality does not follow from exact-arithmetic congruence.

The next step is native implementation and a fresh memory audit: full-length vectors and extra core workspaces cannot inherit the old Schur allocation certificate. The naive recurrence also uses three core inverse actions per iteration. Maintaining the preconditioned top residual could remove one algebraically, but drift and refresh costs require separate verification. This is a promising implementation opportunity, not current-solve-informed novelty or goal achievement. Source/result: `scripts/check_fused_congruence_core.py`, `results/fused-congruence-core.json`. Combined command charge3.89 wall /12.95 CPU seconds including failure and diagnosis.

### Native fused congruence and repeated timings

The streamed implementation computes the exact transformed operator without retaining dense kernel or cross blocks. Suspected transformed residuals trigger paid reconstruction and literal original residual verification. Full-solve traced peaks are997515B(smooth) and985972B(Airfoil); the largest named-array peak plus64KiB reserve among the pilot rows is1042936B, below1MiB. Immutable X,b are excluded consistently.

All36 timed runs converge at original relative residual1e-8. Three repetitions change method order; mean complete times are:

| Case | Fused H0 seconds | Best other control | Control seconds | Fused / control |
|---|---:|---|---:|---:|
| smooth |1.9119|FIRST8_H1|2.0352|0.939|
| mixed |1.7070|FIRST8_H1|1.8427|0.926|
| Concrete |0.1873|original_exact416|0.2041|0.918|
| Airfoil |0.3456|original_exact416|0.3302|1.046|

The fused implementation wins all paired repetitions on smooth, mixed and Concrete, and loses all Airfoil repetitions to exact416. These are repeated timings on seed0, not independent dataset confirmation. The method remains static; neither adaptive novelty nor the broad research goal is established. It becomes a stronger control for subsequent method development.

Independent review found no blocking algebra or allocation defect on observed paths. Initial simultaneous z/p allocation lacks a separate named observation but is below the measured peak and covered by tracemalloc. Two untriggered stop-report paths require repair before further source development: stale residual after failed verification followed by a cap, and successful fallback verification not updating status. Every observed candidate verifies once and succeeds, so these issues do not change the reported rows.

Sources/results: `scripts/check_native_fused_congruence.py`, `scripts/check_native_fused_repeats.py`, `results/native-fused-congruence.json`, `results/native-fused-congruence-repeats.json`. Combined53.16 wall /57.75 CPU seconds.

### Terminal repair and residual recurrence

The repaired solver is `scripts/check_native_fused_residual_recurrence.py` with `recurrent=False`. Prior sources remain unchanged. Non-successful termination now verifies the current iterate and updates convergence if that check passes. A CPU-cap-zero check returns the actual zero iterate with residual1. Initial simultaneous z/p allocation is explicitly observed.

Optional recurrence reuses Ha to update Hr_top, reducing triangular solves but retaining an extra live vector. All16 timing runs and two allocation audits converge. Mixed requires39 rather than38 iterations. Two order-reversed repetitions give:

| Case | Fresh Hr seconds | Recurrent Hr seconds | Recurrent / fresh |
|---|---:|---:|---:|
| smooth |1.9160|1.9956|1.042|
| mixed |1.7084|1.7871|1.046|
| Concrete |0.1847|0.1755|0.950|
| Airfoil |0.3497|0.3429|0.981|

Both synthetic comparisons lose consistently; real-case ordering changes between repetitions. The recurrence is not promoted. Independent algebra, memory and stop-path review passed. This isolates arithmetic reuse, without a new exact416 comparison or adaptive-success claim. Source/result: `scripts/check_fused_recurrence_pilot.py`, `results/fused-residual-recurrence.json`. Command22.29 wall /25.72 CPU seconds.

### Uniform FP32 factors with exact congruence

The candidate adopts the existing packed FP32 core600 factor and applies its inverse directly inside the exact congruence operator. It never invokes the adopted Schur object's refinement method. This tests whether the earlier promoted FP64 columns are still useful after the approximate-Schur error mechanism has been removed. FIRST8 core592 fusion and exact416 are contemporaneous controls.

All24 timed solves and two allocation audits converge at original relative residual1e-8. Two order-reversed repetitions give:

| Case | Uniform600 seconds | FIRST8 fused seconds | Exact416 seconds |
|---|---:|---:|---:|
| smooth |1.7977|1.8775|2.8313|
| mixed |1.6994|1.6932|2.6610|
| Concrete |0.1756|0.1824|0.2125|
| Airfoil |0.3538|0.3380|0.3210|

Only Smooth consistently improves on FIRST8 fusion. Mixed does not improve consistently; Concrete and Airfoil ordering varies, and exact416 remains faster on Airfoil. No global replacement or solve-informed precision policy is promoted. Source/result: `scripts/check_native_uniform_fused.py`, `scripts/check_uniform_fused_pilot.py`, `results/native-uniform-fused.json`. Command34.85 wall /36.93 CPU seconds.

Independent audit confirms direct unrefined factor application and temporary-array release. The runner omitted the FIRST8 source dependency from its manifest; its unchanged hash matches the earlier native-fused manifest and is recorded as supplemental provenance in `docs/method-revision.json`.

### Current-direction omission of exact core products

In the fixed congruence operator, replacing Cq by Ltilde Ltildeᵀq gives transformed error[e,−BᵀHe], with e=(C−Ltilde Ltildeᵀ)q. Mapping back to original residual coordinates cancels the cross term, leaving[e,0]. The inexact recurrence can therefore track a truncation budget through sum|alpha| times a bound on||e||. The diagnostic compares exact actions, an exact-direction-error selector, and an offline exact Frobenius-norm bound times||q||.

All12 solves reach original residual1e-8. The Frobenius rule preserves iteration counts42/37/18/21 and omits4/4/2/3 core products. Its analytical entry ratios against exact fused actions are:

- smooth: 0.992359; omitted steps [31, 34, 38, 42].
- mixed: 0.991373; omitted steps [28, 31, 34, 36].
- Concrete: 0.949350; omitted steps [15, 18].
- Airfoil: 0.968517; omitted steps [19, 20, 21].

The direction-error oracle adds an iteration on smooth and is not optimal over resulting trajectories. All selector, factor-product and offline-error construction costs are free in this diagnostic. Independent algebra review passes the error mapping and sign. Observed floating-point gap defects reach1.30e-9, material beside the2.5e-9 model budget, so this is an exact-arithmetic truncation argument plus empirical original-residual acceptance, not a full floating-point certificate. A practical streamed error estimate and complete native cost comparison remain necessary. Source/result: `scripts/check_fused_core_omission.py`, `results/fused-core-omission.json`. Command3.36 wall /6.08 CPU seconds.

### Packed factor multiplication and streamed error estimate

`first8_factor_product.c` applies the stored factor product directly with two m-vector workspaces. Relative differences from independent dense multiplication are9.3e-16–1.22e-15. Five batches of20 calls give median0.165–0.170ms, versus2.776–3.151ms for streamed exact core products. These are primitive timings without full solver vectors, not end-to-end speedups.

The column-defect norms already computed during factorization yield a rounding-aware estimate for||C-LtildeLtildeᵀ||F. Independent derivation and assumptions are recorded in the literature notes. Replacing the offline exact norm with this estimate gives3/3/2/3 omitted actions on Smooth/Mixed/Concrete/Airfoil, with unchanged42/37/18/21 iterations. All eight exact/control trajectories meet original1e-8. Dense error matrices are used only for diagnostic validation, never selection. Native selector and workspace overhead still require integration.

Sources/results: `scripts/first8_factor_product.c`, `scripts/check_factor_product_cost.py`, `scripts/check_fused_core_omission_estimate.py`, `results/factor-product-cost.json`, `results/fused-core-omission-streamed-estimate.json`. Combined7.35 wall /11.88 CPU seconds.

### Native omission result and accumulated core-error correction

The native cumulative absolute-error selector does not improve complete runtime. All16 timings and two allocation audits converge, but each case has a slower candidate mean; Mixed needs39 rather than38 iterations. Independent code review found no blocking algebra, memory or charging defect. This selector is closed.

| Case | Exact core seconds | Adaptive seconds | Ratio |
|---|---:|---:|---:|
|smooth|1.857762|1.862303|1.0024|
|mixed|1.658851|1.709913|1.0308|
|Concrete|0.183189|0.191123|1.0433|
|Airfoil|0.334526|0.339545|1.0150|

A changed mechanism accumulates s=sum alpha*q across cheap products and evaluates E*s only at refresh, correcting the transformed residual by[-Es,+B^T H Es]. This retains signed cancellation. Four-case dense trajectories compare exact actions, fixed eight-step refresh, a current-residual trigger, and restarting at that trigger. All16 converge. Fixed and adaptive refresh without restart reduce counted kernel entries on all four cases; fixed timing is better on Concrete and ties Mixed. Restart loses on both synthetic cases and Airfoil. Selection, factor-product arithmetic and per-step true-residual diagnostics are free in this screen. Continuing directions after replacement is a heuristic recurrence, not exact CG. Native memory and complete-cost checks remain necessary.

Evidence: `results/native-adaptive-fused.json` (21.92wall/25.17CPU seconds), `results/accumulated-core-error-refresh.json` (3.74wall/6.74CPU seconds). Scientific success remains unachieved.

### Native accumulated residual replacement

Two native implementations compare adaptive refresh, fixed8, exact FIRST8 fusion and exact416. Each completes32 timed solves and two allocation audits; all converge. The second uses H=Ctilde^-1 to cancel the approximate congruence cross blocks, removing one core inverse application per action. Returned q and the accumulated core vector remain explicit and charged. Final original residual checks are paid.

Means for the simplified implementation:

|Case|Adaptive seconds|Fixed8 seconds|Exact FIRST8 seconds|Exact416 seconds|
|---|---:|---:|---:|---:|
|smooth|1.819603|1.824810|1.865926|2.792857|
|mixed|1.670276|1.735103|1.707121|2.606676|
|Concrete|0.150859|0.146654|0.179149|0.202950|
|Airfoil|0.324019|0.326534|0.340610|0.325314|

Concrete has the clearest complete-cost gain. Smaller improvements remain sensitive to timing order; Airfoil is nearly tied with exact416. Adaptive timing has not established an advantage over fixed8, and uniform600 remains a relevant stronger Smooth control. No broad-scope or novelty claim follows. Next compare terminal-only refresh without threshold tuning, then reassess development-scope expansion.

Evidence: `results/native-accumulated-refresh.json` (45.00wall/45.72CPU seconds), `results/native-accumulated-refresh-diagonal.json` (43.73wall/45.42CPU seconds). Independent source audits passed; ideal gap repair does not certify full floating-point residual error.

### Terminal refresh and stronger representation controls

Terminal-only refresh with continued directions fails all four first-repeat cases at the7CPU-second limit. Its two traced runs also reach their8CPU-second caps. The command stops at its CPU gate after24rows, not the intended40; capped times are not convergence times. The late residual correction invalidates the old direction relationships and can amplify the Fletcher–Reeves coefficient.

Resetting p=z at each terminal refresh restores convergence in all four cases, but takes69/61/32/43 iterations versus adaptive44/40/19/23 and is slower in each case. All16 timed rows and two audits in this matched control converge. Contemporaneous uniform600 is faster than adaptive on both synthetic cases, so previous small improvements against FIRST8 are not improvements against the strongest tested representation. Concrete gains remain; Airfoil is near tied.

Sources/results: `results/native-terminal-refresh.json` (73.56wall/76.11CPU seconds, CPU-gated), `results/native-terminal-restart.json` (28.43wall/30.62CPU seconds, complete). No scientific success claimed.

### Factor precision and retained capacity

Three dense screens compare721200bytes of intended factor payload. FP16 with first8 columns/diagonal FP64 supports core822, and column-scaled int16 supports818. Both fail sequential quantized construction on every case; postquantized factors also lose badly against core600 FP32. The same-rank FP64 oracles confirm larger cores could help if accuracy were retained, but exceed the storage allowance.

Scaled packed24bit values support core675 with720045payload bytes, including FP64 first8 columns, diagonals and all column scale slots. Sequential quantized construction succeeds on all four cases:

|Case|Postquantized FP32 core600|Sequential scaled24bit core675|Exact core675 oracle|
|---|---:|---:|---:|
|smooth|42|35|35|
|mixed|40|31|30|
|Concrete|18|14|14|
|Airfoil|23|19|19|

All endpoints in the24bit screen satisfy original relative residual1e-8. Dense buffers and factorization, per-step original checks, and decoded factors are free. The code uses int32 to represent quantized values during this screen;3byte packing is planned, not implemented. This is static capacity headroom, not native speed, total-memory feasibility or solve-informed success. Next implement packed triangular operations and measure full costs.

Evidence: `results/half-core-capacity.json` (16.14wall/18.20CPU seconds), `results/int16-core-capacity.json` (11.62wall/14.88CPU seconds), `results/int24-core-capacity.json` (3.84wall/7.05CPU seconds).

### Actual packed24 factor and complete solver costs

The implementation physically stores signed values in three bytes and performs factorization, triangular solves and factor products without decoding a full matrix. Actual factor storage720109B includes64B of duplicated first8 diagonals. Signed/zero-scale cases and the independently decoded675factor pass solve/product checks at about2e-15 relative difference. Exactfusion traces998217/987381B fit1MiB; all24 timings converge. Synthetic means improve relative to uniform600, while Concrete and Airfoil lose after full setup/decoding costs.

Combining this representation with accumulated residual correction gives40timings:38converge, while fixed8 Concrete reaches its CPU cap twice. Adaptive refresh converges but is slower than matched exactint24 on all four case means, with nearly every step refreshed on the real cases. This combination is not promoted. The packed representation remains a static control, not an adaptive research result.

Evidence: `results/int24-native-primitives.json` (2.75wall/4.46CPU seconds), `results/native-int24-fused.json` (34.14wall/36.26CPU seconds), `results/native-int24-refresh.json` (64.07wall/67.39CPU seconds). Independent source audit passed exactfusion; supplemental transitive source provenance is recorded in the ledger.

### Solve-informed precision promotion screens

At core660, a shared four-step prefix precedes promotion of selected packed24 columns to FP64 under721200bytes including flags/offsets. Core inverse-error scores and then full block-preconditioner derivative scores compare current residual, initial RHS, matrix-only allocation and unchanged restart. All48 exact-action trajectories converge. Current selection does not beat the stronger upfront controls. Core-only raw restarted counts duplicate setup; the ledger records a derived correction removing that duplicate while preserving prefix and boundary costs.

A further20-run screen tests the intended cheap-action benefit using exact E*s for a free oracle refresh trigger. All converge. Current masks tie matrix-only on Smooth countedentries and lose on the other three cases. Exactreference factors, selection and regeneration are free, so a native implementation is not justified by these results. This closes the tested first-order promotion family; it does not establish an impossibility result for all adaptive representations.

Evidence: `results/int24-precision-selection.json` (5.21wall/8.10CPU), `results/int24-coupled-precision-selection.json` (5.54wall/8.71CPU), `results/int24-promoted-refresh.json` (4.90wall/7.86CPU seconds). Core and coupled derivative/budget audits passed. Goal remains active and scientifically unachieved.

### Residual and last-direction repair

The next discriminator corrected both the accumulated residual gap and the preceding operator image, then performed an exact line correction and one-direction conjugation. All20 dense solves converge. Terminal repair takes68/56/33/41 iterations, versus68/56/32/41 for restart. Periodic repair takes43/38/21/24, identical to the unchanged periodic recurrence. Even sharing one two-RHS core/cross query pass with free extra arithmetic/storage yields no improvement. The last-direction repair is not promoted; larger history is not justified by this result or the prior history gates.

Evidence: `results/core-direction-repair.json`;4.21wall/7.17CPU seconds. This restores only local orthogonality/conjugacy, not full CG history. Goal remains unachieved.

### Same-matrix packed representation comparison

The new packed24 representation was checked against uniform600 on identical X/bandwidth and smooth, random and mixed right-hand sides. All12 native timed solves converge. Packed24 wins both order-reversed comparisons for all three targets, so this screen provides no reason to build a solve-informed representation selector. This extends the static observation to the random target without establishing adaptive gains.

smooth: packed24/uniform600 mean seconds ratio 0.915576.
random: packed24/uniform600 mean seconds ratio 0.911689.
mixed: packed24/uniform600 mean seconds ratio 0.926042.

Evidence: `results/same-matrix-packed24-frontier.json`;23.95wall/26.30CPU seconds. Goal remains active and unachieved.

### Solve-conditioned floor/ceil rounding

A different operation keeps int16 scales and factor payload fixed while changing rounding choices to reduce two factor-action losses on initial or current solve vectors. A shared four-step prefix and nearest-rounding restart control isolate calibration. Exact reference factor/calibration/regeneration costs are free in this dense gate.

The first version lacked a code-range guard. Exact replay found values±32768 in all eight calibrated factors; the positive endpoint is not representable in signedint16. `results/calibrated-factor-rounding.json` is invalid as evidence for the claimed representation. The range audit is retained in `results/calibrated-factor-rounding-range-audit.json`.

The corrected implementation restricts all candidates to±32767 and checks every final code. In the full rerun12/16solves converge; four calibrated synthetic runs hit CPU caps. Current calibration takes37/448 iterations on Concrete/Airfoil versus33/337 for nearest rounding and18/23 for the FP32control. Its calibration loss falls about96percent without solver improvement. This closes the tested objective, not all target-dependent quantization.

Valid evidence: `results/calibrated-factor-rounding-bounded.json`. Commands35.08+2.10+35.15wall seconds, including invalid run and repair. No scientific success claimed.

### Locating the quantization failure

Reconstructed bounded-rounding factors reproduce the original calibration losses and flip counts. Nearest-int16 core generalized condition numbers are only1.049–1.103, but inverse actions on the coupling B*r_tail can have relative Euclidean errors above1. Current calibration increases global inverse-factor error despite reducing the measured factor-action loss. These are core diagnostics, not full-solver condition-number claims.

Using one fixed refinement J=2H−HCH only in the congruence transformation, while retaining H in the block preconditioner, gives36/34/11/30iterations. All8 refined/exact-transform-oracle solves converge. Charging the two additional core products per refined action makes all four cases more expensive in kernel entries than the FP32core600 control. Exact coupling with the compressed core inverse is better but is only a free oracle. This motivates testing whether coupling error can be cached cheaply, not another scalar tolerance sweep.

Evidence: `results/calibrated-rounding-inverse-diagnosis.json` (4.25wall/6.96CPU seconds), `results/refined-int16-congruence.json` (5.15wall/6.68CPU seconds). Scientific success remains unachieved.

### Retained low-rank coupling-error cache

A rank8 correction to C^-1B−HB changes the congruence map while retaining an exact SPD transformed operator. Raw SVD and C-energy-weighted SVD are granted free. Core sizes737/737/786/771 leave room for8*n*8bytes of cache within721200bytes of retained factor/cache payload. This is not a native total-memory proof.

All16 solves converge. Weighted rank8 improves the corresponding uncorrected347/345/131/341 iterations to86/82/33/69, but the matched prior FP32core600 control takes42/40/18/23. Countedkernel work still loses on all four cases before charging SVD, cache arithmetic or reconstruction. Full accurate coupling helps further but exceeds the representation allowance. No adaptive selector or nativeimplementation follows from this gate.

Evidence: `results/cached-coupling-error.json`;15.53wall/17.36CPU seconds. Independent algebra/payload audit passed. Scientific success remains unachieved.

### Fresh spatial splitting of the current preconditioned residual

The fixed-width-two gate compares a fresh split after preconditioning with scalar PCG and scalar full-history Galerkin on an identical postquantized FP32 core600 and exact congruence operator. All history storage, orthogonalization and intermediate verification are free for this diagnostic. It is inspired by enlarged CG and is not a new-method claim or a native memory-feasibility result.

Three valid fresh runs require44/43/19 kernel batches on smooth/mixed/Concrete, versus42/40/18 for PCG and41/39/17 for scalar full history. They meet original relative residual1e-8, with history orthogonality defects below2.4e-9. Thus this policy has no batch-count headroom in these cases even before its extra costs.

Airfoil fresh splitting and all four initial-split full-history controls lose numerical orthogonality severely and hit CPU caps. Those rows are invalid as convergence-cost comparisons for the intended full-history algorithms; Airfoil remains unresolved. An initial failed tiny check and its source are preserved. No further native implementation follows from this screen.

Evidence: `results/current-residual-split.json`, `scripts/check_current_residual_split.py`; total charged command wall46.81seconds including failed validation and diagnosis. Goal remains active and unachieved.

### One frozen coupling interpolation

A rank-two rectangular correction can match the accurate-coupling preconditioner on the current original residual while keeping the same compressed inverse H and preserving SPD. The test freezes this correction after four common iterations, with matched restart, initial-target, static weighted-SVD and reference-action warm-step controls. Both Euclidean and C-energy minimum-norm interpolation were tested and retained separately. Reference acquisition, SVD and construction are free in this optimistic screen.

The C-energy current-target runs all converge in386/372/44/346 iterations (smooth/mixed/Concrete/Airfoil), versus377/377/46/343 unchanged,228/247/27/217 with static rank-two correction, and42/40/18/23 for the stronger FP32core600 control. Euclidean current-target smooth/mixed hit CPU caps; the other two need64/604 iterations. Matching one action therefore did not provide a useful persistent correction in these configurations. No native implementation or scientific success follows.

Evidence: `results/residual-coupling-interpolation.json` and `results/residual-coupling-interpolation-weighted.json`; executed sources and input hashes verified. Two commands cost61.33wall/64.82CPU seconds. Delayed rows have cumulative totals but post-prefix-only `entry_components`; the ledger explains the added prefix and warm-step costs. Raw artifacts remain unchanged.

### Frozen refresh policy: regularization and right-hand-side extension

The unchanged adaptive and periodic8 refresh policies were compared with uniform600, packed24 and exact416 at lambda0.1/0.001/0.00001 on smooth/random/mixed n2048 and Concrete/Airfoil. All150 timed solves and30 separate candidate allocation audits reached original relative residual1e-8. Maximum traced allocation was1,008,032bytes; maximum named storage plus reserve was1,044,136bytes. Originaln512 andheldoutseeds remain outside this extension.

Adaptive/periodic8 geometric mean time ratio is0.996399; adaptive/best-static-per-case ratio is1.043077. The best-per-case comparison is an oracle diagnostic. Summed case mean times are12.47869s adaptive,12.67777s periodic8,12.26366s uniform600,12.47130s packed24 and16.11054s exact416. Adaptive has three small mean wins over the static frontier, but each reverses ordering between the two repetitions. These descriptive timings do not establish a reliable adaptive advantage; the frozen policy is not promoted.

Evidence and command receipts: `results/refresh-scope/`; descriptive analysis: `results/refresh-scope/analysis.json`; unchanged-algorithm wrapper: `scripts/check_refresh_regularization_scope.py`. Benchmarks216.54wall/243.72CPU seconds, analysis3.33wall seconds. Inputs and source hashes verified unchanged. Goal remains active and unachieved.

### Measured accumulated-error indicator

The next gate replaces only the frozen native criterion bound*norm(s) by norm(E*s), using E=C−L L^T decoded from the actual streamed FIRST8factor. All15larger-case combinations converge at original1e-8; originalsource/data identities and allnativequerycounts are verified. DenseE construction/storage andindicator work are free in this query screen; observed times include oracle overhead and do not establish speedup.

The actualindicator reduces entries in5/15cases versusbound and8/15versusperiodic8. Totalentries are1.20508billion versus1.18841billion(bound) and1.20159billion(periodic8). All9synthetic cases losepacked24. Only3cases beat the best existingcontrol, by1.72–2.70percent withfreeassistance. Moreaccuratecurrenterror information canreduce refreshes butadditerations. These results do not justify implementing a sketch for this frozenpolicy.

Evidence: `results/actual-refresh-indicator/analysis.json` and underlying per-case records/command receipts. Charged46.31wall/77.29CPU seconds. No scientific success or fullmemory/nativeperformance claim.

### Residual-coordinate audit and unchanged-threshold test

An observer-only replay exactly reproduced four existing hard-case trajectories, including every indicator value, refresh list, kernel count and final residual. The original-coordinate residual is [rS−E*s, rR+B^T H rS]; the small SPD identity check has relativeerror2.59e-16. Its norm would defer some refresh decisions on smooth, mixed andAirfoil; Concrete is unchanged.

Running that one changed denominator with the same0.1threshold and recurrence gives49/41/20/24 iterations. Smooth kernelwork rises4.04percent relative to the previous measured-error trigger, while the remaining three have identical work. All converge at original1e-8; all dense reference/mapping work is free. Therefore no15-case expansion or native tracking implementation follows.

Evidence: `results/refresh-residual-coordinates.json` and `results/original-coordinate-refresh.json`. Commands7.93+6.65wall seconds. Source hashes verified; goal remains active and unachieved.

### Shared kernel accounting improvement

Profiling four unchanged native solves located substantial repeated per-tile allocation-accounting cost; the profiler fractions are not predicted speedups. A common SharedKernel.product variant now checks the first/largest tile only, leaving row selection and all numerical operations unchanged. Within each product, retained arrays and scopes are constant and subsequent tile allocations cannot grow. The original source is preserved; `scripts/check_kernel_accounting_hoist.py` provides a scoped variant applied equally to all five methods.

The two-case pilot completed40timings and10separatewholetraceaudits. Everyold/newpair exactlymatchesiterations, refreshes, kernelcounts, finalresidual, namedmemorypeaks andtileplans. Maximumtrace1,008,352bytes. Smoothmean runtime ratios(new/old) are0.9182adaptive,0.9180periodic8,0.9448uniform600,0.9609packed24 and0.9982exact416. Concretechangesare smaller/mixed withtwo repeats. Packed24remains fastestSmooth(1.6106s versus1.6610s adaptive). Retainthiscommonimplementationimprovement, notascientificadaptivecontribution. Futurecomparisons mustuseit equallyforcandidateandcontrols.

Evidence: `results/kernel-solver-cost-profile.json` and `results/kernel-accounting-hoist/analysis.json`. Commands6.71+61.15wallseconds. Source/inputhashesverified; oldmeasurementsremainoldimplementationevidence. Goal remains active and unachieved.

### Larger-case comparison with common accounting optimization

All 150 timed solves and 75 separate allocation audits pass across 15 dataset/regularization combinations. Thirteen new cases cost 215.20 command-wall seconds; two pilot cases were reused without rerunning or charging again. Maximum traced allocation is 1,008,352 bytes. Every method uses the same kernel accounting optimization.

Adaptive beats all controls in both repetitions on four cases: Airfoil at all three shifts and mixed at lambda0.1. Its geometric mean time ratio versus the per-case best static control is 0.993168; versus periodic8 it is 0.983632. Hard smooth/random/mixed still lose to packed24 by 3.13/12.22/5.51 percent. Two repetitions establish descriptive observations, not reliable broad improvement. Original n512 and heldout seeds remain untested. Scientific success remains unachieved.

Evidence: `results/refresh-scope-hoisted/analysis.json`; protocol corrections and trace scope: `execution-audit.json` in the same directory. Next candidate: line correction along the whole displacement since the previous refresh, followed by a conjugate projection of the next direction. Algebra checked; performance and additional memory costs not yet tested.

### Whole-phase displacement correction

At each residual refresh, the exact transformed image of the displacement since the preceding refresh can be recovered from the two endpoint residuals. The resulting line minimization and projection of the next search direction pass the algebra checks. Nevertheless, periodic full-phase repair takes 53/48/21/25 iterations on smooth/mixed/Concrete/Airfoil, compared with 43/38/21/24 for unchanged periodic FR. Terminal phase repair also never beats the last-direction control.

A separate line-only ablation retains the inherited FR direction and restores exactly 43/38/21/24 iterations and the unchanged kernel counts. Thus removing the direction replacement eliminates the deterioration, but the line correction itself provides no work reduction in this screen. Extra vectors and arithmetic were free, so neither tested variant warrants native implementation. No broad impossibility claim follows.

Evidence: `results/core-phase-direction-repair.json` and `results/core-phase-line-correction.json`; source/data hashes and replay controls verified. Commands cost 19.46+2.13 wall seconds. Goal remains active and unachieved.

### Gaussian feature-moment acquisition screen

Read-only observation reproduces the strong packed24 baseline trajectories exactly (36/33/14/20 iterations). The screen evaluates 216 product approximations across first/middle/last vectors, three fused product types, global versus four source centers, and degrees 2/3/4. A tiny explicit multinomial feature computation agrees with the pairwise polynomial oracle. Captures, dense references and diagnostic work are free; observed baseline timings are not performance measurements.

No sampled four-center product has both relative error at most 1e-4 and fewer feature visits than exact kernel pairs. Global degree3 reaches relative error below 1e-2 with fewer feature visits for all nine sampled products on smooth, mixed and Airfoil; Concrete passes both criteria on two samples. Feature visits and pair counts are different work units, and moment storage is only an incomplete memory lower bound. These observations leave coarse global-product headroom, not a solver improvement. The follow-up measures error amplification through the exact congruence transformation before any new inexact solver is implemented.

Evidence: `results/localized-gaussian-moments.json` and `results/localized-gaussian-moments-analysis.json`. Source/data hashes and unchanged native trajectories checked independently. Command cost 13.09 wall seconds. Local expansions and kernel-machine IFGT already have established prior art; no novelty or scientific success is claimed.

### Propagating global feature error through the solver

Using one common global center over all inputs preserves a symmetric degree3 feature kernel. On first/middle/last exact packed24 search directions from four hard cases, its transformed action error is 1.385–8.175 times the exact action norm. Multiplying this error by the exact-trajectory step length gives 0.460–6.216 times the current recursive residual norm; 11 of 12 exceed one. Offline exact/native action mismatch is at most 4.42e-9, far smaller than these approximation errors.

These measurements reject treating this degree3 expansion as a small-error direct action replacement in the current congruence solver. They do not measure an actual approximate trajectory, whose step lengths and directions could differ, or rule out every Taylor/IFGT or flexible outer method. No native feature implementation follows.

Evidence: `results/global-moment-congruence-error.json` and its audit sidecar. All native baseline work/residual checks and source hashes pass. Command cost 7.29 wall seconds; scientific success remains unachieved.

### Sparse rectangular coupling correction

Unlike the previous rank8 cache, this screen gives each tail column eight independently selected core-coordinate corrections. C-energy OMP refits each support; its projection identity and selected residual correlations pass. Cores755/755/818/799 and coefficient/index payloads fit721200bytes, excluding the disclosed free construction and workspace.

All16 solves converge. Sparse correction takes368/367/30/359 iterations on smooth/mixed/Concrete/Airfoil, versus360/351/29/343 unchanged and38/34/9/20 with free exact coupling. It retains99.92–99.99percent of the C-energy coupling-error norm. Same-core rank8 comparisons exceed the payload allowance in every case and are only overbudget controls. Therefore this coordinate-sparse representation is not advanced to adaptive allocation or native implementation.

Evidence: `results/sparse-coupling-error.json`, executed source and command receipt. Source/data hashes and original residual stopping checked. Charged20.67wall seconds; no scientific success.

### Structural redesign: additive multilevel preconditioning

The principal-core architecture was replaced by leaf block inverses and two levels of sparse local interpolation. Each leaf has at most32 points, four leaves form a middle group, and local bases have at most8 columns. Current preconditioned residuals after four common iterations, with or without three homogeneous relaxation steps, are compared with equally filtered random vectors and unchanged block Jacobi.

None of16 branches reaches original relative residual1e-8. Synthetic branches hit CPU caps after729–1144 iterations; real-data branches reach2000 iterations. Component storage plans fit1MiB, and the small SPD-action check passes. Dense hierarchy construction and exact damping information were free. These observations close this specified additive hierarchy; they do not rule out all multilevel methods. Censored rows are not completed-runtime comparisons. The recorded scalar-RHS total omits coarse-assembly arithmetic and is only an incomplete optimistic proxy.

Evidence: `results/bootstrap-multilevel.json`, executed source and receipt. Charged55.04wallseconds. The structurally different sparse-graph polynomial candidate is being tested next; scientific success remains unachieved.

### Structural redesign: positive graph-polynomial preconditioning

A symmetric eight-neighbor geometric graph replaces the entire principal-core structure. Nine bounded positive Bernstein coefficients define a full-rank degree8 polynomial inverse action. Coefficients minimize current-probe energy using nine original-kernel RHS products; current residual, initial RHS and seeded random probe fits are compared from the same four-step prefix. A one-line current-action warm correction separates a one-shot benefit from persistent preconditioning.

All12 fits pass original-Gram KKT and objective checks with full rank9. Nevertheless, all12 persistent polynomial branches fail to reach original1e-8: synthetic runs are CPU-capped after743–828 iterations, real runs reach2000. Current warm branches also fail; only the two mixed plain-CG controls converge (255/258 iterations). Tiny Horner/eigenbasis equivalence and SPD checks pass. This closes the specified graph, degree and fit objective without a parameter sweep; it is not a claim against every graph preconditioner.

Evidence: `results/graph-polynomial.json`, executed source and command receipt. Independent source/control audit found no blocking defect. Dense construction, fitting and sparse-action arithmetic were free; censored rows are not completed-solve runtime measurements. Charged43.00wallseconds; goal remains active and unachieved.

### Original n512 full-factor reference

A full streamed FP32 packed factor with FP64 residual refinement passes all18 timed solves and9 allocation audits for smooth/random/mixed and lambda0.1/0.001/0.00001. It needs1–3 corrections, with mean setup-inclusive times0.0336–0.0394seconds. Maximum original relative residual is3.46e-9; maximum traced allocation998606bytes and named+reserve1042232bytes. Full FPS ordering remains included in setup. This fills a static-reference gap in the original small-scale scope; it does not validate an adaptive method.

Evidence: `results/full-packed512-refinement.json`, source and receipt. Charged4.50wallseconds.

### Authorized resource-question revision

The user permits changing only the fixed-total-memory clause. Kernel linear systems, current-solve information, CPU execution, complete solve costs, experimental contribution and course requirements remain fixed. The next diagnostic asks whether unknown preconditioner rank supplies a meaningful online investment decision when memory is reported under a common practical ceiling. All methods may cache the kernel; direct Cholesky must be included. Existing negative results remain negative.

### Memory-relaxed static frontier

All methods independently construct/cache dense K under a common128MiB algorithm-owned ceiling, with setup charged. Across24 original-case/regularization combinations,240timings yield224converged,8CPU-capped and8iteration-capped runs. Among successful sampled methods, mean winners are directCholesky13cases, rank32five, rank128four andplainCGtwo. Five separate largest-case traces pass; maximum73,452,457bytes.

The sampled ranks32/128/min(512,n−1) do not show a convincing current-RHS rank-decision opportunity after including direct Cholesky. The clearest consistent rank variation occurs where direct Cholesky is faster; another variation reverses between repeats. Two repetitions are descriptive only. Rank1024 and cached strong Schur controls remain untested, so this is not a global dominance result. Relaxing memory alone is not a demonstrated research mechanism.

Evidence: `results/cached-kernel-static-frontier/analysis.json`, underlying records, audit andcommandreceipts. Charged88.47wallseconds. Original strict-memory results remain valid for their earlier contract; scientific success remains unachieved.


Recycled residual-sketch admission diagnostic (2026-11-24)

`results/recycled-residual-sketch-admission.json` tests the IterativeCUR-inspired reuse of one small residual sketch for four QRCP column-admission rounds, growing FPS64 to96. The current-solve sketch reuses eight paid PCG directions/images; this is one checkpoint, not an online rank schedule. Independent Gaussian, residual-diagonal growth, upfront96, unchanged64 and hindsight final-set controls share the stated diagnostic protocol.

Current/Gaussian post-prefix iterations are431/433 (smooth2048),312/316 (mixed2048),445/439 (Concrete),119/119 (Airfoil), all atlambda1e-5 and original relative residual1e-8. Both delayed branches pay the eight-step prefix; Gaussian additionally acquires eight RHS products. Current-selected columns do not show a consistent quality advantage. Avoid counting three CPU-capped initial64 controls as completed costs. Construction and repeated literal residual checks are free in the analytical work proxy; no full-cost speedup or project success follows. Source/image/PSD checks and independent audit found no blocking defect. Command wall27.53sec charged to existing campaign allowance.


Setup-inclusive follow-up: `results/recycled-admission-cost.json` independently builds cachedK for each solve under the same128MiB diagnostic ceiling. All48timing runs and2separate largest-case candidate traces converge. Current means are0.763845/0.617285/0.118663/0.101495sec versusdirect0.194004/0.184671/0.015105/0.037983sec on smooth/mixed/Concrete/Airfoil. Best staticNyström also beatscurrent inall4cases. Even current continuation-only time exceedsdirect inall4, so removing acquisition/setup cost cannot rescue this implementedconfiguration. Thefixed-sketch batch-admission branch is closed without rank/batch-size tuning. This doesnot refuteIterativeCUR asamatrix-approximation method. Source/accounting auditfoundnoblockingdefect;48timings aredevelopmentcomparisons, notconfirmation evidence. Wall21.26sec charged.


Full-system sparseCholesky capacity gate: `results/sparse-cholesky-capacity.json` preserves exactL diagonal and chooses global offdiagonal entries under a1MiB logical CSR plan. ExactL and inverseactions are free privileged information. All16persistent magnitude/current/initial/random branches fail targetaccuracy on4hardcases. Currentcounts/residuals and completed strongreference kernel-entrycounts are recorded in method-revision.json. Dense triangular solve time contributes todiagnosticCPUcaps, so timeouts are not native sparse-runtime measurements; alreadyspent originalA products establish the observed poor convergence. The specified scalar sparsification selectors are closed, without extrapolation toallincompleteCholesky methods. Independentmath/source audit foundnoblockingdefect. Wall49.33sec charged.


Frozen whitened diagonal metric: `results/whitened-diagonal-metric.json` uses8paidFPS128-PCG pairs to form a positive diagonal metric between two inverse-square-root preconditioner actions. Currentcontinuation iterations609(capped)/512/676/123 versusunchangedrestart289/249/282/55 on smooth/mixed/Concrete/Airfoil. Allmetric fitsvalid; exactJacobi control alsoworse. Independent implementation audit andtinyhalf/inversehalf/Cdiag/SPD checks pass. Thisselector isclosed withoutclipping/schedule tuning; no broadclaim againstdiagonal equilibration. Freeconstruction/applicationarithmetic means no native-costclaim, andcappedcountisnotcompletion. Wall30.17sec charged.


MMF capacity diagnostic: `results/mmf-capacity.json` tests nested2coordinate matrixGram rotations andcore-diagonal preconditioning withlogical1MiB storage. Static/current/initial/random/freeexactremainingerror retirementrules allfail original1e-8 onall4hardcases (20persistentbranches). Current/oraclecounts, residuals and completedstrongreference kernelentrycounts areinmethod-revision.json. Tinyorthogonality/dropaction/SPD/inverse checks andindependent sourceauditpass. Denseconstructionisfree; cappedcountsarenotcompletioncosts andlogicalplannerisnotnativepeakallocation. Specifiedlocalretirementrulesareclosed withoutangle/core-size sweeps; thisdoesnotrefuteMMF generally. Wall59.50sec charged.


Nonsymmetric CUR rowselection: `results/residual-row-cur-gmres.json` fixesFPS64 columns andusesQRCP onresidualweightedC toselectrows, withoutformingAC forselection. All16CUR inverse-action checks pass thedeclared1e-7 validitythreshold; stableQRWoodbury andrightGMRES20 independentlyaudited. Currentselector fails3hardcases andtakes840steps onAirfoil versusmatrix-onlyCUR620,samecolumnNyström580,Nyström128-GMRES73,andPCG12860. Thisrule isclosed without rank/restart/weight tuning. Freeconstruction/verification gate, notnativecost/memoryclaim. GMRES operator-countfields overlaptheinitialproduct byone. Separate `results/cur-collocation-identity.json` verifies24tiny identities andtheone-actionGalerkin capacitycomparison, which isnotanouterGMRES impossibilitytheorem. Wall39.98sec charged.
