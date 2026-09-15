# Scientific implementation audit

Status: in progress. The initial implementation is not approved for the full study.
No model-complexity result has been obtained. The existing v1 generated scenarios
must not be used for experiments; the runner now rejects them.

## Corrected and regression tested

- Every paraphrase includes the proposition; generated text is unique.
- Gettier cells require a correct, accepted proposition and non-guessing support.
  This is a constrained factorial design; impossible factor combinations are excluded.
- Scenario and minimal-pair identifiers form connected components used for splits
  and online coding boundaries. No component is divided across coding blocks.
- First-block marginal probabilities depend only on the disjoint tuning data,
  with Jeffreys smoothing. The whole marginal comparator uses those same
  probabilities. Empirical coding-set entropy is descriptive only. Consequently
  this is a conditional codelength given shared tuning data, an explicit protocol
  clarification from the underspecified empirical-marginal instruction.
- Report eligibility now combines stable-class counts with evaluation macro-F1.
- Confirmatory correlations are withheld because the existing resampling did not
  propagate scenario uncertainty through the four estimators.
- Compression selection uses tuning data, with only the chosen format evaluated
  on the untouched split. A test-label perturbation regression verifies selection
  invariance. SVD nonfactorized tensors are copied to avoid mutable array aliasing.
- The implicit sparse projection has orthonormal columns and full rank at d = D.
  Initialization is fixed across projection seeds; nonpositive improvement makes
  d90 undefined, and only selected dimensions receive test metrics.
- Cached model and tokenizer commit hashes are pinned. Loading is local-only;
  extraction records content token IDs/positions, forbids truncation, and explicitly
  sets padding-independent position IDs. Scenario/config hashes and row ordering
  are verified before downstream use.
- Full-layer escalation uses tuning F1 rather than evaluation F1.

## Remaining work before the full study

- Review natural-language scenario semantics and factor balance. Abstract sentences
  about failed support do not yet constitute a varied bank of concrete Gettier
  mechanisms. Independent-group counts must accompany paraphrase counts.
- Add artifact-byte integrity checks and safe interrupted-stage recovery before
  long GPU runs. Float32 repeatability has been verified on the DGX; bfloat16
  precision sensitivity remains unresolved for the primary configuration.
- Validate LLC on additional regular dimensions and audit the custom sampler
  against a maintained reference. Mixing, basin and frozen-calibration safeguards
  are implemented; actual MLP calibration has not been run because controls fail.
- The common-cohort paired-refit pipeline is implemented and its wiring is tested.
  Confirmatory intervals remain unavailable until eligible data, accepted MLP
  calibration and the complete refit artifacts exist. No full study is authorized
  by the pilot results.

## DGX discovery

Read-only SSH through the existing `spark-remote` Tailscale alias succeeded.
The machine reports an NVIDIA GB10 and 2.9 TB available disk space. No tmux server
was running at inspection; this does not establish that the GPU is idle.
At that initial inspection the pilot had not been launched. Subsequent preflight
and execution evidence is recorded below.

Follow-up discovery: 116 GiB available RAM, no GPU compute processes reported,
PyTorch 2.14.0+cu130, Transformers 5.16.1, CUDA available. All three model snapshots
were located and pinned in configuration. Qwen3-1.7B tokenizes the literal answer
continuations ` A` and ` B` as one token each while preserving the prompt prefix.

Actual Qwen3-1.7B inference preflight loaded successfully on CUDA. Comparing a
scenario alone versus as the first member of a left-padded two-item batch, with
explicit corrected position IDs, gave maximum across-layer relative L2 difference
0.0157703 (bfloat16; maximum absolute difference 8.0). Padding invariance is NOT
verified. Next check repeated identical batches and float32 to distinguish
deterministic numerical precision effects from input-position errors. Do not
claim exact invariance from token-position correction alone.

Float32 follow-up: repeated identical inference had maximum absolute difference
0.0; alone-versus-padded relative L2 difference was 2.42338e-6. The pilot therefore
uses float32 and batch size one. The primary bfloat16 configuration still requires
a precision sensitivity analysis before the full study.

## Pilot run

Prepared `configs/knows_pilot.yaml`: 128 scenarios, three paraphrases, KNOWS only,
Qwen3-1.7B pinned cached revision, float32, one training seed. All five extraction
depths and both pools are collected; final/last probes, controls, intrinsic
dimension and compression form the small execution pilot. This is an engineering
pilot, not an eligible comparative study. MLP LLC is not included.

Remote execution directory: `/home/forestzhang001/knows-pilot.FQh2F6` on
`spark-remote`. `scripts/run_knows_pilot.sh` runs sequential stages under nohup,
with `pilot.log` and terminal exit status `pilot.exit`. Check live processes and
this exit file before resuming; never relaunch merely because SSH disconnects.
Launch verified: shell PID 227348, labeling PID 227349, with 50/384 completed in
the first live log inspection. No exit file existed at that inspection.

Pilot terminal outcome: exit status 1, shell PID absent. Labeling and extraction
completed; MDL refused the single-class tuning split. Counts: 132/384 stable,
158/384 answer-order disagreements. Stable labels: coding 78 positive/12 negative,
tuning 15 positive/0 negative, evaluation 26 positive/1 negative. The study cannot
proceed on this dataset. Do not choose another split seed to force a passing result.
Investigate task adherence of the raw forced-choice prompt on the instruct model
and scenario balance. Any revised prompt must create a separately versioned pilot;
the failed pilot remains part of the record.

Automatic LLC calibration is now implemented in `src/probe_calibration.py`, bound
to sampler source, checkpoint, coding data and chain seeds. It selects from a
declared grid based on diagnostic acceptance, freezes limits/settings, and rejects
stale reuse. CLI uses the KNOWS-selected MDL-best layer for every predicate.
Seventeen local tests pass, including frozen-calibration regression tests. Actual
MLP calibration has not run; broader estimator validation remains required.

Follow-up prompt diagnosis used the first 24 tuning examples only, with Qwen's
native chat template and thinking disabled. Fifteen passed stability (12 positive,
3 negative). This improves class coverage but does not establish prompt validity.
Inspection exposed another contradiction: paraphrase 1 asserted that the subject
formed a view favoring the claim before saying they rejected it. Scenario v3
removes that assertion, adds missing sentence separators, and specifies apparent
evidential support. Both changes are versioned in `configs/knows_pilot_v3.yaml`;
the factor sample and split seed stay fixed. Old pilot artifacts are preserved
locally at `results/probe_experiments/knows_pilot_float32`.

Paired bootstrap orchestration now resamples independent scenario components
within fixed splits, shares indices across all 15 cells, and refuses missing or
censored estimates. The real four-estimator refit callback and report integration
remain to be implemented; confirmatory correlations are still suppressed.

Revised pilot remote directory: `/home/forestzhang001/knows-pilot-v3.Kk4nUV`.
Uses `scripts/run_knows_pilot.sh configs/knows_pilot_v3.yaml` with `pilot.log`
and `pilot.exit`. The initial pilot used for prompt/scenario decisions must be
excluded from eventual confirmatory evaluation; that exclusion still needs to
be enforced in the full-study scenario manifest.

Full-study generation now excludes the 128 development families, including
their minimal-pair variants, before sampling its 2,048 scenarios. Regression
tests verify disjoint family sets. Nineteen local tests pass.

The joint-refit driver is implemented in `src/probe_joint_refits.py`, invoked
through report-stage `--bootstrap-draws`. It requires accepted calibration,
15 distinct cells, and adequate class counts in the common stable cohort. It
refits all four estimators for every paired scenario resample and uses independent
chain seeds. Confirmatory intervals require at least 1,000 complete paired draws;
missing/censored estimates reject inference instead of being silently dropped.
This expensive path has not yet been exercised end-to-end on real eligible data.
Compression stage now runs all configured training seeds rather than only seed 0.

Revised pilot labeling completed: 194/384 stable; 152 answer-order disagreements.
Stable counts are coding 94 positive/35 negative, tuning 23 positive/3 negative,
evaluation 35 positive/4 negative. Both classes are now present, permitting the
engineering probe run, but class counts and instability remain far short of a
credible comparative study. Do not lower the 1,000-per-class threshold.

Revised pilot completed with exit status 0. MLP final/last macro-F1 was 0.7692,
but online codelength 1.2741 bits/label was not better than shuffled labels
(1.2638); the controlled MDL criterion failed. Linear macro-F1 was 0.5477.
Intrinsic dimension was undefined because full-probe tuning accuracy equaled
the majority baseline (both 0.8846). Tuning selected 3-bit quantization (216,231
serialized bytes), but its independent test macro-F1 fell to 0.4923 and accuracy
to 0.7179. No valid compact-complexity result follows from this pilot.

Subsequent audit fix: null permutations now exchange whole equal-size independent
groups within splits, preserving paraphrase dependence. The completed pilot used
the earlier row permutations and must retain that limitation in its record.

LLC corrections now freeze the preconditioner halfway through burn-in, retain
loss/distance traces, require split R-hat <= 1.05 and ESS >= 100, check frozen
basin-distance/loss limits, and require all four axial neighboring settings to
stay within 10%. The logistic validation uses longer chains and a 20% reference
tolerance plus mixing checks. Automatic calibration and broader reference-task
validation remain requirements for a full study; automatic calibration itself is
implemented and regression tested. A passing logistic check is not approval for MLPs.

## Latest verified reanalysis

With group-preserving null permutations, the MLP code is 1.274093 bits/label,
versus 1.532426 for shuffled labels and 1.263910 for shuffled representations.
It passes the specific 10%-over-shuffled-label gate, but does not outperform the
representation control. The earlier pilot's row-shuffled values remain historical
results, not the current control estimate. Reanalysis artifacts are in
`results/probe_experiments/knows_pilot_v3_corrected_controls` with runtime/source
metadata. The reanalysis used the same frozen DGX activations and was trained
locally, so its runtime differs from the original pilot.

Twenty-three local tests pass, including a 30-cell-invocation integration test of
the joint-refit driver (two resamples of 15 cells, with estimator stubs). This tests
wiring, not numerical validity of full MLP LLC inference. Subspace solutions now
persist phi and reconstruction seeds. Report gates explicitly distinguish
within-predicate quality from cross-predicate comparability.

Current implementation fixes: generated minimal-pair families now contain the
two defeater variants and are sampled as units; development-family exclusion
therefore removes complete pairs. Joint-refit resample positions are explicitly
mapped back to original activation rows. The full suite now has 23 passing tests.
These fixes do not alter or promote the existing DGX pilot artifacts.

## Minimal-pair-fix label pilot

Version `knows_pilot_v4_labels` was run on the DGX using the pinned Qwen3-1.7B
revision, native chat formatting, and float32 inference. It completed with exit
status 0. Of 384 labeled paraphrases, 214 were stable and 135 changed semantic
answer under answer-order reversal. Stable labels were 170 positive and 44
negative (coding 116/27, tuning 19/5, evaluation 35/12). This remains a gating
failure: the stability rate is low and both classes are far below the required
1,000 stable examples per class. No full 2,048-scenario run was launched.

Prompt-validation version `knows_pilot_v5_explicit_labels` kept the same model,
scenario factors, seed, and DGX float32 execution but added an explicit semantic
A/B mapping sentence. It completed with exit status 0 and performed worse: only
16/384 labels were stable and 358 reversed under answer-order counterbalancing
(all 16 stable labels were positive). This variant is rejected and does not
justify changing the preregistered protocol.

Version `knows_pilot_v6_semantic_labels` instead scored the displayed Yes/No
strings directly while still reversing their displayed order. It completed on
the DGX with 260/384 stable labels and 58 answer-order reversals. This is a real
reliability improvement, but the stable labels were still 213 positive and only
47 negative. A 2,048-scenario run at that observed rate would be expected to
remain below the required 1,000 stable negatives. Factor audit also found many
positive KNOWS judgments when the stated belief or truth factor was false, so
the 1.7B pilot is not adequate evidence to launch the primary study.

Primary-model version `knows_pilot_v7_qwen14b_semantic_labels` completed on the
DGX with exit status 0. Qwen3-14B produced 377/384 stable labels and only two
answer-order reversals, but the stable labels were 352 negative and 25 positive;
the tuning split had no positives. The prompt is reliable on the primary model,
but the unstratified factor sampler cannot meet the 1,000-per-class gate.

The prospective v4 full-study configuration therefore allocates 24% of scenarios
to canonical-support families (truth, belief, reliable evidence, no Gettier
structure, and no guessing) and retains 76% factorial controls. Minimal-pair
defeater variants remain together. This yields 492 canonical-support and 1,556
control scenarios after excluding all 128 development families. In v7, 17/18
canonical-support paraphrases were stable positives, providing headroom above
1,000 positives without selecting on labels. A balanced v8 primary-model label
pilot must pass before the 2,048-scenario run begins.

Version `knows_pilot_v8_qwen14b_balanced_labels` completed on the DGX with exit
status 0. It produced 373/384 stable labels: 86 positive and 287 negative, with
only three semantic answer-order reversals. Within the prospectively allocated
canonical-support stratum, 79/90 rows were stable positives; within the control
stratum, 282/294 were stable negatives. The corresponding 95% Wilson lower-bound
projections for the full manifest are about 1,172 stable positives and 4,341
stable negatives, so the label pilot passes the pre-launch 1,000-per-class gate.
The batched continuation scorer exactly reproduced forward labels, reversed
labels, stability, and semantic labels for all 312 scenario texts shared with
the unbatched v7 run.

The deterministic full manifest contains 2,048 scenarios, 6,144 paraphrases,
and 1,024 intact minimal-pair families. Canonical-support/control scenario counts
are 364/1,080 in coding, 76/318 in evaluation, and 52/158 in tuning. The full
Qwen3-14B label stage was first launched in the new DGX directory
`/home/forestzhang001/knows-mdl-v4-balanced`, but failed before producing a label
artifact: YAML had parsed the unquoted predicate name `true` as Boolean `True`,
and the labeler rejected that unknown predicate. The failed run and exit status 1
remain preserved. The configuration now quotes every YAML-sensitive predicate,
has a regression test requiring the exact five-string predicate list, and uses
the new output/directory version `knows_mdl_v5_balanced` /
`/home/forestzhang001/knows-mdl-v5-balanced`.

The corrected v5 full label job is detached on the DGX. A second detached
continuation waits for its terminal exit. On successful labeling it writes a
hash-bound `label_gate.json` with per-predicate and per-split counts, and starts
scenario-only extraction only if KNOWS has at least 1,000 stable examples in
both classes. It never starts extraction after a failed label process or failed
KNOWS sample-count gate. Later probing must still apply each predicate's own
class-count and held-out-F1 gates.

The v5 label and extraction stages subsequently completed with exit status 0.
Every predicate passed the preregistered 1,000-per-class sample gate. Stable
negative/positive counts were: KNOWS 4,253/1,640; BELIEVES 2,194/3,828; TRUE
3,858/2,027; JUSTIFIED 4,025/1,933; and LUCKY/GUESSED 3,881/1,704. The activation
audit found 49 float32 arrays of shape `(6144, 5120)`, no malformed arrays, the
pinned Qwen3-14B revision, and the matching scenario digest. Scenario and
question-only activation artifacts occupy about 11.5 GiB in total.

Primary MDL execution uses a separately versioned `analysis_v2` source tree so
the label/extraction producer source remains immutable. This analysis revision
trains probes on CUDA while leaving the frozen activations unchanged, and writes
the resolved device plus source hashes in every MDL sidecar. A deterministic toy
CUDA validation produced the same logits in two repeated fits (maximum difference
0.0) and reached macro-F1 0.937. The five-depth, final-token,
Qwen3-14B KNOWS MDL grid is running under the detached `mdl_primary` wrapper.

The primary MDL run completed with exit status 0 and its built-in preregistered
escalation also completed all 39 intermediate block cells. An independent gate
selected p75 among the five declared depths: mean normalized MLP codelength
0.177663, 0.155137 bits per label, untouched-evaluation macro-F1 0.986700, and
mean reductions of 86.07% versus shuffled labels and 86.23% versus shuffled
representations. The embedding-state baseline had normalized codelength 1.001818
and macro-F1 0.436426. Across the complete layer sweep, the MLP minimum was
block 29 (normalized codelength 0.169629; 0.148122 bits per label; macro-F1
0.983778); the linear minimum was block 28 (0.171150 normalized; macro-F1
0.980312). All 41 unique depths have complete three-control, five-seed tables for
both probe kinds.

Surface baselines remain an important qualification: bag-of-token macro-F1 was
0.947505, while token-count/style macro-F1 was 0.737073. The hidden-state result
therefore establishes compact model-relative decodability and strong null-control
separation, but the scenarios also make much of the decision rule lexically
accessible. A mistakenly launched duplicate all-layer successor was stopped
after it began re-running early blocks; the completed original sweep was audited
afterward and no cell was missing or malformed.

The separately versioned `analysis_v3` runner adds CUDA execution and complete
source/device sidecars for random-subspace ID. Its known low-dimensional toy task
reached accuracy 0.966667 and exactly repeated its logits. Primary block-29 KNOWS
ID and compression both completed with exit status 0. Every one of five random
projection seeds reached d90 at dimension 1,024, or 0.078% of the full
1,311,233-parameter MLP. Untouched-evaluation macro-F1 at d90 ranged from
0.969752 to 0.974863. All 55 saved subspace states reloaded successfully.

Every primary compression seed met the critical compression criterion with a
truncated-SVD artifact: ranks 4 or 8, 132,796--218,828 bytes versus 5,288,464
serialized bytes for the original probe (39.82x--24.17x compression), retaining
22,021 or 43,529 represented parameters. Untouched-evaluation macro-F1 ranged
from 0.971448 to 0.983729. All 110 compressed artifacts across quantization,
pruning, and SVD reloaded successfully and every recorded byte count matched the
actual file size.

An independent untouched-test replay then found that the original parameter
vector used for compression diagnostics could alias PyTorch parameter storage.
The tuning-selected artifacts and compression choices were unaffected, but a
later SVD state load mutated the in-memory reference and caused sidecar test
deltas to be recorded as exactly zero. `load_state_vector` now clones its input,
with a regression test that mutates the model afterward and verifies the source
array is unchanged. The affected primary compression cell was rerun under
versioned `analysis_v5` and re-audited successfully. Correct untouched-test
delta-NLL values are 0.000756--0.025411 nats and accuracy losses are
0.001742--0.009582; all remain within the declared 0.05-nat/one-percentage-point
criterion.

The two replication data passes run under a separate detached wrapper behind
primary ID/compression. It labels and extracts the identical manifest
first with pinned Qwen3-1.7B and then pinned Llama-3.1-8B-Instruct, writing a
per-model `label_gate.json`. Extraction is shared infrastructure and proceeds
after successful labeling even when some predicates fail the class gate; all
comparative reporting remains suppressed for any ineligible predicate.

A third detached queue waits for both replication data passes. It runs the
five-depth final-token MDL grid for the four remaining Qwen3-14B predicates and
for only the class-eligible predicates of each replication model. Qwen3-14B
KNOWS is explicitly excluded from this queue because its complete sweep already
exists. Missing label gates or activation sidecars are recorded as failures, not
silently treated as an empty eligible set.

A fourth detached queue computes the five-depth intrinsic-dimension and
compression grids only for model/predicate pairs that pass both the 1,000-per-
class label gate and the untouched-final-layer macro-F1 gate. It waits for all
quality-gate artifacts, skips already complete cells on restart, and overlaps
CPU-only compression with the corresponding GPU intrinsic-dimension fit. This
queue is independent of the controlling Mac and writes per-cell plus aggregate
exit markers.

On 2026-09-12, the primary wrapper was found to have written its two successful
component exits but omitted the aggregate `id_compress_primary.exit` marker.
The replication watcher therefore exited 3, and the dependent MDL and quality
queues recorded nonzero exits. A downstream grid also incorrectly checked only
for a terminal quality marker rather than requiring exit 0 and began eligible
primary cells prematurely. All failed markers, PIDs, and wrapper logs were moved
to `queue_failure_20260912` on the DGX. The partial grid process was stopped;
completed cell tables were retained for resumable reuse. Every queue boundary
now explicitly requires a zero upstream exit. After independently verifying
both primary component exits and auditing their artifacts, the missing aggregate
success marker was reconstructed and the complete replication-to-report chain
was relaunched. Qwen3-1.7B labeling was then observed live in the corrected
chain.

The LLC implementation has been held back from execution and revised before any
estimate was made. It now uses the maintained DevInterp v2 low-level sampler
with a custom binary-cross-entropy evaluator and RMSprop-SGLD, pinned at commit
`fbbf4c54e1f6ee46acb149f004df261fb05055c6`. Chains use minibatches on the GPU,
record periodic full-coding-set losses for the trained-basin check, and evaluate
the intended five axial sensitivity settings rather than a 3-by-3 Cartesian
grid. The prerequisite gate now uses untouched evaluation macro-F1, and the LLC
layer selector searches the completed all-layer MDL sweep (therefore selecting
block 29 for primary KNOWS). No LLC run may begin until the regular logistic
`d/2` validation passes in the exact pinned environment and all required
MDL/ID/compression cells have passed their controls.

The pinned DevInterp environment installed successfully and the corrected
regular logistic validation passed before any MLP LLC run: expected LLC 1.5,
observed 1.724556 (14.97% relative error), split-R-hat 1.001548, and ESS
1,808.33. The four chain estimates were 1.81744, 1.70073, 1.63948, and 1.74057.
Two prior statistical validation failures and one bookkeeping failure remain
preserved as failed v1-v3 artifacts. They identified (1) a dropped-partial-batch
reference mismatch and (2) thinning/minibatch phase aliasing in the initial
adapter. The accepted adapter chooses an exact-divisor batch size and defines
each retained loss as the mean of all post-burn-in transition minibatches since
the preceding draw. The production LLC queue remains stopped behind the full
ID/compression prerequisite marker.

A detached artifact-audit stage now sits between the ID/compression grid and
LLC. For every eligible cell it requires every declared projection seed and one
d90 hit per seed, reloads all subspace states, requires one acceptable selected
compression artifact per probe seed, reloads all quantized, pruned, and SVD
formats, and verifies reported serialized byte counts against the filesystem.
LLC requires this audit's zero exit, not mere filename existence. A final report
stage runs after LLC reaches any terminal outcome so rejected estimates and
diagnostics remain visible rather than suppressing the report.

The first full audit pass failed the five Qwen3-14B input-embedding cells, one
per predicate. The cause was the auditor, not the artifacts. At `emb` the full
MLP never exceeds its majority baseline on tuning data -- tuning accuracy equals
majority accuracy to machine precision for all five predicates -- so the ID stage
correctly declares `status: "undefined"` with reason
`full_probe_has_no_positive_improvement`, writes a sidecar, and writes no
dimension table. The auditor still demanded that table and crashed on the missing
file. `audit_id` now accepts that declared undefined outcome as a valid terminal
state, still fails a missing table that carries no declared outcome, and still
fails an undefined cell that nevertheless wrote subspace artifacts; three
regression tests cover those branches. The audit source was versioned as
`analysis_v6` and the audit, LLC, and report queues re-run from it; the earlier
stages remain pinned to the `analysis_v5` snapshot they actually used. On rerun
all 30 eligible cells passed with aggregate exit 0. These five cells were already
excluded from LLC by the existing `llc_gate` prerequisite check, so no comparative
claim rests on an undefined `d90`.

## LLC calibration: the default candidate could not satisfy the basin criterion

The first production LLC run reached calibration and was rejected: every chain
under the single default candidate (learning rate 3e-4, localization 1.0) left
the trained basin, so the sensitivity grid never had four valid chains at any
setting and `calibrate` raised. The cause is geometric and was predictable
without running anything. A Gaussian localization gamma holds a chain at
stationary radius `sqrt(d / gamma)`. The primary probe has d = 1,311,233 and
`||w|| = 14.2095`, so `sqrt(d)` = 1,145.09 -- and the observed mean chain
distance was 1,145.5, matching to four significant figures across all twenty
chains, at every learning-rate multiplier. The preregistered basin limit is
`0.25 * ||w||` = 3.5524, which requires `gamma >= 1.04e5`. The declared default
was three thousand times too weak for a probe of this dimension.

The configuration declared no `calibration_candidates`, so `calibrate` fell back
to its one-element default and had a single attempt. The first declared grid
raised localization by five orders of magnitude while holding the learning rate
near its original value, and all twelve candidates diverged within ninety
seconds. Step size must be coupled to localization: the discretized localization
step contracts by `lr * gamma` per step, so a gamma large enough to satisfy the
basin criterion needs a proportionally smaller step. A diagnostics-only
stability probe on the calibration cell (200 burn-in, 200 draws, inspecting only
completion, chain distance, and loss increase) placed the divergence boundary
near `lr * gamma = 1`: at gamma 3e5 the chain completed for lr 1e-6, 1e-7, 1e-8
and returned NaN losses at 1e-5 and 3e-4; at gamma 1e6 it completed at 1e-7 and
below. Observed radii tracked `sqrt(d / gamma)` throughout (2.17 against 2.09
predicted at gamma 3e5, 1.14 against 1.15 at gamma 1e6). The grid in force fixes
`lr = c / gamma` for `c` in `0.3, 0.03, 0.003` at gamma `3e5, 1e6, 3e6, 1e7`,
weakest localization and coarsest stable step first.

No acceptance criterion changed at any point. Selection remains the declared
mixing, basin, and axial-sensitivity checks, and the LLC value is never a
selection criterion. The same stability probe already indicates what the honest
outcome will be, and this expectation is recorded in the amendment before the
run rather than after it: in the basin-compliant regime the estimate sits at the
noise floor and scales roughly as `1/gamma` -- LLC 1.16 at gamma 1e4, 0.073 at
gamma 1e5, and between -0.012 and 0.05 at gamma 3e5 and above, against `d/2` =
655,616 for a regular model of this size -- while every chain that produces a
large positive estimate has left the basin, with loss increases of 1.1 to 3.9
nats at gamma 10 to 1e3 against the 0.05 limit. There is no plateau between the
two regimes. If the axial-sensitivity check rejects every candidate, LLC is
reported as non-identifiable for this probe under the declared locality
criteria; that rejection is the result and will not be tuned away.

A related diagnostic defect surfaced here and is fixed: `run_llc` overwrote a
chain's specific failure reason with `incomplete_trace` whenever the trace was
short, so a diverged sampler was recorded as bookkeeping rather than as
`sampler_error:*` or `incomplete_or_nonfinite_trace`. The specific reason now
survives, with a regression test.

### Verdict: the localized learning coefficient is not identifiable here

The coupled grid ran to completion and every one of the twelve declared
candidates was rejected, as the pre-declared expectation said it would be. The
rejection is substantive rather than numerical. Mixing was excellent everywhere
(split R-hat 0.999, ESS 4,000 at every candidate) and eleven of twelve stayed
well inside the basin (maximum chain distance 0.36 to 2.29 against the 3.55
limit; loss increase at most 0.0004 nats against the 0.05 limit). What failed
was axial sensitivity: eight candidates were rejected for exceeding the declared
10% tolerance and four because doubling the learning rate crossed the stability
boundary at `lr * gamma = 0.6` and produced non-finite chains.

The estimates themselves show why. Across the basin-compliant grid the central
LLC runs 0.031, 0.009, 0.008, 0.009, 0.003, 0.002, 0.003, 0.001, 0.001, 0.001,
0.000, 0.000 as localization rises from 3e5 to 1e7 -- an almost exact `1/gamma`
law, with halving gamma roughly doubling the estimate (candidate 02: 0.0076 at
center, 0.0157 at half gamma, 0.0039 at double). An estimate proportional to the
localization strength is an estimate of the localization prior, not of the local
geometry of the probe's loss landscape. For reference, `d/2` = 655,616 for a
regular model of this size, and the only settings that produced values of that
order (2,744 to 9,113 at gamma 10 to 1e3) had left the basin by 1.1 to 3.9 nats.
There is no plateau between the two regimes.

The reported outcome is therefore that the learning coefficient is
non-identifiable for this probe under the declared locality criteria. No LLC
value is reported, the confirmatory four-measure convergence test cannot run,
and the study stands on three measurements: online MDL, random-subspace
dimension, and practical probe compression. Nothing was tuned to avoid this
outcome, and the expectation was recorded in `config_amendments.json` before the
run that produced it.

Two reporting defects found while reviewing that output are fixed. `_tables`
globbed the whole results tree, so two superseded calibration directories that
had been archived in place were reported alongside live results; the archives
now live outside the experiment tree, and table rows are sorted so the report is
deterministic. The LLC section also printed a table of rejections without ever
stating the conclusion; it now reads the frozen calibration manifest and says
plainly that no estimate is reported and why. The final report was regenerated
from the `analysis_v9` snapshot.

Amending a frozen configuration mid-experiment previously had no honest route:
`run` refused any digest change outright, which invites editing the frozen
sidecar instead. It now accepts a change only when `config_amendments.json`
declares it in advance -- chaining from the digest actually in force, naming the
exact changed setting paths as `config_diff` computes them, and recording a
reason. The amendment is appended to `experiment.sidecar.json`, so the frozen
record carries what changed, when, and why. Five regression tests cover the
accepted case and every rejection: a broken chain in either direction, a
mis-named change, an empty change list, and a blank reason. The rejected default
calibration is preserved as `llc_calibration_default_rejected_20260913`, the
uncoupled-grid rejection as `llc_calibration_uncoupled_grid_rejected_20260913`,
the stability probe as `llc_stability_probe.py`, and the report generated
without any LLC section as `REPORT.no_llc.md`. The recalibration runs from the
`analysis_v8` snapshot.

## Overnight experiments E1-E4

Hypotheses, predictions, and falsification criteria were written to
`NIGHT_EXPERIMENTS.md` and copied into the run directory before any of the four
produced a number. Outcomes are appended to the same file beneath the unedited
predictions. Two predictions did not survive and are reported as failures.

The four reuse the frozen activations, labels, groups, and splits of the
completed run; none refits label generation. They are motivated by a finding
that was already in the completed artifacts and had gone unremarked: the
**linear probe is as cheap as or cheaper than the MLP for all five predicates**
(0.84, 0.89, 1.01, 0.80, 0.96 in bits). The decision rules are linearly
decodable, so quoting the bound on a 1,311,233-parameter MLP overstates it.

**E1, the bound ladder.** For KNOWS at block 29 the description shrinks from
5,244,932 bytes (MLP float32) to 5,357 bytes (quantized linear probe) at 0.979
untouched-evaluation macro-F1 -- a factor of 979, and the predicted three orders
of magnitude. Two results are worth more than the headline. First, folding the
standardizer into the first affine layer is exact, verified at a maximum
relative logit difference of 5.7e-7, so it is not part of a direct probe's
description. Second, the intrinsic-dimension move -- section 3.1's move applied
to this object -- **does not tighten the bound**. It is cheapest only when the
standardizer is free: theta0 + P phi reconstructs a probe over standardized
inputs and so cannot absorb it, leaving the linear subspace at 43,016 bytes
standalone against 5,357 for simply compressing the linear probe. The subspace
is worth reporting as overhead shared across predicates and layers, not as a
standalone bound. The predicted collapse of the linear probe's d90 did not
happen: 512 against the MLP's 1,024, a factor of two.

**E2, JTB composition.** Component features are out-of-fold logits from 5-fold
cross-fitting over independent scenario groups, so no example is scored by a
probe that saw it or any paraphrase of its scenario. On the 4,970-example cohort
stable for all five predicates, KNOWS costs 0.139 bits per label from three
numbers against 0.129 from the full 5,120-dimensional representation, a ratio of
1.08 against a declared ceiling of 1.5. Both controls behave: shuffled
components land at 0.85-0.88 against the cohort's 0.902-bit marginal, and a
random three-dimensional projection at 0.62-0.80.

Two structural findings came out of the subsets. The composition is **not
linear**: a linear readout of the same three numbers costs 0.380, a ratio of
3.18, even though a linear probe reads KNOWS off the raw activations more
cheaply than an MLP does. The components are linearly present in the
representation; combining them is not a linear operation, which is what a
conjunction looks like. And **belief contributes nothing**: justified + true
alone reaches 0.139, tied with all three, while believes alone (0.596) is no
better than a random three-dimensional projection.

The Gettier prediction could not be tested, and the reason is itself the
finding. In the evaluation split all 344 `lucky_guessed = 1` examples have
`knows = 0`; marginal entropy is exactly zero, against 0.980 bits on the rest,
so no per-example codelength comparison between those subsets can mean anything.
Across the whole cohort the model called an example a lucky guess and also
granted knowledge in 2 of 1,563 cases. The model's KNOWS excludes epistemic luck
essentially without exception -- a consistency property of its own two answers on
a design where luck varies separately from truth and justification, but evidence
that it finds those cases *easy*, not hard.

**E3, ordering intervals.** 400 paired scenario-group resamples on the
4,970-example common cohort, zero degenerate resamples. The declared prediction
that believes would be cheapest in at least 80% of draws **fails**: it is
cheapest in 35.7%, behind justified at 40.0%, and no predicate approaches the
threshold. The dearest half holds, with lucky_guessed dearest in 93.5%. Only 2
of 10 paired differences have 95% intervals excluding zero, both against
lucky_guessed. Believes, justified, knows, and true sit between 0.141 and 0.161
normalized MDL with heavily overlapping intervals; **knows does not separate
from any of its components**. The point-estimate ordering also moved between the
descriptive table (per-predicate subsets, bits per label) and this one (common
cohort, normalized MDL), which is further evidence that the fine ranking is not
a stable quantity. The single cross-predicate claim the study can make is that
*lucky guess* is the most expensive rule to decode. The descriptive five-way
ordering must not be reported as a ranking.

**E4, depth profile.** Every one of the five predicates reaches its minimum
codelength between block 26 and block 29 of 40 -- 65% to 72% of depth -- under
both probe kinds, ten cells out of ten, and in a far narrower band than the
declared 24-36 window. Five different epistemic predicates agreeing to within
three layers indicates that this depth is a property of the model's processing
rather than of the concept being decoded. The final layer costs 1.53x to 1.99x
the minimum under the MLP, confirming the declared 1.4x floor; under the linear
probe knows (1.38x) and lucky_guessed (1.34x) fall just below it, and that half
of the prediction is reported as failing. The four new predicates are minima
over a 15-point grid; KNOWS, which has the complete 40-layer sweep, puts its
minimum in the same place under both grids.

Three code defects surfaced and are fixed with regression tests. `run_compression`
assumed a two-layer probe and crashed on `nn.Linear`; the SVD rung is now skipped
for single-layer probes, since factorizing a one-row weight stores more numbers
than the row. `run_id_cell` and `train_projected` were hardwired to the MLP and
are now parameterized by probe kind. The first ladder assembly read sizes
straight off compression artifacts, which also store the standardizer, so
compressed rungs carried 40,960 bytes that the float32 and subspace rungs did
not -- the compressed linear probe appeared larger than the uncompressed one.
Rungs are now re-serialized without the standardizer before measurement.

## External review of the overnight report: two blockers, both confirmed

An independent review of the night report found two defects. Both are real.

**The compressed bound was not a valid description.** `payload_bytes` deleted
the standardizer arrays from a compression artifact whose weights had been
quantized in standardized-input space. Folding had been *verified* exact but
never *performed* before compression, so the 5,357-byte linear rung and the
177,334-byte MLP rung could not be applied to raw activations. Fixed: the probe
is folded first and saved with an identity standardizer; the folded probe is
what gets compressed; selection and reporting use raw activations; and
`payload_bytes` now refuses any artifact whose standardizer is not the identity.
Rerun from `analysis_v11`: the linear probe compresses to 13,210 bytes at 16 or
8 bits (4-bit collapses to 0.22 macro-F1) with 0.980 evaluation macro-F1, a 397x
reduction rather than the withdrawn 979x; the MLP to 177,506 bytes via SVD. The
reviewer also caught a framing error: the standardizer is fit per predicate, so
"shared overhead across predicates" was not a description of this pipeline.

**The ordering bootstrap merged duplicated groups.** Resampled copies kept
their original identifiers and `online_code` merged them with `np.unique`;
diagnostics on the coding split show ~455 distinct identifiers reaching the
coder out of 720 sampled instances. `paired_resample` now labels every copy
distinctly and underlies both the ordering bootstrap and the joint refit driver.
Every summary of the defective run -- medians included, since the defect changed
the effective size and endpoints of each draw -- is withdrawn. A second review
then found the corrected driver training on a single probe seed against a
protocol of five; multi-seed support with per-draw averaging before ranking was
added, and the confirmatory 400 x 5 run launched from `analysis_v12` alongside
the single-seed control run. That control finished with 400 clean draws: lucky_guessed dearest in 96.8%, believes cheapest in only 40.3%, three of ten pairs separated and all against lucky_guessed, and every median about 30% below the withdrawn run's -- the merged groups had inflated every codelength. The confirmatory five-seed run (400 draws, per-draw mean over seeds before ranking) reproduces the control's levels within 0.005 with tighter intervals: lucky_guessed dearest in 100% of draws, believes cheapest in 31.0% (prediction fails), and five of ten pairs separated -- the four against lucky_guessed plus justified cheaper than knows (-0.017 [-0.029, -0.007]), which falsifies part of the declared non-separation prediction. The study's ordering claims are exactly those two, stated for the five-seed average: per seed the justified-knows sign is constant but three of five individual intervals contain zero, so the separation belongs to the average, not to any single training. Non-separation of knows from believes and true is not evidence of equal complexity.

**E1 is failed on its own criterion.** The preregistration falsifies E1 if the
subspace does not beat direct compression; it does not, and no correction of the
compressed rung changes that, since even the uncompressed linear probe is
smaller than the standalone subspace.

**Interpretive corrections adopted.** "Conjunction" is softened to "compatible
with a conjunctive composition"; "belief contributes nothing" becomes "no
detectable improvement, necessity not identifiable", because among the 1,701
common-cohort examples with JUSTIFIED = 1 and TRUE = 1 only one has
BELIEVES = 0; and the design factor `gettier` (418 scenarios, KNOWS granted in
none, 395 judged lucky by the model) is separated from the model's own
`lucky_guessed` judgement (1,563 examples).

## Earlier embedding experiments

Do not schedule a wholesale rerun of the pre-probing embedding experiments.
Their preserved caches and sidecars are the authoritative artifacts for the
historical analyses, including the retired Curie endpoint result. Regenerating
with current Qwen/Llama models or replacement APIs would create different
representations and is a new robustness study, not a reproduction of the same
experiment. A later preservation audit should verify cache/sidecar inventories,
hash or archive immutable artifacts, and rerun analysis code from those preserved
caches where feasible. Only missing or corrupt reproducible artifacts justify a
targeted rerun; retired-endpoint artifacts should be documented as irreproducible
from source rather than silently substituted.

## Method references

- [Voita and Titov, Information-Theoretic Probing with Minimum Description Length](https://arxiv.org/abs/2003.12298)
- [Li et al., Measuring the Intrinsic Dimension of Objective Landscapes](https://arxiv.org/abs/1804.08838)
- [DevInterp source and implementation](https://github.com/timaeus-research/devinterp)
