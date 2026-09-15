# Plan: Variational MDL extension for epistemic probes

## Decision

Add the variational-code estimator from Voita and Titov (2020) as a
**separately versioned follow-up experiment**. It will reuse the frozen
activations, labels, scenario groups, and splits from the current study without
altering its artifacts or preregistered analysis.

Do not launch this extension until the current detached DGX chain reaches a
terminal report outcome. This avoids GPU contention and preserves a clean
boundary between the existing study and the follow-up.

## Current DGX run snapshot

Checked through the `spark-remote` Tailscale SSH alias at
**2026-09-12 19:56 EDT**.

- The complete run has not finished. The aggregate markers
  `replication_data.exit`, `replication_mdl.exit`, `quality_gates.exit`,
  `id_compression_grid.exit`, `artifact_audit.exit`, `llc.exit`, and
  `report.exit` were not yet present.
- Qwen3-1.7B labeling, its label gate, and scenario-only extraction all
  completed with exit status 0.
- Llama-3.1-8B-Instruct labeling was active at 850/6,144 examples.
- The active Llama process used approximately 38.8 GiB of GPU memory.
- The replication, MDL, quality-gate, ID/compression, audit, and LLC wrappers
  were alive. Their parent process was PID 1 and each had an independent
  session, so they do not depend on the controlling Mac remaining powered on.

This section is a dated observation, not a live status indicator. Before any
new launch, recheck processes, logs, and terminal exit markers on the DGX.

## Scientific question

Does variational MDL rank the model-layer-predicate cells similarly to online
MDL, and does it support the same conclusion about the compact decodability of
the model's epistemic decision rules?

The variational codelength is

\[
L_{\mathrm{var}}
= \mathrm{KL}(\beta\|\alpha)
+ \mathbb{E}_{\theta\sim\beta}
  [-\log_2 p_\theta(y\mid x)],
\]

where \(\alpha\) is the agreed parameter prior and \(\beta\) is the learned
variational posterior. The first term prices the probe; the second prices the
labels given sampled probe parameters.

This is distinct from the existing practical-compression experiment:

- variational MDL is a label codelength containing a Bayesian model cost;
- quantization, pruning, and SVD measure the serialized implementation size of
  an already trained deterministic probe;
- neither quantity should be renamed or treated as numerically interchangeable
  with online MDL.

## Reference implementation

- Method source: `2003.12298v1.pdf`, especially Sections 2.2.1 and 3.1.
- Official reference repository:
  `https://github.com/lena-voita/description-length-probing`.
- Pin the inspected repository commit rather than tracking its default branch:
  `2696af04226cff191eff265dbed0744512bad7b5`.
- Reconstruct and document the exact prior, posterior parameterization,
  Bayesian dense layer, pruning rule, optimizer, and codelength evaluation used
  by the reference code before implementing the modern version.

The legacy repository is a specification and validation reference. Do not run
its old environment unmodified as the production implementation.

## Experimental variants

### Primary: controlled comparison

Use the same architecture as the current deterministic MLP:

- input -> 256 hidden units -> GELU -> binary output;
- identical standardized frozen inputs;
- identical tuning, coding, and untouched evaluation groups;
- the same five experiment seeds;
- the same observed-label, shuffled-label, and shuffled-representation
  controls.

Holding the architecture and data constant makes differences between online
and variational codelength easier to interpret.

### Secondary: paper-faithful sensitivity

If the primary pilot succeeds and resources permit, run a sensitivity variant
using the paper's main MLP-2 design: two 1,000-unit ReLU hidden layers. Treat
this only as an architecture sensitivity analysis, not as the primary
comparison.

## Staged scope

### Stage 0: specification audit

1. Extract the exact variational objective and Bayesian layer behavior from the
   pinned reference implementation.
2. Decide and record which information is shared with the decoder: architecture,
   prior, optimization algorithm, seeds, standardizer, and tuning data.
3. Define all logarithm conversions explicitly; stored codelengths are in bits,
   while training losses may be in nats.
4. Freeze the Monte Carlo sampling and evaluation protocol before looking at
   cross-predicate results.

### Stage 1: local implementation and tests

Add a new stage without changing `src/probe_mdl.py` outputs:

```bash
python -m src.probe_experiments \
  --config configs/knows_variational_mdl.yaml \
  --stage vmdl
```

Suggested new modules and artifacts:

- `src/probe_variational_mdl.py`
- `configs/knows_variational_mdl.yaml`
- `results/probe_experiments/knows_mdl_v5_balanced_vmdl_v1/`

The new result directory should reference and hash the existing activation and
label artifacts read-only instead of copying or regenerating them.

### Stage 2: validation

Require all of the following before a DGX scientific pilot:

- analytic KL tests for the selected prior/posterior family;
- agreement between nats and bits calculations;
- a Monte Carlo convergence test for expected label NLL;
- deterministic replay under fixed seeds within the declared tolerance;
- no tuning or evaluation labels entering the coding objective improperly;
- finite losses and gradients throughout training;
- serialization/reload equality for posterior parameters and metadata;
- a synthetic learnable rule shorter than its shuffled-label control;
- a random-label task close to its declared trivial-code comparator;
- a small reference-reproduction check against the pinned Voita–Titov code or
  published qualitative ordering.

### Stage 3: minimal DGX pilot

Run Qwen3-14B KNOWS only at:

- input embedding;
- block 29, the best layer in the completed all-layer online-MDL sweep;
- final layer.

Use all five seeds and all three label/representation controls. Benchmark wall
time and GPU memory before estimating the cost of the expansion.

### Stage 4: expansion gate

Expand beyond the pilot only if:

- every seed completes without non-finite values or posterior collapse;
- Monte Carlo uncertainty is small enough to preserve the layer ranking;
- observed labels beat both shuffled controls in the expected direction;
- untouched-evaluation macro-F1 remains at least 0.75;
- results are reproducible after serialization and reload;
- the estimator and its prior are frozen before other predicates are examined.

If the gate passes, run the three declared layers for all class-eligible
Qwen3-14B predicates. Replication models are optional and should be attempted
only after reviewing the Qwen3-14B comparison and its compute cost.

## Hyperparameters to preregister

- prior family and all prior scales;
- posterior family and initialization;
- number of posterior samples used in training and evaluation;
- optimizer, learning rate, batch size, weight decay, and stopping rule;
- coefficient or schedule applied to the KL term;
- numerical precision and device;
- Monte Carlo seeds and convergence tolerance;
- criterion for posterior collapse or unusable sparsity;
- whether biases and all normalization parameters are included in the KL;
- exact trivial code used for normalized compression.

Hyperparameters may be calibrated only on Qwen3-14B KNOWS at block 29 using
the tuning split. Freeze them before the three-layer comparison. Any sensitivity
grid must be declared in advance and reported in full.

## Outputs

For every model-layer-predicate-seed-control cell, save:

- total variational codelength in bits and bits per label;
- KL/model component and expected data component separately;
- normalized codelength and compression relative to the declared trivial code;
- tuning and untouched-evaluation NLL, accuracy, macro-F1, and calibration;
- Monte Carlo standard error or interval;
- learned sparsity and effective architecture where the posterior supports it;
- posterior checkpoint and reload verification;
- source, config, activation, label, scenario, and split hashes;
- software versions, GPU, precision, timings, and peak memory;
- complete diagnostics for failed or rejected cells.

Generate a standalone variational-MDL report first. Integrate it into the main
scientific report only after review; preserve rejected and null results.

## Comparisons and interpretation

Report:

1. Spearman and paired-bootstrap agreement in cell rankings between online and
   variational MDL.
2. Whether both estimators select the same best layer for each eligible
   predicate.
3. Associations with random-subspace \(d_{90}\), practical compressed probe
   bytes, and LLC.
4. Decomposition of disagreements into variational model cost versus predictive
   data cost.

Do not claim that close numeric values mean the codes are identical. The useful
convergence claim is that independent valid upper bounds lead to the same
qualitative ordering, with uncertainty excluding rank reversals of interest.

## Launch procedure after the current run

1. Confirm the current `report.exit` exists and inspect every upstream aggregate
   exit code; do not infer success merely from a missing process.
2. Review the current final report and decide whether unresolved controls make
   the variational extension scientifically worthwhile.
3. Sync a versioned implementation to a new DGX run directory.
4. Run preflight tests and the three-cell pilot.
5. Audit pilot artifacts and estimate full-grid cost.
6. Launch the gated expansion as a detached process with PID, log, and atomic
   terminal exit marker, so it also survives disconnection or shutdown of the
   Mac.

## Stop conditions

Stop rather than tune around the result if:

- the reference objective cannot be reconstructed unambiguously;
- the prior dominates all cells or the posterior collapses across the declared
  calibration grid;
- random labels appear substantially compressible;
- results depend materially on undeclared Monte Carlo choices;
- serialization changes measured codelength or predictions;
- the current study fails its artifact audit in a way that invalidates the
  shared labels, splits, or activations.

In that case, report the variational estimator as unsuccessful or
non-identifiable under the tested specification; do not replace it silently
with practical probe compression.
