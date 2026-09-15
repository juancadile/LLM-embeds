# Overnight experiments, declared before running

Written 2026-09-14, before any of E1-E4 produced a result. Predictions are
recorded here so that a failed prediction stays visible. Every experiment reuses
the frozen activations, labels, scenario groups, and splits of
`knows_mdl_v5_balanced`; none of them refits the label generation.

Motivating context. The reviewer's section 5 makes two asks: that the naive
"100 kB of parameters" bound be tightened, since quantization and pruning are
standard, and that the bound be placed on *something bounded* -- a small probe
over the model's representations rather than the whole prompted model. E1 and E2
answer those directly. E3 and E4 characterise what the measurement is sensitive
to.

A free finding from the completed run motivates E1 and E2: the **linear probe is
as cheap as or cheaper than the MLP for all five predicates** (ratio 0.80, 0.89,
1.01, 0.80, 0.96 for believes, justified, knows, true, lucky_guessed). The
decision rules are linearly decodable, so the 1,311,233-parameter MLP is not
what the bound should be quoted on.

## E1 -- Bound ladder

**Question.** How large is the honest description of the model's KNOWS rule,
given frozen representations?

**Hypothesis.** The tightest defensible bound is about three orders of magnitude
below the naive float32 MLP.

**Predictions.**
- Naive MLP float32: 5,244,932 B (measured, not predicted).
- Compressed MLP: 218,828 B (already measured).
- Compressed linear probe: < 10,000 B.
- d90 of the *linear* probe: well below the MLP's 1,024 -- a risky prediction,
  since it could equally come out higher.
- Subspace description (d90 floats plus the projection seed): < 5,000 B.

**Declared accounting trap.** The standardizer is 2 x 5,120 floats = 40,960 B
float32, which would dominate every tightened bound. The report must either
count it or state precisely why it belongs to the representation rather than to
the rule. Both totals will be reported; the headline number is the one that
counts it.

**Falsified if** the compressed linear probe is not smaller than the compressed
MLP, or if the subspace description does not beat direct compression.

## E2 -- JTB composition

**Question.** Is the model's KNOWS rule approximately the conjunction of its own
BELIEVES, JUSTIFIED, and TRUE rules?

**Hypothesis.** If it is, KNOWS labels are nearly as cheap to code from three
numbers -- the out-of-fold outputs of the three component probes -- as from the
full 5,120-dimensional representation.

**Predictions.**
- `bits(KNOWS | 3 JTB features)` <= 1.5 x `bits(KNOWS | 5,120 raw dims)`.
- `bits(KNOWS | 3 JTB features)` far below `bits(KNOWS | random 3-dim
  projection)` and below the shuffled-component control.
- Gettier: under the JTB-only model, per-example cost on the evaluation split is
  higher for `lucky_guessed = 1` than for `lucky_guessed = 0`, and that gap
  shrinks when `lucky_guessed` is added as a fourth feature.

**Leakage control.** Component features are produced by 5-fold cross-fitting over
independent scenario groups: every example's feature comes from a probe that saw
neither that example nor any paraphrase of its scenario. Component probes train
on out-of-fold coding rows and early-stop on out-of-fold tuning rows.

**Falsified if** three JTB features cost much more than the raw representation --
which would say the model's KNOWS is not composed from its own JTB components,
and is equally reportable.

## E3 -- Intervals for the cross-predicate ordering

**Question.** Does the codelength ordering survive scenario resampling?

**Hypothesis.** The observed ordering believes (0.108) < justified (0.145) ~
knows (0.148) < true (0.156) < lucky_guessed (0.195) reflects the model rather
than this particular scenario sample.

**Primary statistic.** Normalized MDL, the total code divided by the marginal
code, which is comparable across predicates whose class balance differs. Raw
bits per label is secondary and reported alongside. The probe is linear, which
the completed run showed to be as cheap as the MLP.

**Predictions.** Over paired scenario-group resamples on the cohort stable for
all five predicates (4,970 examples, 1,022 independent groups), believes is
cheapest and lucky_guessed dearest in at least 80% of draws; justified, knows,
and true are not separated, meaning their paired 95% difference intervals
contain zero.

**Method.** Paired resampling of independent groups within fixed splits, every
predicate evaluated on the same index vector, MDL refit per draw. LLC and
intrinsic dimension are excluded: LLC is non-identifiable for this probe, and
refitting all four estimators costs about 11.5 h per draw.

**Falsified if** the cheapest and dearest predicates change across draws.

## E4 -- Depth profile of the other four predicates

**Question.** Is the depth at which a rule is most compactly decodable a property
of the model or of the predicate?

**Hypothesis.** Every predicate reaches its minimum around 70-75% depth, and the
final layer is markedly more expensive.

**Predictions.** For each of believes, justified, true, lucky_guessed the
minimum-codelength layer falls between block 24 and block 36 of 40, and the
final layer costs at least 1.4x the minimum. For KNOWS, already measured, the
minimum is block 29 and the final layer costs 1.88x.

**Falsified if** any predicate's minimum sits at the final layer or before block
16.

## Shared rules

- Nothing below is selected on its outcome. Where a criterion is needed it is
  stated above.
- Every experiment writes its own directory, a sidecar recording software
  versions and input hashes, and a terminal exit marker.
- A failed prediction is reported as a failed prediction.

---

# Outcomes

Recorded against the predictions above, which are left unedited.

## E1 -- Bound ladder: FIRST OUTCOME INVALID, see corrections below

Measured for KNOWS at block 29, macro-F1 on the untouched evaluation split.

| Rung | Parameters | Standalone bytes | Marginal bytes | Eval macro-F1 |
|:--|--:|--:|--:|--:|
| MLP float32 | 1,311,233 | 5,244,932 | 5,244,932 | -- |
| MLP compressed | 1,311,233 | 177,334 | 177,334 | 0.978 |
| MLP subspace d90 | 1,024 | 45,064 | 4,104 | -- |
| Linear float32 | 5,121 | 20,484 | 20,484 | -- |
| **Linear compressed** | 5,121 | **5,357** | **5,357** | **0.979** |
| Linear subspace d90 | 512 | 43,016 | 2,056 | -- |

- **Withdrawn.** The 5,357-byte figure and the factor of 979 were produced by an
  invalid procedure (see corrections) and are not a result. The corrected
  numbers appear in the corrections section once the rerun lands.
- ~~**Headline confirmed.**~~ 5,244,932 -> 5,357 bytes is a factor of 979, essentially
  the predicted three orders of magnitude, and the compressed linear probe still
  scores 0.979 macro-F1 on data never used for selection. The predicted ceiling
  of 10,000 bytes holds.
- **Sub-prediction not confirmed.** The linear probe's d90 is 512 against the
  MLP's 1,024. That is a factor of two, not the "well below" that was predicted.
- **The standardizer is not part of a direct probe's description.** Folding it
  into the weights is exact, verified numerically at a maximum relative logit
  difference of 5.7e-7 for both probe kinds. The first affine layer absorbs it.
- **The intrinsic-dimension move does not tighten this bound.** It is the
  cheapest rung only if the standardizer is free. Standalone, the linear
  subspace costs 43,016 bytes against 5,357 for simply compressing the linear
  probe -- eight times worse. theta0 + P phi reconstructs a probe over
  *standardized* inputs, so unlike a direct probe it cannot absorb the
  standardizer. The move is worth reporting as shared overhead across predicates
  and layers, not as a standalone bound.
- **E1 is falsified on its own declared criterion.** The preregistration says
  the experiment fails "if the subspace description does not beat direct
  compression." It does not, and it cannot under any correction of the
  compressed rung: even the uncompressed linear probe at 20,484 bytes beats the
  standalone subspace at 43,016. The headline reduction survives; the
  experiment as declared does not, and is reported as failed/mixed.
- **Accounting defect found and fixed mid-experiment.** The first assembly read
  sizes straight off the compression artifacts, which also store the
  standardizer, so the compressed rungs carried 40,960 bytes the float32 and
  subspace rungs did not. The compressed linear probe appeared *larger* than the
  uncompressed one, 46,851 against 20,484. Rungs are now re-serialized without
  the standardizer before being measured, with a regression test.

## E2 -- JTB composition: confirmed, and the composition is not linear

Bits per label for KNOWS on the 4,970-example cohort stable for all five
predicates, averaged over five seeds. Component features are out-of-fold logits
from 5-fold cross-fitting over independent scenario groups.

| Condition | MLP readout | Linear readout |
|:--|--:|--:|
| Full 5,120-dim representation | 0.129 | 0.119 |
| **Justified + true (2 numbers)** | **0.139** | 0.452 |
| **Believes + justified + true (3 numbers)** | **0.139** | 0.380 |
| JTB + lucky guess (4 numbers) | 0.140 | 0.347 |
| Justified alone | 0.142 | 0.564 |
| True alone | 0.159 | 0.563 |
| Believes alone | 0.596 | 0.761 |
| Lucky guess alone | 0.720 | 0.864 |
| Random 3-dim projection (control) | 0.618 | 0.804 |
| Shuffled components (control) | 0.852 | 0.883 |

- **H2a confirmed for the MLP readout.** Three numbers cost 0.139 bits per label
  against 0.129 from the full 5,120-dimensional representation: a ratio of 1.08,
  far inside the declared 1.5 ceiling.
- **H2a fails for the linear readout**, at a ratio of 3.18, and that failure is
  the informative part. The components are *linearly* available in the
  representation -- a linear probe reads KNOWS off the raw activations at 0.119,
  cheaper than the MLP -- but combining the three component outputs into KNOWS
  requires a nonlinearity. This is *compatible with* a conjunctive composition;
  it does not show that conjunction is the model's internal operation. The
  defensible statement is that a nonlinear summary of the three component
  scores preserves almost all of the codelength of the full representation.
- **H2b confirmed decisively.** Both controls behave: shuffled components land at
  0.85-0.88 against the cohort's marginal entropy of 0.902 bits, that is, they
  carry no information; a random 3-dim projection lands at 0.62-0.80.
- **Belief adds no detectable improvement, and its necessity is not
  identifiable.** Justified + true alone reaches 0.139, tied with all three, and
  believes alone (0.596) is no better than a random three-dimensional
  projection. But the cohort cannot separate the two readings: among the 1,701
  examples with JUSTIFIED = 1 and TRUE = 1, exactly one has BELIEVES = 0. With
  near-total collinearity there is no support on which an incremental effect of
  belief could show. "Belief contributes nothing" would overstate this.

### H2c could not be tested: the Gettier subset is deterministic

The prediction was that residual cost would concentrate on `lucky_guessed = 1`
cases. The measured gap runs the other way and its interval excludes zero
(JTB/MLP: 0.004 bits on Gettier cases against 0.144 on the rest, gap -0.140,
95% interval [-0.188, -0.097]). The comparison is confounded beyond repair:

| Subset | n (evaluation) | KNOWS = 1 rate | Marginal entropy |
|:--|--:|--:|--:|
| lucky_guessed = 1 | 344 | 0.0000 | 0.0000 bits |
| lucky_guessed = 0 | 596 | 0.4178 | 0.9804 bits |

A subset with zero marginal entropy is cheaper to code than one with 0.98 bits
whatever the probe does, so the prediction was not falsifiable as written.

Two different things were being run together here and are now separated.
`lucky_guessed = 1` is a *judgement the model produced*; `gettier = True` is a
*factor of the scenario design*. On the common cohort the design marks 418
Gettier scenarios; the model grants KNOWS in 0 of them and calls 395 of them a
lucky guess. The model's own luck judgement covers 1,563 examples, so it is the
broader category. Both show near-total exclusion of KNOWS, but they are not the
same variable and the report should not use one as a name for the other. The
underlying fact is the finding, and it is a sharp one: across the whole cohort
the model called an example a lucky guess and also said the subject knows in 2
of 1,563 cases, and in the evaluation split never. **The model's KNOWS excludes
epistemic luck without exception.** Caveat: both labels are model outputs on a
factorial design in which luck varies separately from truth and justification,
so this is a consistency property of the model's own answers, not a scenario
artifact -- but it is not independent evidence that the model finds Gettier
cases *hard*. It finds them easy, and resolves them in one direction.

## E4 -- Depth profile: confirmed for the depth, partly for the ratio

Minimum-codelength layer per predicate, over a 15-point depth grid for the four
new predicates (emb, blocks 4, 8, 10, 12, 16, 20, 24, 26, 29, 30, 32, 34, 36,
40) and the complete 40-layer sweep for KNOWS. Bits per label, mean over five
seeds.

| Predicate | MLP min layer | MLP min | MLP final/min | Linear min layer | Linear min | Linear final/min |
|:--|:--|--:|--:|:--|--:|--:|
| believes | block29 | 0.108 | 1.99 | block29 | 0.091 | 1.59 |
| justified | block26 | 0.139 | 1.65 | block26 | 0.126 | 1.52 |
| knows | block29 | 0.148 | 1.64 | block28 | 0.149 | 1.38 |
| true | block26 | 0.152 | 1.71 | block26 | 0.122 | 1.41 |
| lucky_guessed | block26 | 0.195 | 1.53 | block26 | 0.187 | 1.34 |

- **Depth prediction confirmed, ten times out of ten.** Every predicate, under
  both probe kinds, reaches its minimum between block 26 and block 29 of 40 --
  65% to 72% of depth, inside the declared 24-36 window and in a much narrower
  band than the window allowed. Five different epistemic predicates agreeing to
  within three layers is evidence that the depth is a property of the model's
  processing, not of the concept being decoded.
- **Ratio prediction confirmed for the MLP, not for the linear probe.** Final
  layer costs 1.53x to 1.99x the minimum under the MLP, all above the declared
  1.4x. Under the linear probe two of five fall below it: knows at 1.38 and
  lucky_guessed at 1.34. The claim as stated fails for those two cells and is
  reported as failing.
- The four new predicates are minima over a 15-point grid rather than a full
  sweep. KNOWS, which has all 40 layers, puts its minimum at the same place under
  the coarse and the complete grid, so the grid is not hiding the optimum.

## E3 -- Ordering intervals: FIRST OUTCOME SUPERSEDED, rerun pending (see corrections)

400 paired scenario-group resamples on the 4,970-example cohort stable for all
five predicates, zero degenerate resamples, linear probe, block 29. Primary
statistic is normalized MDL.

| Predicate | Median | 95% interval |
|:--|--:|:--|
| justified | 0.1408 | [0.1018, 0.1958] |
| true | 0.1458 | [0.1100, 0.2013] |
| believes | 0.1482 | [0.1063, 0.2338] |
| knows | 0.1608 | [0.1195, 0.2267] |
| lucky_guessed | 0.2502 | [0.1829, 0.3732] |

Probability of being cheapest: justified 0.400, believes 0.357, true 0.207,
knows 0.033, lucky_guessed 0.003. Probability of being dearest: lucky_guessed
0.935, knows 0.035, believes 0.025, true 0.003, justified 0.003.

- **The "cheapest" half of the prediction fails.** Believes was predicted to be
  cheapest in at least 80% of draws; it is cheapest in 35.7%, behind justified at
  40.0%. No predicate comes close to the threshold. The point-estimate ordering
  also moved between the descriptive table (per-predicate stable subsets, bits
  per label, believes cheapest) and this one (common cohort, normalized MDL,
  justified cheapest), which is itself evidence that the fine ordering is not a
  stable quantity.
- **The "dearest" half holds.** Lucky_guessed is dearest in 93.5% of draws,
  above the declared 80%.
- **The non-separation prediction holds, and then some.** Justified, knows, and
  true were predicted not to separate, and they do not. Neither does anything
  else: only 2 of 10 paired differences have 95% intervals excluding zero, both
  against lucky_guessed (justified -0.111 [-0.235, -0.015] and true -0.103
  [-0.233, -0.016]). Believes and knows do not separate from lucky_guessed
  either, at 0.970 and 0.963 probability but intervals that graze zero.

**What survives.** One cross-predicate claim: *lucky guess* is the most
expensive of the five rules to decode. Believes, justified, knows, and true sit
between 0.141 and 0.161 normalized MDL with heavily overlapping intervals and
are not distinguishable by this measurement. In particular **knows is not
separated from any of its components**.

This is the experiment's job. The descriptive five-way ordering that motivated
it does not survive scenario resampling, and reporting it as a ranking would
have been an overclaim. What the paper can say is the lucky-guess separation,
the non-separation of the rest, and -- from E2 -- the composition result, which
does not depend on the ranking.


---

# Corrections after external review (2026-09-14)

An independent review of the report found two technical defects that invalidate
numbers above. Both are confirmed against the code and both are fixed with
regression tests; the affected experiments are rerunning from the `analysis_v11`
snapshot. The original outcome sections are left in place, struck where wrong.

## Correction 1: the compressed rungs of E1 were not valid descriptions

`payload_bytes` measured the compression artifact after deleting its
standardizer arrays. But the weights inside that artifact had been quantized in
*standardized-input* space, so without the standardizer they cannot be applied
to raw activations: the 5,357-byte "description" could not reproduce the probe.
The same defect affected the 177,334-byte MLP rung. Folding was verified as
exact, but folding was never actually performed before compression.

The corrected procedure folds the standardizer into the first affine layer,
saves that probe with an identity standardizer, compresses *the folded probe*,
selects on raw tuning activations under the declared acceptance rule, and
reports on raw evaluation activations. `payload_bytes` now refuses any artifact
whose stored standardizer is not the identity, so the invalid measurement cannot
be repeated. Folded weights `w / s` have a far wider dynamic range than
standardized ones, so coarser quantization is expected to fail selection; the
reviewer's in-memory replication found 4-bit collapsing (macro-F1 ~ 0.18) and
the rule selecting 8 or 16 bits, at roughly 13.5 kB and a factor near 386.

**Corrected E1 result (rerun from `analysis_v11`, exit 0).** Linear probe, folded
then compressed: the acceptance rule selects 16-bit quantization on three seeds
and 8-bit on two; 4-bit collapses to the majority class (macro-F1 0.22) and
8-bit sits at the boundary. Median payload **13,210 bytes**, untouched-
evaluation macro-F1 **0.980**, a reduction of **397x** against the 5,244,932-byte
MLP -- 2.6 orders of magnitude, so the "about three" prediction is partial, and
the "< 10,000 bytes" sub-prediction fails. MLP, folded then compressed: SVD rank
8 or 16, median payload 177,506 bytes, macro-F1 0.982. Linear d90 stays at 512
and the MLP's at 1,024; both subspace rungs remain more expensive than direct
compression once the per-predicate standardizer is counted (43,016 and 45,064
bytes).

Also corrected: the standardizer is fit per predicate on that predicate's
stable coding rows, so it is *not* automatically "shared overhead across
predicates". The subspace rung's honest number is the standalone one; the
column formerly called `marginal_bytes` is renamed `bytes_if_standardizer_shared`
and describes a design that was not run.

## Correction 2: the E3 bootstrap merged duplicated groups

`paired_indices` resampled scenario groups with replacement but kept each
copy's original identifier. The online coder then called `np.unique(groups)`,
merging every repeated copy into a single oversized group. On the coding split
that meant roughly 455 distinct identifiers reaching the coder out of 720
sampled instances (63%), which changes the number of groups, the group
shuffle, and every prequential endpoint. The point estimates were informative;
the intervals, the 35.7% and the 93.5% were not.

`paired_resample` now labels every sampled copy distinctly and is the primitive
behind both the ordering bootstrap and the joint refit driver, which carried the
same latent defect. A regression test checks that the number of distinct labels
reaching the coder equals the number of sampled instances in every split.

**Every summary of the defective bootstrap is withdrawn** -- medians, intervals
and probabilities alike -- because the defect changed the effective size, order
and endpoints of every draw, so its point estimates were not valid estimates
either. Only the unresampled per-cell estimates stand, as descriptive results.

## Correction 3: the corrected bootstrap trained on a single probe seed

A second review found that `run_ordering_bootstrap` trained every draw with
`cfg["seeds"][0]` only, although the protocol declares five seeds. Its
cheapest/dearest percentages would therefore be conditional on seed 2026 and
carry no training variability. The single-seed run is left to finish as a
seed-2026 result and as a control on the group-identifier fix, and a
protocol-conforming run -- 400 draws x five seeds, averaged per draw before any
ranking -- was launched alongside it from `analysis_v12`. The confirmatory
analysis is the five-seed one; if only the single-seed run is ever reported it
must be labelled "scenario bootstrap conditional on a fixed probe seed".

## E3 control result: corrected bootstrap, single probe seed (2026)

400 paired draws from `analysis_v11`, zero degenerate, every sampled copy its
own group, probe seed fixed at 2026. **Conditional on that seed; not the
confirmatory analysis**, which is the five-seed run.

| Predicate | Median normalized MDL | 95% interval |
|:--|--:|:--|
| justified | 0.0954 | [0.0728, 0.1317] |
| believes | 0.0990 | [0.0664, 0.1469] |
| true | 0.1053 | [0.0811, 0.1393] |
| knows | 0.1127 | [0.0875, 0.1508] |
| lucky_guessed | 0.1764 | [0.1341, 0.2491] |

Cheapest: justified 0.495, believes 0.403, true 0.087, knows 0.015. Dearest:
lucky_guessed 0.968, knows 0.028, true 0.005.

- The absolute levels sit about 30% below the withdrawn run's (justified 0.141
  there): merging duplicated groups had inflated every codelength, which is
  direct evidence that the withdrawn medians were not valid estimates.
- **believes-cheapest prediction fails** again: 40.3% against a declared 80%.
- **lucky_guessed-dearest holds**: 96.8%.
- **Non-separation holds.** Three of ten paired differences exclude zero, all
  against lucky_guessed (justified -0.082 [-0.161, -0.015]; believes -0.077
  [-0.151, -0.019]; true +0.071 [+0.010, +0.150]); knows-vs-lucky misses by
  0.002 at the upper end. Believes, justified, knows and true remain mutually
  unseparated; knows separates from none of its components.

## E3 confirmatory result: corrected bootstrap, five probe seeds

400 paired draws from `analysis_v12`, zero degenerate, every sampled copy its
own group, probe seeds 2026-2030 averaged per draw before any ranking. **This
is the analysis the protocol calls for.**

| Predicate | Median normalized MDL | 95% interval |
|:--|--:|:--|
| justified | 0.0966 | [0.0814, 0.1149] |
| believes | 0.1039 | [0.0816, 0.1265] |
| true | 0.1071 | [0.0891, 0.1282] |
| knows | 0.1131 | [0.0956, 0.1373] |
| lucky_guessed | 0.1796 | [0.1497, 0.2212] |

Cheapest: justified 0.618, believes 0.310, true 0.072, knows 0.000. Dearest:
lucky_guessed 1.000.

- **lucky_guessed-dearest confirmed**: dearest in 100% of draws, against a
  declared 80%.
- **believes-cheapest fails**: 31.0%, against a declared 80%; justified leads at
  61.8%.
- **The non-separation prediction fails in part.** Five of ten paired
  differences exclude zero: all four predicates against lucky_guessed, and
  **justified against knows** (-0.0168 [-0.0288, -0.0065], justified cheaper in
  every draw). Averaging over seeds removed training noise that the single-seed
  control had left in the intervals (there the same pair was -0.017 [-0.038,
  +0.003]). Justified-true, knows-true, believes-knows, believes-true and
  believes-justified remain unseparated. So knows is separated from one of its
  components -- justification is cheaper to decode -- and not from the other
  two; the difference is about 15% of the codelength, small next to the
  lucky_guessed gap of roughly 60%.
- Levels agree with the single-seed control to within 0.005 across the board;
  the control run's only material difference was wider intervals.

**Scope and three refinements from the third review.** E3 compares normalized
MDL under a linear probe at block 29 on the common cohort; it does not measure
absolute complexity or each predicate's own minimum. Averaging the five fixed
seeds estimates mean performance over those seeds and *reduces* training
variability rather than incorporating it: the intervals are scenario-resampling
intervals for that average, not a joint bootstrap over scenarios and seeds.
Per seed, the justified-minus-knows difference has the same sign throughout
(medians -0.0154 to -0.0174; justified cheaper in 95-98% of draws) but its
95% interval contains zero for seeds 2026, 2027 and 2029 and touches zero for
2028 and 2030; the separation is clear for the average, not for any single
training. The averaged pair also separates at 99.5% ([-0.0367, -0.0033]),
as do the same five pairs under that conservative check, though with 400
draws those tails are imprecise. And non-separation is not equality: "knows
does not separate from believes or true" is supported; "they are equally
complex" is not.

**What the study can claim about ordering.** Two things, both with intervals
excluding zero under paired scenario resampling of the five-seed average: the
model's lucky-guess rule is the most expensive of the five to decode, and its
justification rule is cheaper to decode than its knowledge rule. Belief, truth,
knowledge and justification are otherwise not distinguished by this
measurement -- which is not a finding that they are equally complex.

## Presentation corrections adopted from the second review

- The 397x decomposes as 256x from replacing the MLP with a linear probe
  (5,244,932 -> 20,484 bytes) and 1.55x from quantization (20,484 -> 13,210);
  compressing the MLP without changing architecture gives 29.5x. Nearly all of
  the reduction comes from the linear probe sufficing, not from quantization.
- "Complete description" is replaced by "serialized payload of the map,
  conditional on the declared probe format and decoder": the bytes exclude the
  generic decoder code and do not identify model, layer or architecture.
- Sizes are medians of five seeds with ranges: linear 13,210 (8,089-13,210,
  two seeds accept 8 bits); MLP 177,506 (177,506-349,570, two seeds need SVD
  rank 16). Stored values are reported separately from base-probe parameters
  (43,529 or 86,545 for the SVD rungs).
- Wording: believes alone (0.596) "improves only marginally on the random
  projection (0.618), with no evidence of separation"; "a linear readout of
  the three logits does not match the nonlinear readout" rather than "combining
  them is not a linear operation"; Gettier cases "show no higher decoding cost
  under this protocol" rather than "are easy"; LLC chains "pass the standard
  mixing diagnostics nominally".
- The depth figure marks KNOWS at c26 because blocks 26 and 29 tie (0.1514) on
  the 15-point grid; the full sweep puts the minimum at block 28, which the
  table uses.
- No externally timestamped registry existed, so the hypotheses are
  "prospectively prespecified predictions", not a preregistration.
