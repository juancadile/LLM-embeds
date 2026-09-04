# LLM-embeds rerun — report

Generated 2026-09-04 15:03 by `python -m src.run_all` at git 4576936; python 3.11.4, numpy 1.26.4, host MacBook-Pro-5.local.
Seed 2026 throughout. Original notebooks untouched; code in `src/`.

## Phase 1

_Not run (no results/phase1.json)._

## Phase 2 — magnitude study, redone on the cached embeddings

### 2.1 Coverage: pairs with both words in the cached vocabulary

| model | us_uk | plural | verb |
| :--- | ---: | ---: | ---: |
| opt-1.3b | 2/11 | 3/27 | 5/36 |
| opt-13b | 2/11 | 3/27 | 5/36 |
| t5-large | 2/11 | 3/27 | 5/36 |
| t5-3b | 2/11 | 3/27 | 5/36 |
| flan-t5-xxl | 2/11 | 3/27 | 5/36 |

The notebooks embedded out-of-vocabulary words on the fly and never saved them, so from the caches only these pairs can be scored. Missing words are listed in results/phase2.json; the GPU re-extraction in Phase 3 (`phase2_extra`) fills them where the original model could be loaded.

### 2.2 |%diff| in L1, raw — similar pairs vs 5000 random pairs (mean / median; p = Mann–Whitney, similar < random)

| model | random | us_uk | plural | verb | all pooled |
| :--- | :--- | :--- | :--- | :--- | :--- |
| opt-1.3b | 15.7 / 13.3 | 3.9 / 3.9 (n=2, p=0.0487*) | 10.8 / 7.9 (n=3, p=0.269) | 2.5 / 3.0 (n=5, p=0.00111**) | 5.3 / 3.3 (n=10, p=0.000601***) |
| opt-13b | 12.5 / 10.4 | 6.5 / 6.5 (n=2, p=0.192) | 7.8 / 9.5 (n=3, p=0.225) | 11.3 / 11.6 (n=5, p=0.519) | 9.3 / 8.0 (n=10, p=0.221) |
| t5-large | 13.3 / 9.4 | 21.1 / 21.1 (n=2, p=0.627) | 13.3 / 3.3 (n=3, p=0.321) | 15.6 / 6.0 (n=5, p=0.567) | 16.0 / 5.9 (n=10, p=0.504) |
| t5-3b | 12.8 / 9.7 | 1.1 / 1.1 (n=2, p=0.015*) | 11.2 / 13.9 (n=3, p=0.578) | 17.5 / 12.0 (n=5, p=0.842) | 12.4 / 11.5 (n=10, p=0.439) |
| flan-t5-xxl | 7.7 / 6.1 | 8.5 / 8.5 (n=2, p=0.743) | 3.1 / 2.5 (n=3, p=0.0792) | 2.4 / 2.1 (n=5, p=0.0121*) | 3.8 / 2.8 (n=10, p=0.0191*) |

### 2.2 |%diff| in L2, raw — similar pairs vs 5000 random pairs (mean / median; p = Mann–Whitney, similar < random)

| model | random | us_uk | plural | verb | all pooled |
| :--- | :--- | :--- | :--- | :--- | :--- |
| opt-1.3b | 16.5 / 13.8 | 1.8 / 1.8 (n=2, p=0.0187*) | 16.5 / 16.8 (n=3, p=0.615) | 3.9 / 4.0 (n=5, p=0.00435**) | 7.3 / 4.6 (n=10, p=0.00434**) |
| opt-13b | 30.5 / 24.3 | 36.5 / 36.5 (n=2, p=0.588) | 17.9 / 22.1 (n=3, p=0.231) | 5.6 / 4.9 (n=5, p=0.00409**) | 15.5 / 5.0 (n=10, p=0.0149*) |
| t5-large | 14.5 / 9.9 | 24.9 / 24.9 (n=2, p=0.718) | 18.0 / 5.6 (n=3, p=0.486) | 18.0 / 8.9 (n=5, p=0.707) | 19.4 / 8.1 (n=10, p=0.734) |
| t5-3b | 12.8 / 9.7 | 1.7 / 1.7 (n=2, p=0.0246*) | 10.9 / 13.8 (n=3, p=0.544) | 17.3 / 12.5 (n=5, p=0.842) | 12.3 / 12.1 (n=10, p=0.457) |
| flan-t5-xxl | 6.4 / 5.1 | 6.0 / 6.0 (n=2, p=0.624) | 3.0 / 2.7 (n=3, p=0.113) | 2.3 / 1.6 (n=5, p=0.0195*) | 3.2 / 2.6 (n=10, p=0.0239*) |

### 2.2 |%diff| in L1, center — similar pairs vs 5000 random pairs (mean / median; p = Mann–Whitney, similar < random)

| model | random | us_uk | plural | verb | all pooled |
| :--- | :--- | :--- | :--- | :--- | :--- |
| opt-1.3b | 18.9 / 16.0 | 9.6 / 9.6 (n=2, p=0.172) | 11.7 / 8.1 (n=3, p=0.199) | 3.2 / 2.9 (n=5, p=0.00136**) | 7.1 / 4.0 (n=10, p=0.00134**) |
| opt-13b | 15.3 / 12.9 | 10.1 / 10.1 (n=2, p=0.306) | 16.6 / 18.4 (n=3, p=0.725) | 17.0 / 15.5 (n=5, p=0.737) | 15.5 / 13.8 (n=10, p=0.708) |
| t5-large | 12.0 / 10.0 | 13.4 / 13.4 (n=2, p=0.556) | 2.9 / 3.3 (n=3, p=0.0188*) | 7.2 / 8.4 (n=5, p=0.147) | 7.1 / 5.1 (n=10, p=0.0347*) |
| t5-3b | 13.8 / 11.3 | 13.9 / 13.9 (n=2, p=0.538) | 9.0 / 9.7 (n=3, p=0.29) | 12.4 / 14.9 (n=5, p=0.48) | 11.7 / 10.0 (n=10, p=0.383) |
| flan-t5-xxl | 14.9 / 12.5 | 14.3 / 14.3 (n=2, p=0.512) | 2.0 / 2.1 (n=3, p=0.00657**) | 3.9 / 3.2 (n=5, p=0.00508**) | 5.4 / 3.0 (n=10, p=0.000787***) |

### 2.2 |%diff| in L2, center — similar pairs vs 5000 random pairs (mean / median; p = Mann–Whitney, similar < random)

| model | random | us_uk | plural | verb | all pooled |
| :--- | :--- | :--- | :--- | :--- | :--- |
| opt-1.3b | 19.7 / 16.7 | 9.9 / 9.9 (n=2, p=0.171) | 15.8 / 10.7 (n=3, p=0.382) | 5.2 / 3.8 (n=5, p=0.00558**) | 9.3 / 7.0 (n=10, p=0.00862**) |
| opt-13b | 23.5 / 20.5 | 27.1 / 27.1 (n=2, p=0.639) | 27.8 / 33.1 (n=3, p=0.716) | 7.4 / 7.2 (n=5, p=0.00831**) | 17.4 / 10.1 (n=10, p=0.111) |
| t5-large | 12.1 / 10.0 | 9.5 / 9.5 (n=2, p=0.346) | 3.1 / 3.0 (n=3, p=0.0203*) | 5.8 / 5.3 (n=5, p=0.0568) | 5.7 / 4.4 (n=10, p=0.00785**) |
| t5-3b | 13.4 / 10.9 | 13.5 / 13.5 (n=2, p=0.537) | 8.9 / 8.4 (n=3, p=0.298) | 12.7 / 13.6 (n=5, p=0.535) | 11.7 / 10.0 (n=10, p=0.426) |
| flan-t5-xxl | 14.3 / 11.9 | 14.4 / 14.4 (n=2, p=0.584) | 2.0 / 2.0 (n=3, p=0.00771**) | 3.5 / 3.9 (n=5, p=0.0037**) | 5.2 / 3.5 (n=10, p=0.00089***) |

**Reading (L2, raw, pooled).** opt-1.3b: similar pairs closer (median 4.6 vs 13.8, p=0.0043); opt-13b: similar pairs closer (median 5.0 vs 24.3, p=0.015); t5-large: no significant difference (median 8.1 vs 9.9); t5-3b: no significant difference (median 12.1 vs 9.7); flan-t5-xxl: similar pairs closer (median 2.6 vs 5.1, p=0.024). 
The paper's sentence "similar pairs were not closer in magnitude than random pairs" is a claim about the pooled comparison; where p < 0.05 in the "closer" direction it is contradicted on the authors' own pairs. Where results differ across models, the defensible statement is "no consistent effect across models". None of this tests a complexity–magnitude *correlation*: the design only checks that magnitude is invariant under near-synonymy, a necessary condition.

### 2.3 Tokens per word (tokenizer of each model family, word alone, no special tokens)

| model | words with >1 token (count) |
| :--- | :--- |
| opt-1.3b | alumnae=4, alumna=3, alumnus=3, appendixes=3, bacterium=3, maneuver=3, manoeuvre=3, nuclei=3, nucleus=3, paediatric=3, phenomena=3, phenomenon=3, travelled=3, alumni=2 … |
| opt-13b | alumnae=4, alumna=3, alumnus=3, appendixes=3, bacterium=3, maneuver=3, manoeuvre=3, nuclei=3, nucleus=3, paediatric=3, phenomena=3, phenomenon=3, travelled=3, alumni=2 … |
| t5-large | alumnae=6, apologise=6, alumna=5, alumnus=5, appendixes=5, appendices=4, analogue=3, appendix=3, bacterium=3, chewed=3, cries=3, indices=3, nuclei=3, nucleus=3 … |
| t5-3b | alumnae=6, apologise=6, alumna=5, alumnus=5, appendixes=5, appendices=4, analogue=3, appendix=3, bacterium=3, chewed=3, cries=3, indices=3, nuclei=3, nucleus=3 … |
| flan-t5-xxl | alumnae=6, apologise=6, alumna=5, alumnus=5, appendixes=5, appendices=4, analogue=3, appendix=3, bacterium=3, chewed=3, cries=3, indices=3, nuclei=3, nucleus=3 … |

Because the cached OPT/T5 vectors are means over subword tokens (T5 also averages in the EOS token), a pair whose members differ in token count is not a clean magnitude comparison. Outliers in the pair tables above line up with these words.

### 2.4 L1 vs L2 and rogue dimensions

| model | Pearson(L1, L2) | Spearman | top-5 variance dims: mean share of squared L2 norm |
| :--- | :--- | :--- | :--- |
| opt-1.3b | 0.957 | 0.973 | d1346: 5.1%; d1359: 2.1%; d1279: 1.2%; d614: 1.0%; d1326: 0.5% |
| opt-13b | -0.185 | -0.007 | d902: 28.8%; d4960: 1.9%; d3964: 1.9%; d440: 0.2%; d387: 0.5% |
| t5-large | 0.971 | 0.937 | d79: 2.5%; d58: 3.1%; d300: 1.5%; d674: 1.1%; d801: 0.5% |
| t5-3b | 0.995 | 0.993 | d792: 0.2%; d71: 0.2%; d17: 0.2%; d751: 0.2%; d923: 0.2% |
| flan-t5-xxl | 0.973 | 0.965 | d1478: 11.5%; d1814: 0.1%; d2152: 0.1%; d3799: 0.1%; d892: 0.5% |

The notebooks' "magnitude" was L1 (sum of absolute components). Where L1 and L2 are weakly correlated, the choice of norm changes the result; a dimension holding a large share of squared norm on its own is a rogue dimension (Timkey & van Schijndel 2021) and dominates L2 but not L1.

_Phase 2 ran in 14 s._

## Phase 3

_Not run (no results/phase3.json)._
