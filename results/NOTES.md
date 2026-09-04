### What the cached OPT vectors are (established 2026-09-04)

The cached OPT-1.3B vectors (`opt/1_3B.txt`) are the **residual stream before OPT's final layer norm**, not `last_hidden_state` as produced by any current transformers build. Evidence, all in `results/`:

1. Re-extracting with the notebook recipe under current transformers (5.16) gives vectors with cosine 0.992 to the cache but **0.45× the norm** (per word: p5 0.38, p50 0.45, p95 0.53; `phase3.json`, opt-1.3b postln side).
2. Capturing both sides of `decoder.final_layer_norm` with a forward hook: the **pre-LN side matches the cache with cosine 1.0000 and norm ratio 0.999 (CV 0.002) over all 5,124 vocab words**; the post-LN side is the 0.45× one (`phase3.json`, opt-1.3b preln/postln).
3. Reproduction under the pinned version, transformers 4.20.1 (`src/repro_tf420.py`, throwaway venv): the notebook function verbatim gives the **post-LN** vectors (cosine 0.99995, norm ratio 1.0000 to the current re-extraction; `repro_tf420.json`). So 4.20.1 by itself did not drop the norm.
4. Loading under 4.20.1 with `_remove_final_layer_norm=True` (the OPT config flag of that era that disables the final LayerNorm, emitting the "final_layer_norm.weight/bias were not used" warning seen in the notebooks) reproduces the cache to **max abs difference 4e-4** (`repro_tf420_noln.json`). That flag is the mechanism; which config revision set it on the cluster in 2022 is a Hub-history question, not a vector question.
5. `opt/1_3B.txt.orig` is the same extraction on the older 3,471-word vocab (cosine 1.0000, ratio 1.000 on shared words); it is not a different generation.

Consequences for the magnitude study: the paper's OPT magnitude numbers were computed on pre-LN residual-stream vectors. Post-LN vectors are normalised per token and their norms carry little per-word information by construction, so a magnitude test on them is trivially near-null. Phase 3 therefore reports the 74-pair test on **both** sides, labelled, and the paper's method section has to say which one it means. Whether the same holds for T5 (whose encoder also ends in a norm) is answered by the t5-large / t5-3b rows of the same table.
