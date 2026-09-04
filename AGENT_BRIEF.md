# Agent brief — LLM-embeds rerun (R&R for "Conceptual Complexity, Conceptual Analysis, and Machine Learning")

You are working in a clone of `maxb00/LLM-embeds` on a DGX Spark. Read `Comments - Conceptual analysis complexity and ML.md` first: it says what's wrong with the paper's experiments and is the spec for what you're rebuilding. The original notebooks are the *record*, not the codebase — do not edit them.

## Ground rules

- Work on branch `rerun-2026`. New code goes in `src/`, outputs in `results/`, one markdown report at `results/REPORT.md`. Never modify `*.ipynb`, `gpt/`, `opt/`, `t5/`, `vocab/`.
- Every number in the report must be reproducible by one command: `python -m src.run_all`. Seed everything.
- Cached embeddings are whitespace floats, one row per line, aligned by position to `vocab/expanded_vocab.txt` (5,124 words). `gpt/gpt_babbage.txt` is aligned to `vocab/valid_vocab.txt` (3,471) instead — don't mix. `gpt/gpt_ada-v2.txt` is an LFS pointer, the data was never pushed; ignore it.
- Load caches with `pandas.read_csv(sep=" ", header=None, dtype=float32).dropna(axis=1)` (4 s for Curie). Save `.npy` copies under `results/cache/` on first load.
- OpenAI's `text-similarity-*-001` endpoints are retired. Don't try to call them.
- Commit after each phase with a message that says what the numbers were. If a phase can't be completed, write why in REPORT.md and move on; don't loop on it.

## Phase 1 — Loaders and controls on the cached embeddings (no GPU)

1. `src/load.py`: load any cache by name → (vocab list, float32 matrix). Assert shapes.
2. `src/similarity.py`: vectorised `most_similar(vec, k, exclude)`, `define_ab(a, b)`, and `define2(target, k)` as a matrix product over all pairs (top-k pruning on `V @ t` is fine; must run in seconds, not hours). Support optional mean-centering and all-but-the-top (Mu & Viswanath 2018).
3. Reproduce the p. 16 table on Curie exactly (cosines to 4 decimals). This is the pipeline check.
4. Controls, for every (a, b, target) on p. 16 and for `bachelor`:
   - rank and cosine of target for v(b) alone, v(a) alone, v(a)+v(b);
   - random-pair baseline: for 1,000 seeded random pairs, cosine of the best non-a/b neighbour; report mean, p10, p90;
   - all of the above raw and mean-centered.
5. Rerun define2 for every target on pp. 18–19 with *one* algorithm (cognate filter = plural + prefix + edit-distance ≤ 2 against target), on Curie, Davinci, OPT-13B, T5-3B, flan-T5-XXL. Report top-3 per target with cosine, raw and centered, plus a null: define2 on 100 random unit vectors, distribution of the best score.
6. Fix the `mikolov()` bug (model name lands in the `end` slot) in a new `src/analogy.py` and report king − man + woman on each model, raw and centered. Not in the paper; it's for the record.

Acceptance: REPORT.md has the p. 16 reproduction, the control table, the baseline, and the define2 table, each with a one-paragraph plain-language reading.

## Phase 2 — Magnitude study, redone (cached embeddings first, then GPU)

The paper's claim ("similar pairs not closer in magnitude than random pairs") was never computed; the notebooks print signed per-pair L1 diffs and never aggregate.

1. Recover the word-pair lists from `magnitudes*.ipynb` (US/UK spellings, singular/plural, verb forms; fix the typos `talkling`, `indicies`). Put them in `src/pairs.py`.
2. For each cached model (OPT-1.3B, OPT-13B, T5-large, T5-3B, flan-T5-XXL): |%diff| in L1 and in L2, raw and mean-centered, for similar pairs vs. 5,000 seeded random pairs. Mann-Whitney per category and pooled. Report token counts per word so tokenization outliers are visible.
3. Report L1-vs-L2 correlation per model and the top-5 dimensions by variance (OPT-13B has a rogue dimension holding ~11% of squared norm — confirm or refute).

Acceptance: one table, models × categories, with p-values, plus a paragraph saying which of "no correlation", "no consistent effect", or "similar pairs closer" the data supports.

## Phase 3 — 2026 open model (GPU)

Use the largest Qwen3 already on the box (check `~/.cache/huggingface` and `ollama list` / whatever is there; do not download anything over 40 GB without leaving a note). Fall back to Llama-3.x-8B or OLMo-2 if needed.

1. `src/extract.py`: embed all 5,124 vocab words with explicit, switchable choices: layer ∈ {input embedding, 25%, 50%, 75%, final}; pooling ∈ {mean over tokens, last token}; context ∈ {word alone, "The word is: {w}"}; drop BOS/EOS always. Save each configuration as `results/cache/{model}_{layer}_{pool}_{ctx}.npy`. Log tokens-per-word.
2. Rerun Phase 1 steps 3–5 and Phase 2 step 2 on every configuration.
3. Report which configuration best reproduces the Curie behaviour on the p. 16 table, and whether the magnitude result changes with layer/pooling/context.

Acceptance: a configuration table and a recommendation for the paper's method section (which vectors, why).

## Stop conditions

Stop when Phase 3's acceptance is met, or after the third failed attempt at any single step, or if disk under 20 GB. On stop: make sure REPORT.md is current, commit, and print the last 30 lines of REPORT.md.
