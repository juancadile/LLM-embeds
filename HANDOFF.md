# Hand-off — Phase 3 on the DGX Spark

You are running on a DGX Spark, unattended, inside the `LLM-embeds` repo on branch `rerun-2026`. Nobody will answer questions. Your job is Phase 3 of `AGENT_BRIEF.md` only. Read that file and `results/REPORT.md` first.

## What already exists

- Phases 1 and 2 are done and committed by another agent. `results/phase1*`, `results/phase2*`, and the modules they depend on (`src/common.py`, `src/load.py`, `src/similarity.py`, `src/filters.py`, `src/analogy.py`, `src/pairs.py`, `src/phase1.py`, `src/phase2.py`) are frozen: do not edit, rerun, or overwrite them. If you need a change to a shared module, put it in a new file `src/phase3_ext.py` and import from there.
- Phase 3 is already implemented: `src/extract.py` (grid extraction, sidecars) and `src/phase3.py` (evaluation of every configuration by importing the Phase 1/2 functions, plus the original OPT/T5 recipe re-extraction for the magnitude pairs). Entry point: `python -m src.run_all --phase 3 --models <hf ids>`. Extraction caches under `results/cache/` are reused automatically; a stale cache (wrong row count) is re-extracted.
- A Phase 3 job is running or has finished in tmux session `phase3`, logging to `results/phase3.log`, for `Qwen/Qwen3-14B`, `meta-llama/Llama-3.1-8B-Instruct` and `Qwen/Qwen3-1.7B` (all in the HF cache). Run `tmux ls` and read the log before starting anything. If it is alive, watch it and take over only when it exits. If it died, read the traceback, fix, and relaunch the same command; it continues from the cached configurations, not from zero.
- Configurations extracted before sidecars existed have none. Run `python -c "from src.extract import backfill_sidecars; print(backfill_sidecars(['Qwen/Qwen3-14B','meta-llama/Llama-3.1-8B-Instruct','Qwen/Qwen3-1.7B']))"` once to create them.
- Environment: `source ~/venvs/llm-embeds/bin/activate`. `TORCH_DISABLE_NATIVE_JIT=1` is set in `src/common.py`; keep it (triton cannot compile here, no root). GPU: GB10, 128 GB unified memory, CUDA 13.

## Hard rules

1. No model downloads above 40 GB. Use what's already in `~/.cache/huggingface` or on disk. `google/flan-t5-xxl` (≈45 GB) is therefore skipped by the code; leave it skipped and say so in REPORT.md.
2. Check free disk before every model load (`df -h /`). If it's under 20 GB, stop, commit, and write why.
3. Every configuration's embeddings go to `results/cache/{model}_{layer}_{pool}_{ctx}.npy` with a sidecar `.json` recording model id, revision, dtype, layer index, pooling, context template, and tokens-per-word stats. A configuration without a sidecar doesn't count as done.
4. Commit after every completed model (or configuration, if you relaunch per configuration) with the config names and the p. 16 reproduction scores in the message. Do not push (no GitHub credentials on this machine); the MacBook side pulls from here.
5. Do not modify `*.ipynb`, `gpt/`, `opt/`, `t5/`, `vocab/`, or the paper. Do not touch `Comments - *.md` or `AGENT_BRIEF.md`.
6. If the same step fails three times, log it, mark the configuration as skipped in REPORT.md, and move on. Never loop on a single failure.

## Work

- Grid: layer ∈ {input embedding, 25%, 50%, 75%, final} × pooling ∈ {mean over tokens, last token} × context ∈ {word alone, "The word is: {w}"}. BOS/EOS are never pooled. Extraction is batched; 5,178 words (vocab + magnitude-pair words + the three define2 targets missing from the vocab) per model takes minutes.
- For each configuration the code runs the Phase 1 evaluations (p. 16 reproduction, definiendum-rank controls, random-pair baseline, define2 on the pp. 18–19 targets, raw and mean-centred) and the Phase 2 magnitude test (L1 and L2, similar pairs vs. 5,000 seeded random pairs, Mann–Whitney). It imports the Phase 1–2 functions; do not reimplement them.
- Results go to `results/phase3.json` (one entry per configuration) and `python -m src.run_all --report-only` regenerates `results/REPORT.md`, which has the configuration table (definiendum ranks for the seven p. 16 cases raw and centred, random-baseline mean, pooled magnitude p-value). Keep REPORT.md current after every completed model.
- Then write `results/NOTES.md` (it is folded into REPORT.md by `--report-only`) with: which configuration best matches the Curie behaviour on the p. 16 table; whether the magnitude result changes across layer/pooling/context and across Phase 2 (cached), the original-recipe re-extraction, and the Phase 3 grid; which of "no correlation", "no consistent effect", or "similar pairs closer" the data supports; and a one-sentence recommendation for the paper's method section (which vectors, why). Also list any deviation from AGENT_BRIEF.md.

## Done

Stop when the grid is complete for every model, or a hard rule triggers. Before stopping: NOTES.md written, `python -m src.run_all --report-only` run, everything committed. Then print the last 40 lines of REPORT.md and the output of `git log --oneline -10`.
