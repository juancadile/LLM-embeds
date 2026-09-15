#!/usr/bin/env bash
# Launches the LLM-embeds rerun agent on the DGX inside a detached tmux session.
# Safe to re-run: it will not start a second agent if one is already running.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
REPO="$HOME/LLM-embeds"
VENV="$HOME/venvs/llm-embeds"
SESSION="llm-embeds-rerun"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists. Attach with: tmux attach -t $SESSION"; exit 0
fi

# --- preflight ---
grep -q CLONE_EXIT=0 "$HOME/clone.log" || { echo "clone not finished/failed, see ~/clone.log"; exit 1; }
grep -q PIP_DONE "$HOME/venvs/llm-embeds-install.log" || { echo "venv install not finished, see ~/venvs/llm-embeds-install.log"; exit 1; }
"$VENV/bin/python" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
claude auth status >/dev/null 2>&1 || { echo "Claude Code is not logged in on this machine. Run: claude auth login"; exit 1; }

# --- repo setup ---
cd "$REPO"
git config user.name  "juancadile"
git config user.email "juancadile@users.noreply.github.com"
cp -n "$HOME/AGENT_BRIEF.md" "$HOME/Comments - Conceptual analysis complexity and ML.md" . 2>/dev/null || true
git rev-parse --verify rerun-2026 >/dev/null 2>&1 || git checkout -b rerun-2026
git checkout -q rerun-2026
git add AGENT_BRIEF.md "Comments - Conceptual analysis complexity and ML.md"
git diff --cached --quiet || git commit -q -m "Add agent brief and reviewer comments for rerun"
mkdir -p results src

# --- launch ---
cat > "$HOME/rerun-prompt.txt" <<'PROMPT'
You are running unattended on a DGX Spark inside tmux. Nobody will answer questions; make reasonable decisions yourself and record them in results/REPORT.md.
Environment: activate the Python venv with `source ~/venvs/llm-embeds/bin/activate` before any python command (torch with CUDA, transformers, pandas, scipy, scikit-learn, rapidfuzz, inflect are installed). The GPU is an NVIDIA GB10 with 128 GB unified memory, CUDA 13. HF models already cached: Qwen/Qwen3-14B, Qwen/Qwen3-1.7B, meta-llama/Llama-3.1-8B-Instruct. Do not `git push`; commit locally only.
Your full instructions are in AGENT_BRIEF.md in this directory. Read it, then read the file it names, then execute all three phases in order and obey its stop conditions.
PROMPT

# The agent runs with permission prompts disabled because nobody is attached to answer them.
tmux new-session -d -s "$SESSION" -c "$REPO" \
  "export PATH=\"$HOME/.local/bin:\$PATH\"; source $VENV/bin/activate; claude --dangerously-skip-permissions \"\$(cat $HOME/rerun-prompt.txt)\" 2>&1 | tee -a $HOME/rerun-console.log; echo; echo '[agent process exited]'; sleep 999999"
echo "Started. Attach with:  tmux attach -t $SESSION   (detach with Ctrl-b d)"
