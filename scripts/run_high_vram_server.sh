#!/usr/bin/env bash
# ==============================================================================
# High-VRAM GPU Server Script (>= 64 GiB combined free VRAM, e.g. 2x 48GB A6000 or 80GB H100/A100)
# ==============================================================================
# Usage on the high-VRAM server:
#   git pull
#   ./run.sh [--env-file /path/to/.env]
# ==============================================================================

set -euo pipefail

cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
unset HF_HUB_DISABLE_IMPLICIT_TOKEN || true
if [[ -z "${HF_TOKEN:-}" ]]; then
    if [[ -f ".hf_token" ]]; then
        export HF_TOKEN="$(tr -d '[:space:]' < .hf_token)"
    elif [[ -f "$HOME/.cache/huggingface/token" ]]; then
        export HF_TOKEN="$(tr -d '[:space:]' < "$HOME/.cache/huggingface/token")"
    fi
fi

ENV_FILE=""
if [[ "${1:-}" == "--env-file" && -n "${2:-}" ]]; then
    ENV_FILE="$2"
fi

echo "=== [0/6] Environment & GPU Preflight Check ==="
uv sync --python 3.12 --extra gpu --extra dev
uv run ruff check src tests
uv run --extra gpu pytest -q

# Verify combined free GPU memory >= 62 GiB before loading Gemma-4-31B-it in bf16
uv run --extra gpu python -c '
import torch
assert torch.cuda.is_available(), "CUDA is required"
num_gpus = torch.cuda.device_count()
infos = [torch.cuda.mem_get_info(i) for i in range(num_gpus)]
free_gib = sum(f for f, _ in infos) / (2**30)
total_gib = sum(t for _, t in infos) / (2**30)
names = " + ".join(f"GPU {i}: {torch.cuda.get_device_name(i)} ({infos[i][0]/(2**30):.1f}/{infos[i][1]/(2**30):.1f} GiB)" for i in range(num_gpus))
print(f"Detected {num_gpus} GPU(s): {names} | Combined Free: {free_gib:.1f} GiB / Total: {total_gib:.1f} GiB")
if free_gib < 62.0:
    raise SystemExit(
        f"ERROR: Gemma-4-31B-it in bf16 requires >= 62 GiB combined free VRAM (found {free_gib:.1f} GiB across {num_gpus} GPU(s))."
    )
'

# Seed baseline per-prompt scores from git-tracked compressed cache so we only need
# to run GPU audits for gemma-31b and qwen-27b-int8 (saving ~3.5 hours of redundant runs)
uv run python -c '
import gzip, json
from pathlib import Path

cache_path = Path("results/reports/baseline_scores_cache.json.gz")
if cache_path.exists():
    with gzip.open(cache_path, "rt", encoding="utf-8") as f:
        cache = json.load(f)
    scores_dir = Path("results/scores")
    scores_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "hc3-50-expanded-open-frontier-detok-20261101.jsonl": "hc3-50-expanded-gemma31b-detok-20261102.jsonl",
        "hc3-wiki-50-expanded-frontier-20261102.jsonl": "hc3-wiki-50-expanded-gemma31b-20261102.jsonl",
        "hc3-50-top1000-expanded-detok-20261102.jsonl": "hc3-50-top1000-gemma31b-detok-20261102.jsonl",
    }
    for src_name, dst_name in mapping.items():
        dst = scores_dir / dst_name
        if src_name in cache and not dst.exists():
            dst.write_text("\n".join(json.dumps(r) for r in cache[src_name]) + "\n")
            print(f"Seeded {dst} ({len(cache[src_name])} baseline rows)")
'

# Optional: Complete the single missing MiMo-V2.6-Pro response (eli5_537) if API env file is provided
if [[ -n "$ENV_FILE" ]]; then
    echo "=== [Optional] Completing missing frontier responses via explicit env file ==="
    uv run wcsmatrix --config configs/open-frontier.yaml frontier --env-file "$ENV_FILE" --workers 2 --max-cost-usd 5
fi

echo "=== [1/6] Running 2-Prompt GPU Smoke Test for Gemma-4-31B-it (bf16) ==="
[[ -f data/texts/smoke-local-gemma31b.jsonl ]] || cp data/texts/smoke-local.jsonl data/texts/smoke-local-gemma31b.jsonl
uv run --extra gpu wcsmatrix --config configs/high-vram-eli5-gemma31b.yaml preflight
uv run --extra gpu wcsmatrix --config configs/high-vram-smoke-gemma31b.yaml preflight
uv run --extra gpu wcsmatrix --config configs/high-vram-smoke-gemma31b.yaml generate --model gemma-31b --limit 2
uv run --extra gpu wcsmatrix --config configs/high-vram-smoke-gemma31b.yaml audit --model gemma-31b --limit 2
echo "Smoke test passed!"

echo "=== [2/6] HC3 ELI5: Generating & Auditing Gemma-4-31B-it (bf16) ==="
[[ -f data/texts/local-expanded-gemma31b.jsonl ]] || cp data/texts/local-expanded.jsonl data/texts/local-expanded-gemma31b.jsonl

# Generate 50 ELI5 answers from gemma-31b (greedy, T=0)
uv run --extra gpu wcsmatrix --config configs/high-vram-eli5-gemma31b.yaml generate --model gemma-31b

# Audit gemma-31b as scorer across all 21 ELI5 sources
echo "--- Auditing ELI5 scorer: gemma-31b (all 21 sources) ---"
uv run --extra gpu wcsmatrix --config configs/high-vram-eli5-gemma31b.yaml audit --model gemma-31b

# Audit the 9 existing scorers ONLY on the newly generated gemma-31b source
for model in qwen-4b qwen-9b qwen-27b-int8 qwen-32b-int8 gemma-e4b gemma-12b llama-3.2-1b llama-3.1-8b mistral-7b; do
    echo "--- Auditing ELI5 scorer: $model on new source gemma-31b ---"
    uv run --extra gpu wcsmatrix --config configs/high-vram-eli5-gemma31b.yaml audit --model "$model" --source gemma-31b
done
uv run wcsmatrix --config configs/high-vram-eli5-gemma31b.yaml analyze --allow-partial

echo "=== [3/6] HC3 Wikipedia CS/AI: Generating & Auditing Gemma-4-31B-it & Qwen-27B-int8 ==="
[[ -f data/texts/wiki-local-expanded-gemma31b.jsonl ]] || cp data/texts/wiki-local-expanded.jsonl data/texts/wiki-local-expanded-gemma31b.jsonl

# Generate 50 Wikipedia CS/AI answers from gemma-31b and qwen-27b-int8
for model in gemma-31b qwen-27b-int8; do
    uv run --extra gpu wcsmatrix --config configs/high-vram-wiki-gemma31b.yaml generate --model "$model"
done

# Audit gemma-31b and qwen-27b-int8 as scorers across all 20 Wikipedia CS/AI sources
for model in gemma-31b qwen-27b-int8; do
    echo "--- Auditing Wiki CS/AI scorer: $model (all 20 sources) ---"
    uv run --extra gpu wcsmatrix --config configs/high-vram-wiki-gemma31b.yaml audit --model "$model"
done

# Audit the 7 existing bf16 scorers ONLY on the 2 newly generated Wiki sources
for model in qwen-4b qwen-9b gemma-e4b gemma-12b llama-3.2-1b llama-3.1-8b mistral-7b; do
    echo "--- Auditing Wiki CS/AI scorer: $model on new sources gemma-31b & qwen-27b-int8 ---"
    uv run --extra gpu wcsmatrix --config configs/high-vram-wiki-gemma31b.yaml audit --model "$model" --source gemma-31b --source qwen-27b-int8
done
uv run wcsmatrix --config configs/high-vram-wiki-gemma31b.yaml analyze --allow-partial

echo "=== [4/6] Top-1000 Validation (K=1000): Auditing Gemma-4-31B-it & Peer Scorers ==="
echo "--- Auditing Top-1000 scorer: gemma-31b (all 21 sources) ---"
uv run --extra gpu wcsmatrix --config configs/high-vram-top1000-gemma31b.yaml audit --model gemma-31b

for model in qwen-9b gemma-12b llama-3.1-8b mistral-7b; do
    echo "--- Auditing Top-1000 scorer: $model on new source gemma-31b ---"
    uv run --extra gpu wcsmatrix --config configs/high-vram-top1000-gemma31b.yaml audit --model "$model" --source gemma-31b
done

echo "=== [5/6] Running Extended Reachability Analyses ==="
uv run python scripts/reachability_v5_analyses.py \
    configs/high-vram-eli5-gemma31b.yaml \
    configs/high-vram-wiki-gemma31b.yaml

echo "=== [6/6] Recomputing Paper Tables, Regenerating Figures & Building Overleaf Package ==="
uv run python scripts/recompute_revised_nov2.py
uv run python scripts/generate_revised_figures.py
uv run python scripts/build_revised_main_tex.py

echo "=== High-VRAM Server Pipeline Completed Successfully! ==="
echo "Updated artifacts ready:"
echo "  - results/reports/revised_nov2_recomputed.json"
echo "  - overleaf/main.tex & overleaf/figures_nov2/"
echo "  - WCS_revised_2_Nov_updated.pdf"
echo "  - WCS_revised_2_Nov_updated.zip"
