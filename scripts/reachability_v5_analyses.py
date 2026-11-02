from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from wcsmatrix.io import evaluation_prompts, read_jsonl, write_json
from wcsmatrix.local import source_texts
from wcsmatrix.metrics import coverage, paired_bootstrap, word_groups

SCORER_DEFAULTS = {
    "shipped": {
        "qwen-2b": {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "provenance": "generation_config.json (thinking default)"},
        "qwen-4b": {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "provenance": "generation_config.json (thinking default)"},
        "qwen-9b": {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "provenance": "generation_config.json (thinking default)"},
        "qwen-27b-int8": {"temperature": 0.6, "top_k": 20, "top_p": 0.95, "provenance": "generation_config.json (thinking default)"},
        "qwen-32b-int8": {"temperature": 0.7, "top_k": 20, "top_p": 0.80, "provenance": "generation_config.json"},
        "gemma-e2b": {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "provenance": "generation_config.json & model card"},
        "gemma-e4b": {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "provenance": "generation_config.json & model card"},
        "gemma-12b": {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "provenance": "generation_config.json & model card"},
        "gemma-31b": {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "provenance": "generation_config.json & model card"},
        "llama-3.2-1b": {"temperature": 0.6, "top_k": 100, "top_p": 0.90, "provenance": "generation_config.json"},
        "llama-3.1-8b": {"temperature": 0.6, "top_k": 100, "top_p": 0.90, "provenance": "generation_config.json"},
        "mistral-7b": {"temperature": 1.0, "top_k": 100, "top_p": 0.95, "provenance": "none shipped; fallback top-100 p=0.95"},
    },
    "card_nonthinking": {
        "qwen-2b": {"temperature": 0.7, "top_k": 20, "top_p": 0.80, "provenance": "Qwen3.5 card non-thinking mode"},
        "qwen-4b": {"temperature": 0.7, "top_k": 20, "top_p": 0.80, "provenance": "Qwen3.5 card non-thinking mode"},
        "qwen-9b": {"temperature": 0.7, "top_k": 20, "top_p": 0.80, "provenance": "Qwen3.5 card non-thinking mode"},
        "qwen-27b-int8": {"temperature": 0.7, "top_k": 20, "top_p": 0.80, "provenance": "Qwen3.5 card non-thinking mode"},
        "qwen-32b-int8": {"temperature": 0.7, "top_k": 20, "top_p": 0.80, "provenance": "generation_config.json"},
        "gemma-e2b": {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "provenance": "Gemma 4 card & generation_config.json"},
        "gemma-e4b": {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "provenance": "Gemma 4 card & generation_config.json"},
        "gemma-12b": {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "provenance": "Gemma 4 card & generation_config.json"},
        "gemma-31b": {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "provenance": "Gemma 4 card & generation_config.json"},
        "llama-3.2-1b": {"temperature": 0.6, "top_k": 100, "top_p": 0.90, "provenance": "generation_config.json"},
        "llama-3.1-8b": {"temperature": 0.6, "top_k": 100, "top_p": 0.90, "provenance": "generation_config.json"},
        "mistral-7b": {"temperature": 1.0, "top_k": 100, "top_p": 0.95, "provenance": "none shipped; fallback top-100 p=0.95"},
    },
    "unscaled_T1": {
        "qwen-2b": {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "provenance": "shipped (k=20, p=0.95) at T=1.0"},
        "qwen-4b": {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "provenance": "shipped (k=20, p=0.95) at T=1.0"},
        "qwen-9b": {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "provenance": "shipped (k=20, p=0.95) at T=1.0"},
        "qwen-27b-int8": {"temperature": 1.0, "top_k": 20, "top_p": 0.95, "provenance": "shipped (k=20, p=0.95) at T=1.0"},
        "qwen-32b-int8": {"temperature": 1.0, "top_k": 20, "top_p": 0.80, "provenance": "shipped (k=20, p=0.80) at T=1.0"},
        "gemma-e2b": {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "provenance": "shipped (k=64, p=0.95) at T=1.0"},
        "gemma-e4b": {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "provenance": "shipped (k=64, p=0.95) at T=1.0"},
        "gemma-12b": {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "provenance": "shipped (k=64, p=0.95) at T=1.0"},
        "gemma-31b": {"temperature": 1.0, "top_k": 64, "top_p": 0.95, "provenance": "shipped (k=64, p=0.95) at T=1.0"},
        "llama-3.2-1b": {"temperature": 1.0, "top_k": 100, "top_p": 0.90, "provenance": "shipped (k=100, p=0.90) at T=1.0"},
        "llama-3.1-8b": {"temperature": 1.0, "top_k": 100, "top_p": 0.90, "provenance": "shipped (k=100, p=0.90) at T=1.0"},
        "mistral-7b": {"temperature": 1.0, "top_k": 100, "top_p": 0.95, "provenance": "fallback (k=100, p=0.95) at T=1.0"},
    },
}


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def coverage_custom_filter(top_probs, target_top_index, groups, temperature=1.0, top_k=100, top_p=0.95):
    k = min(int(top_k), top_probs.shape[1])
    probs = np.asarray(top_probs[:, :k], dtype=np.float64)
    indices = np.asarray(target_top_index, dtype=int)
    if temperature != 1.0:
        probs = np.power(np.maximum(probs, 1e-30), 1.0 / float(temperature))
    probs = probs / probs.sum(axis=-1, keepdims=True)
    before = probs.cumsum(axis=-1) - probs
    valid = (indices >= 0) & (indices < k)
    survives = valid & (before[np.arange(len(indices)), indices.clip(0, k - 1)] <= float(top_p))
    words = [bool(np.all(survives[g])) for g in groups]
    return {"wcs": float(np.mean(words)), "covered_words": int(sum(words)), "total_words": len(words)}


def run_extended_analysis(config_path: str):
    config = yaml.safe_load(Path(config_path).read_text())
    exp_name = config["name"]
    audit_root = Path("results/audits") / exp_name
    report_dir = Path("results/reports") / exp_name
    report_dir.mkdir(parents=True, exist_ok=True)

    prompts = {p["id"] for p in evaluation_prompts(config)}
    scorers = [m["name"] for m in config["models"]]
    local_sources = [m["name"] for m in config["models"]]
    closed_frontier = [
        m["name"]
        for m in config.get("frontier_models", [])
        if m.get("metadata", {}).get("weights_access") != "open_weights"
    ]
    open_frontier = [
        m["name"]
        for m in config.get("frontier_models", [])
        if m.get("metadata", {}).get("weights_access") == "open_weights"
    ]
    sources = ["human"] + local_sources + closed_frontier + open_frontier
    seed = config["analysis"]["seed"]
    reps = config["analysis"]["bootstrap_replicates"]

    # Per-(scorer, source, prompt_id) records
    records = defaultdict(dict)
    tail_rows = []
    prefix150_records = defaultdict(dict)

    for scorer in scorers:
        scorer_dir = audit_root / scorer
        if not scorer_dir.exists():
            continue
        for source in sources:
            src_dir = scorer_dir / source
            if not src_dir.exists():
                continue
            for npz_path in sorted(src_dir.glob("*.npz")):
                pid = npz_path.stem
                if pid not in prompts:
                    continue
                meta = json.loads(npz_path.with_suffix(".json").read_text())
                with np.load(npz_path, allow_pickle=False) as arrays:
                    top_probs = arrays["top_probs"]
                    target_top_index = arrays["target_top_index"]
                    mass_before = arrays["mass_before"]
                    offsets = arrays["offsets"].tolist()
                groups = word_groups(meta["text"], offsets, config["scoring"]["word_policy"])
                groups_150 = groups[:150] if len(groups) >= 150 else groups

                entry = {
                    "tail_mass_top100": float(np.mean(1.0 - top_probs.sum(axis=-1))),
                    "top100_miss_rate": float(np.mean(target_top_index < 0)),
                    "total_words": len(groups),
                    "total_tokens": len(target_top_index),
                }
                for mode in ("realistic", "strict", "strict_full"):
                    for p in (0.9, 0.95, 0.99):
                        entry[(mode, p)] = coverage(
                            top_probs, target_top_index, groups, p, mode, mass_before
                        )["wcs"]
                for variant, table in SCORER_DEFAULTS.items():
                    if scorer in table:
                        spec = table[scorer]
                        entry[("default", variant)] = coverage_custom_filter(
                            top_probs,
                            target_top_index,
                            groups,
                            spec["temperature"],
                            spec["top_k"],
                            spec["top_p"],
                        )["wcs"]
                records[(scorer, source)][pid] = entry
                prefix150_records[(scorer, source)][pid] = coverage(
                    top_probs, target_top_index, groups_150, 0.99, "realistic", mass_before
                )["wcs"]

    available_scorers = [s for s in scorers if (audit_root / s).exists()]
    if not available_scorers:
        print(f"No audited scorers found for {exp_name}")
        return

    # 1. Tail-mass & Top-100 vs Full-Vocabulary Validation
    for scorer in available_scorers:
        for source in sources:
            cell = records.get((scorer, source), {})
            if not cell:
                continue
            pids = sorted(cell)
            real_99 = [cell[i][("realistic", 0.99)] for i in pids]
            full_99 = [cell[i][("strict_full", 0.99)] for i in pids]
            real_95 = [cell[i][("realistic", 0.95)] for i in pids]
            full_95 = [cell[i][("strict_full", 0.95)] for i in pids]
            boot_99 = paired_bootstrap(full_99, real_99, seed, reps)
            tail_rows.append(
                {
                    "scorer": scorer,
                    "source": source,
                    "n": len(pids),
                    "mean_tail_mass_outside_top100": float(np.mean([cell[i]["tail_mass_top100"] for i in pids])),
                    "mean_top100_miss_rate": float(np.mean([cell[i]["top100_miss_rate"] for i in pids])),
                    "wcs_realistic_0.99": float(np.mean(real_99)),
                    "wcs_strict_full_0.99": float(np.mean(full_99)),
                    "full_minus_realistic_0.99": boot_99["delta"],
                    "full_minus_realistic_0.99_ci_low": boot_99["ci_low"],
                    "full_minus_realistic_0.99_ci_high": boot_99["ci_high"],
                    "wcs_realistic_0.95": float(np.mean(real_95)),
                    "wcs_strict_full_0.95": float(np.mean(full_95)),
                    "full_minus_realistic_0.95": float(np.mean(full_95) - np.mean(real_95)),
                }
            )
    write_csv(report_dir / "tail-mass-and-full-vocab-validation.csv", tail_rows)

    # 2. Scorer-default filter matrices & human-minus-source contrasts
    default_cells = []
    default_contrasts = []
    for variant in ("shipped", "card_nonthinking", "unscaled_T1"):
        for scorer in available_scorers:
            if scorer not in SCORER_DEFAULTS[variant]:
                continue
            spec = SCORER_DEFAULTS[variant][scorer]
            human_cell = records.get((scorer, "human"), {})
            for source in sources:
                cell = records.get((scorer, source), {})
                if not cell:
                    continue
                pids = sorted(cell)
                vals = [cell[i][("default", variant)] for i in pids]
                default_cells.append(
                    {
                        "variant": variant,
                        "scorer": scorer,
                        "source": source,
                        "temperature": spec["temperature"],
                        "top_k": spec["top_k"],
                        "top_p": spec["top_p"],
                        "provenance": spec["provenance"],
                        "n": len(vals),
                        "wcs": float(np.mean(vals)),
                        "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    }
                )
                if source != "human" and human_cell:
                    shared = sorted(set(human_cell) & set(cell))
                    if shared:
                        h_vals = [human_cell[i][("default", variant)] for i in shared]
                        g_vals = [cell[i][("default", variant)] for i in shared]
                        default_contrasts.append(
                            {
                                "variant": variant,
                                "scorer": scorer,
                                "source": source,
                                **paired_bootstrap(h_vals, g_vals, seed, reps),
                            }
                        )
    write_csv(report_dir / "matrix-scorer-default.csv", default_cells)
    write_csv(report_dir / "human-minus-source-scorer-default.csv", default_contrasts)

    # Determine scorers that have complete coverage across human and all frontier source tiers
    core_sources = ["human"] + open_frontier + closed_frontier
    complete_scorers = [
        s
        for s in available_scorers
        if all(bool(records.get((s, src))) for src in core_sources)
    ]
    eval_scorers = complete_scorers if complete_scorers else available_scorers

    # 3. Table 4: Per-source human-minus-source averaged across eval_scorers (with prompt-paired bootstrap)
    # Plus length-matched first-150-words sensitivity check
    source_summary_rows = []
    filter_conditions = [
        ("realistic", 0.99),
        ("realistic", 0.95),
        ("realistic", 0.90),
        ("strict_full", 0.99),
        ("strict_full", 0.95),
        ("default", "shipped"),
        ("default", "card_nonthinking"),
        ("default", "unscaled_T1"),
    ]
    for cond in filter_conditions:
        cond_label = f"{cond[0]}_{cond[1]}"
        for source in sources:
            if source == "human":
                continue
            source_scorers = [s for s in eval_scorers if bool(records.get((s, source)))]
            if not source_scorers:
                continue
            common_pids = None
            for scorer in source_scorers:
                h_pids = set(records.get((scorer, "human"), {}))
                g_pids = set(records.get((scorer, source), {}))
                both = h_pids & g_pids
                common_pids = both if common_pids is None else (common_pids & both)
            if not common_pids:
                continue
            pids = sorted(common_pids)
            h_prompt_means = [
                float(np.mean([records[(s, "human")][i][cond] for s in source_scorers])) for i in pids
            ]
            g_prompt_means = [
                float(np.mean([records[(s, source)][i][cond] for s in source_scorers])) for i in pids
            ]
            boot = paired_bootstrap(h_prompt_means, g_prompt_means, seed, reps)
            tier = (
                "small_open_weight"
                if source in local_sources
                else ("large_open_weight" if source in open_frontier else "frontier_proprietary")
            )
            row = {
                "condition": cond_label,
                "source": source,
                "tier": tier,
                "scorers_count": len(source_scorers),
                "mean_human_wcs": float(np.mean(h_prompt_means)),
                "mean_source_wcs": float(np.mean(g_prompt_means)),
                **boot,
            }
            if cond == ("realistic", 0.99):
                h_150 = [
                    float(np.mean([prefix150_records[(s, "human")][i] for s in source_scorers]))
                    for i in pids
                ]
                g_150 = [
                    float(np.mean([prefix150_records[(s, source)][i] for s in source_scorers]))
                    for i in pids
                ]
                boot_150 = paired_bootstrap(h_150, g_150, seed, reps)
                row["prefix150_delta"] = boot_150["delta"]
                row["prefix150_ci_low"] = boot_150["ci_low"]
                row["prefix150_ci_high"] = boot_150["ci_high"]
            source_summary_rows.append(row)
    write_csv(report_dir / "source-reachability-summary.csv", source_summary_rows)

    # 4. Tier-level contrasts averaged across eval_scorers (including family-balanced large OW)
    tier_contrast_rows = []
    tier_defs = {
        "human": ["human"],
        "small_open_weight": [s for s in local_sources if s != "qwen-32b-int8" and all(records.get((sc, s)) for sc in eval_scorers)],
        "large_open_weight": [s for s in open_frontier if all(records.get((sc, s)) for sc in eval_scorers)],
        "frontier_proprietary": [s for s in closed_frontier if all(records.get((sc, s)) for sc in eval_scorers)],
    }
    pairs = [
        ("human", "small_open_weight"),
        ("human", "large_open_weight"),
        ("human", "frontier_proprietary"),
        ("large_open_weight", "small_open_weight"),
        ("large_open_weight", "frontier_proprietary"),
        ("frontier_proprietary", "small_open_weight"),
    ]
    for cond in filter_conditions:
        cond_label = f"{cond[0]}_{cond[1]}"
        for tier_a, tier_b in pairs:
            srcs_a, srcs_b = tier_defs[tier_a], tier_defs[tier_b]
            if not srcs_a or not srcs_b:
                continue
            common_pids = None
            for scorer in eval_scorers:
                for src in srcs_a + srcs_b:
                    pids_src = set(records.get((scorer, src), {}))
                    common_pids = pids_src if common_pids is None else (common_pids & pids_src)
            if not common_pids:
                continue
            pids = sorted(common_pids)
            a_means = [
                float(np.mean([records[(sc, sa)][i][cond] for sc in eval_scorers for sa in srcs_a]))
                for i in pids
            ]
            b_means = [
                float(np.mean([records[(sc, sb)][i][cond] for sc in eval_scorers for sb in srcs_b]))
                for i in pids
            ]
            boot = paired_bootstrap(a_means, b_means, seed, reps)
            tier_contrast_rows.append(
                {
                    "condition": cond_label,
                    "contrast": f"{tier_a}_minus_{tier_b}",
                    "weighting": "equal_source",
                    "scorers_count": len(eval_scorers),
                    "mean_a": float(np.mean(a_means)),
                    "mean_b": float(np.mean(b_means)),
                    **boot,
                }
            )
    write_csv(report_dir / "tier-contrasts-summary.csv", tier_contrast_rows)
    print(f"Completed v5 reachability analysis for {exp_name} ({len(eval_scorers)} complete scorers / {len(available_scorers)} total)")


if __name__ == "__main__":
    import sys

    for arg in sys.argv[1:]:
        run_extended_analysis(arg)
