#!/usr/bin/env python3
"""
Comprehensive recomputation for WCS_revised_2_Nov:
1. Core 4 bf16 scorers (qwen-4b, qwen-9b, gemma-e4b, gemma-12b) + qwen-27b-int8
   across the 16 ELI5 sources (dropping qwen-2b and gemma-e2b).
2. Sampled T>0 decoding ablation (4 bf16 scorers, 15 sources) using detokenized human reference.
3. 4-Family Expanded ELI5 audit (9 scorers: 7 bf16 + 2 int8, across 20 sources, dropping qwen-2b and gemma-e2b).
4. Wikipedia CS/AI replication (4 bf16 scorers across 15 sources, plus any expanded Wiki scorers/sources).
5. Top-100 vs. Top-1000 vs. Full-Vocabulary Nucleus Validation (including Small Open-Weight mean over the 5 sources,
   plus Llama-3.1-8B and Mistral-7B when available).
6. Orthographic exclusion sensitivity check for Same-Family Advantage (excluding curly apostrophes, standalone
   em-dashes, and 'alright').
7. Appendix A: Zipf-adjusted lexical category and loss-rate taxonomy (Tables 1 & 2 and inline statistics).
"""

from __future__ import annotations

import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from wcsmatrix.metrics import coverage, paired_bootstrap, word_groups

SCORER_DEFAULTS = {
    "shipped": {
        "qwen-4b": {"temperature": 0.6, "top_k": 20, "top_p": 0.95},
        "qwen-9b": {"temperature": 0.6, "top_k": 20, "top_p": 0.95},
        "qwen-27b-int8": {"temperature": 0.6, "top_k": 20, "top_p": 0.95},
        "qwen-32b-int8": {"temperature": 0.7, "top_k": 20, "top_p": 0.80},
        "gemma-e4b": {"temperature": 1.0, "top_k": 64, "top_p": 0.95},
        "gemma-12b": {"temperature": 1.0, "top_k": 64, "top_p": 0.95},
        "gemma-31b": {"temperature": 1.0, "top_k": 64, "top_p": 0.95},
        "llama-3.2-1b": {"temperature": 0.6, "top_k": 100, "top_p": 0.90},
        "llama-3.1-8b": {"temperature": 0.6, "top_k": 100, "top_p": 0.90},
        "mistral-7b": {"temperature": 1.0, "top_k": 100, "top_p": 0.95},
    },
    "card_nonthinking": {
        "qwen-4b": {"temperature": 0.7, "top_k": 20, "top_p": 0.80},
        "qwen-9b": {"temperature": 0.7, "top_k": 20, "top_p": 0.80},
        "qwen-27b-int8": {"temperature": 0.7, "top_k": 20, "top_p": 0.80},
        "qwen-32b-int8": {"temperature": 0.7, "top_k": 20, "top_p": 0.80},
        "gemma-e4b": {"temperature": 1.0, "top_k": 64, "top_p": 0.95},
        "gemma-12b": {"temperature": 1.0, "top_k": 64, "top_p": 0.95},
        "gemma-31b": {"temperature": 1.0, "top_k": 64, "top_p": 0.95},
        "llama-3.2-1b": {"temperature": 0.6, "top_k": 100, "top_p": 0.90},
        "llama-3.1-8b": {"temperature": 0.6, "top_k": 100, "top_p": 0.90},
        "mistral-7b": {"temperature": 1.0, "top_k": 100, "top_p": 0.95},
    },
    "unscaled_T1": {
        "qwen-4b": {"temperature": 1.0, "top_k": 20, "top_p": 0.95},
        "qwen-9b": {"temperature": 1.0, "top_k": 20, "top_p": 0.95},
        "qwen-27b-int8": {"temperature": 1.0, "top_k": 20, "top_p": 0.95},
        "qwen-32b-int8": {"temperature": 1.0, "top_k": 20, "top_p": 0.80},
        "gemma-e4b": {"temperature": 1.0, "top_k": 64, "top_p": 0.95},
        "gemma-12b": {"temperature": 1.0, "top_k": 64, "top_p": 0.95},
        "gemma-31b": {"temperature": 1.0, "top_k": 64, "top_p": 0.95},
        "llama-3.2-1b": {"temperature": 1.0, "top_k": 100, "top_p": 0.90},
        "llama-3.1-8b": {"temperature": 1.0, "top_k": 100, "top_p": 0.90},
        "mistral-7b": {"temperature": 1.0, "top_k": 100, "top_p": 0.95},
    },
}

CORE_4_SCORERS = ["qwen-4b", "qwen-9b", "gemma-e4b", "gemma-12b"]
CORE_5_SCORERS = ["qwen-4b", "qwen-9b", "qwen-27b-int8", "gemma-e4b", "gemma-12b"]
EXPANDED_9_SCORERS = [
    "qwen-4b",
    "qwen-9b",
    "qwen-27b-int8",
    "qwen-32b-int8",
    "gemma-e4b",
    "gemma-12b",
    "llama-3.2-1b",
    "llama-3.1-8b",
    "mistral-7b",
]
EXPANDED_7_BF16_SCORERS = [
    "qwen-4b",
    "qwen-9b",
    "gemma-e4b",
    "gemma-12b",
    "llama-3.2-1b",
    "llama-3.1-8b",
    "mistral-7b",
]

FRONTIER_PROP = ["gpt-6-astra", "claude-opus-5.5", "gemini-4-argon"]
LARGE_OW = [
    "deepseek-v4.1-flash-1017",
    "glm-5.3",
    "minimax-m3",
    "kimi-k3",
    "mimo-v2.6-pro",
    "deepseek-v4-pro-0813",
    "qwen3.8-2.4t-a95b",
]
SMALL_OW_CORE_5 = ["qwen-4b", "qwen-9b", "qwen-27b-int8", "gemma-12b", "gemma-e4b"]
SMALL_OW_BF16_4 = ["qwen-4b", "qwen-9b", "gemma-12b", "gemma-e4b"]
SMALL_OW_EXPANDED_9 = [
    "qwen-4b",
    "qwen-9b",
    "qwen-27b-int8",
    "qwen-32b-int8",
    "gemma-e4b",
    "gemma-12b",
    "llama-3.2-1b",
    "llama-3.1-8b",
    "mistral-7b",
]

FAMILY_MAP = {
    "qwen-4b": "qwen",
    "qwen-9b": "qwen",
    "qwen-27b-int8": "qwen",
    "qwen-32b-int8": "qwen",
    "qwen3.8-2.4t-a95b": "qwen",
    "gemma-e4b": "gemma",
    "gemma-12b": "gemma",
    "gemma-31b": "gemma",
    "llama-3.2-1b": "llama",
    "llama-3.1-8b": "llama",
    "mistral-7b": "mistral",
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "up", "about", "into", "over", "after", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should", "can", "could",
    "may", "might", "must", "it", "its", "they", "them", "their", "we", "us", "our", "you", "your",
    "i", "me", "my", "he", "him", "his", "she", "her", "that", "this", "these", "those", "which",
    "who", "whom", "what", "where", "when", "why", "how", "as", "so", "than", "then", "there", "here",
    "not", "no", "nor", "only", "own", "same", "too", "very", "just", "also",
}


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
    return float(np.mean(words))


def is_orthographic_confound_word(word_str: str) -> bool:
    # Check if word is standalone em-dash/punctuation, contains curly apostrophe, or is 'alright'
    clean = re.sub(r"^[^\w]+|[^\w]+$", "", word_str.lower())
    if not clean:
        return True
    if "’" in word_str or "‘" in word_str or "—" in word_str:
        return True
    if clean == "alright":
        return True
    return False


def load_suite_records(
    audit_dir: Path,
    scorers: list[str],
    sources: list[str],
    override_human_dir: Path | None = None,
):
    records = defaultdict(dict)
    for scorer in scorers:
        for source in sources:
            base_dir = override_human_dir if (source == "human" and override_human_dir is not None) else audit_dir
            src_dir = base_dir / scorer / source
            if not src_dir.exists():
                continue
            for npz_path in sorted(src_dir.glob("*.npz")):
                pid = npz_path.stem
                meta = json.loads(npz_path.with_suffix(".json").read_text())
                text = meta["text"]
                with np.load(npz_path, allow_pickle=False) as arrays:
                    top_probs = arrays["top_probs"]
                    target_top_index = arrays["target_top_index"]
                    mass_before = arrays["mass_before"]
                    offsets = arrays["offsets"].tolist()
                groups = word_groups(text, offsets, "whitespace")
                groups_150 = groups[:150] if len(groups) >= 150 else groups
                matches = list(re.finditer(r"\S+", text, flags=re.UNICODE))
                ortho_groups = [
                    g for m, g in zip(matches, groups) if not is_orthographic_confound_word(m.group())
                ]
                if not ortho_groups:
                    ortho_groups = groups

                entry = {
                    "tail_mass": float(np.mean(1.0 - top_probs.sum(axis=-1))),
                    "top_miss": float(np.mean(target_top_index < 0)),
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
                        )
                entry["prefix150_0.99"] = coverage(
                    top_probs, target_top_index, groups_150, 0.99, "realistic", mass_before
                )["wcs"]
                for p in (0.9, 0.95, 0.99):
                    entry[("ortho_clean", p)] = coverage(
                        top_probs, target_top_index, ortho_groups, p, "realistic", mass_before
                    )["wcs"]
                records[(scorer, source)][pid] = entry
    return records


def compute_tier_table_row(records, scorers, tier_sources, cond, seed=42, reps=10000):
    # Compute both unweighted mean of per-source means AND prompt-paired bootstrap contrasts
    tier_means = {}
    for tier_name, srcs in tier_sources.items():
        cell_means = []
        for s in srcs:
            for sc in scorers:
                vals = [v[cond] for v in records[(sc, s)].values() if cond in v]
                if vals:
                    cell_means.append(float(np.mean(vals)))
        tier_means[tier_name] = float(np.mean(cell_means)) if cell_means else None

    # Prompt-paired bootstrap contrasts across common prompts
    def paired_tier_contrast(tier_a, tier_b):
        srcs_a, srcs_b = tier_sources[tier_a], tier_sources[tier_b]
        # All-available prompt-paired mean: for each prompt, average over available sources in tier and scorers
        all_pids = sorted(
            set.intersection(
                *(
                    set(records[(sc, s)].keys())
                    for sc in scorers
                    for s in (srcs_a + srcs_b)
                    if records[(sc, s)]
                )
            )
        )
        a_vec, b_vec = [], []
        for pid in all_pids:
            a_vals = [
                records[(sc, s)][pid][cond]
                for sc in scorers
                for s in srcs_a
                if pid in records[(sc, s)] and cond in records[(sc, s)][pid]
            ]
            b_vals = [
                records[(sc, s)][pid][cond]
                for sc in scorers
                for s in srcs_b
                if pid in records[(sc, s)] and cond in records[(sc, s)][pid]
            ]
            a_vec.append(float(np.mean(a_vals)))
            b_vec.append(float(np.mean(b_vals)))
        return paired_bootstrap(a_vec, b_vec, seed=seed, replicates=reps)

    contrasts = {
        "large_ow_minus_small_ow": paired_tier_contrast("large_ow", "small_ow"),
        "large_ow_minus_frontier_prop": paired_tier_contrast("large_ow", "frontier_prop"),
        "human_minus_frontier_prop": paired_tier_contrast("human", "frontier_prop"),
        "human_minus_large_ow": paired_tier_contrast("human", "large_ow"),
        "human_minus_small_ow": paired_tier_contrast("human", "small_ow"),
        "frontier_prop_minus_human": paired_tier_contrast("frontier_prop", "human"),
    }
    return {"tier_means": tier_means, "contrasts": contrasts}


def compute_same_vs_cross_family(records, scorers, sources, cond=("realistic", 0.99), seed=42, reps=10000):
    same_cells, cross_cells = [], []
    for sc in scorers:
        sc_fam = FAMILY_MAP.get(sc)
        for src in sources:
            if src == sc or src == "human":
                continue
            src_fam = FAMILY_MAP.get(src)
            if src_fam is None:
                continue
            if not records[(sc, src)]:
                continue
            if sc_fam == src_fam:
                same_cells.append((sc, src))
            else:
                cross_cells.append((sc, src))
    common_pids = sorted(
        set.intersection(*(set(records[pair].keys()) for pair in (same_cells + cross_cells)))
    )
    same_prompt_means = [
        float(np.mean([records[pair][pid][cond] for pair in same_cells])) for pid in common_pids
    ]
    cross_prompt_means = [
        float(np.mean([records[pair][pid][cond] for pair in cross_cells])) for pid in common_pids
    ]
    boot = paired_bootstrap(same_prompt_means, cross_prompt_means, seed=seed, replicates=reps)
    # Also compute unweighted mean of 2-decimal rounded cells vs 4-decimal exact cells
    same_exact = float(np.mean([np.mean([v[cond] for v in records[p].values()]) for p in same_cells]))
    cross_exact = float(np.mean([np.mean([v[cond] for v in records[p].values()]) for p in cross_cells]))
    same_2dec = float(np.mean([round(float(np.mean([v[cond] for v in records[p].values()])), 2) for p in same_cells]))
    cross_2dec = float(np.mean([round(float(np.mean([v[cond] for v in records[p].values()])), 2) for p in cross_cells]))
    return {
        "same_count": len(same_cells),
        "cross_count": len(cross_cells),
        "exact_delta": same_exact - cross_exact,
        "rounded_2dec_delta": same_2dec - cross_2dec,
        **boot,
    }


def analyze_zipf_subset(audit_dir: Path, scorers: list[str], sources: list[str], p: float = 0.99):
    tier_map = {
        "human": "human",
        **{s: "frontier_proprietary" for s in FRONTIER_PROP},
        **{s: "large_open_weight" for s in LARGE_OW},
    }
    tier_cat_total = defaultdict(Counter)
    tier_cat_lost = defaultdict(Counter)
    tier_word_total = defaultdict(Counter)
    tier_word_lost = defaultdict(Counter)
    tier_raw_lost = defaultdict(Counter)
    tier_raw_total = defaultdict(Counter)

    # Also track mechanisms for 'the' and 'a' losses
    the_a_mechanisms = Counter()
    mistral_alright = [0, 0]
    mistral_lets = [0, 0]
    gemma_curly = defaultdict(lambda: [0, 0])

    for sc in scorers:
        for src in sources:
            if sc == src:
                continue  # Off-diagonal only
            src_dir = audit_dir / sc / src
            if not src_dir.exists():
                continue
            tier = tier_map.get(src, "small_open_weight")
            for npz_path in sorted(src_dir.glob("*.npz")):
                meta = json.loads(npz_path.with_suffix(".json").read_text())
                text = meta["text"]
                with np.load(npz_path, allow_pickle=False) as arrays:
                    top_probs = arrays["top_probs"]
                    target_top_index = arrays["target_top_index"]
                    offsets = arrays["offsets"].tolist()
                probs = np.asarray(top_probs, dtype=np.float64)
                probs = probs / probs.sum(axis=-1, keepdims=True)
                before = probs.cumsum(axis=-1) - probs
                indices = np.asarray(target_top_index, dtype=int)
                valid = (indices >= 0) & (indices < probs.shape[1])
                survives = valid & (before[np.arange(len(indices)), indices.clip(0, probs.shape[1] - 1)] <= p)

                matches = list(re.finditer(r"\S+", text, flags=re.UNICODE))
                groups = word_groups(text, offsets, "whitespace")
                if len(matches) != len(groups):
                    continue
                for idx_w, (m, g) in enumerate(zip(matches, groups)):
                    w_str = m.group()
                    w_surv = bool(np.all(survives[g]))
                    clean = re.sub(r"^[^\w]+|[^\w]+$", "", w_str.lower())
                    if not clean:
                        cat = "standalone_punct_or_symbol"
                        norm_w = w_str.lower()
                    elif clean in STOPWORDS:
                        cat = "function_stopword"
                        norm_w = clean
                    else:
                        cat = "content_lexical"
                        norm_w = clean

                    tier_cat_total[tier][cat] += 1
                    tier_word_total[tier][norm_w] += 1
                    tier_raw_total[tier][w_str.lower()] += 1
                    if not w_surv:
                        tier_cat_lost[tier][cat] += 1
                        tier_word_lost[tier][norm_w] += 1
                        tier_raw_lost[tier][w_str.lower()] += 1

                    # Track specific phenomena
                    if src == "mistral-7b":
                        if norm_w == "alright":
                            mistral_alright[1] += 1
                            if not w_surv:
                                mistral_alright[0] += 1
                        if norm_w in ("let's", "let’s"):
                            mistral_lets[1] += 1
                            if not w_surv:
                                mistral_lets[0] += 1
                    if src in ("gemma-e4b", "gemma-12b") and "’" in w_str:
                        lw = w_str.lower().strip(".,!?;:\"'()")
                        gemma_curly[lw][1] += 1
                        if not w_surv:
                            gemma_curly[lw][0] += 1

                    if not w_surv and norm_w in ("the", "a"):
                        first_tok = g[0]
                        max_p = float(probs[first_tok, 0])
                        is_cap = w_str[0].isupper() and (
                            idx_w == 0 or matches[idx_w - 1].group().endswith((".", "!", "?", ":"))
                        )
                        if max_p >= 0.95:
                            the_a_mechanisms["lock_in"] += 1
                        elif is_cap:
                            the_a_mechanisms["sentence_initial"] += 1
                        else:
                            the_a_mechanisms["mid_sentence"] += 1

    summary = {}
    for tier in ("human", "frontier_proprietary", "large_open_weight", "small_open_weight"):
        tot_lost = sum(tier_cat_lost[tier].values())
        tot_eval = sum(tier_cat_total[tier].values())
        cats = {}
        for cat in ("content_lexical", "function_stopword", "standalone_punct_or_symbol"):
            l_c = tier_cat_lost[tier][cat]
            t_c = tier_cat_total[tier][cat]
            cats[cat] = {
                "lost": l_c,
                "total": t_c,
                "loss_rate": l_c / t_c if t_c else 0.0,
                "share": l_c / tot_lost if tot_lost else 0.0,
            }
        cats["tier_total"] = {
            "lost": tot_lost,
            "total": tot_eval,
            "loss_rate": tot_lost / tot_eval if tot_eval else 0.0,
            "share": 1.0,
        }
        # Top raw count leaders
        raw_leaders = []
        for w, l_cnt in tier_word_lost[tier].most_common(12):
            t_cnt = tier_word_total[tier][w]
            raw_leaders.append({"word": w, "lost": l_cnt, "total": t_cnt, "rate": l_cnt / t_cnt})
        # Also check raw with curly apostrophe for small_open_weight
        curly_its = [
            sum(v for k, v in tier_raw_lost[tier].items() if k.strip(".,!?;:\"'()") == "it’s"),
            sum(v for k, v in tier_raw_total[tier].items() if k.strip(".,!?;:\"'()") == "it’s"),
        ]
        # Top zipf leaders (min_eval threshold: >= 55 for 9/11 scorers, or >= 20 for 4 scorers)
        min_eval = 45 if len(scorers) >= 7 else 20
        zipf_leaders = []
        for w, t_cnt in tier_word_total[tier].items():
            if t_cnt >= min_eval:
                l_cnt = tier_word_lost[tier][w]
                zipf_leaders.append({"word": w, "lost": l_cnt, "total": t_cnt, "rate": l_cnt / t_cnt})
        zipf_leaders.sort(key=lambda x: (-x["rate"], -x["lost"]))
        summary[tier] = {
            "categories": cats,
            "raw_leaders": raw_leaders,
            "zipf_leaders": zipf_leaders[:25],
            "curly_its": curly_its,
        }
    summary["phenomena"] = {
        "the_a_mechanisms": dict(the_a_mechanisms),
        "mistral_alright": mistral_alright,
        "mistral_lets": mistral_lets,
        "gemma_curly": dict(gemma_curly),
    }
    return summary


def serialize_records(records):
    out = {}
    for (sc, src), pid_map in records.items():
        if not pid_map:
            continue
        pair_key = f"{sc}||{src}"
        out[pair_key] = {}
        for pid, entry in pid_map.items():
            ser_entry = {}
            for k, v in entry.items():
                if isinstance(k, tuple):
                    ser_entry[f"__tuple__:{k[0]}:{k[1]}"] = round(float(v), 7)
                else:
                    ser_entry[k] = round(float(v), 7)
            out[pair_key][pid] = ser_entry
    return out


def deserialize_records(ser_dict):
    records = defaultdict(dict)
    for pair_key, pid_map in ser_dict.items():
        sc, src = pair_key.split("||", 1)
        for pid, ser_entry in pid_map.items():
            entry = {}
            for k, v in ser_entry.items():
                if k.startswith("__tuple__:"):
                    _, k0, k1 = k.split(":", 2)
                    try:
                        k1_val = float(k1)
                    except ValueError:
                        k1_val = k1
                    entry[(k0, k1_val)] = float(v)
                else:
                    entry[k] = float(v)
            records[(sc, src)][pid] = entry
    return records


def merge_records(base_records, extra_records):
    for pair, pid_map in extra_records.items():
        if pid_map:
            base_records[pair].update(pid_map)
    return base_records


def main():
    out = {}
    cache_path = Path("results/reports/baseline_prompt_cache.json.gz")
    cached_bundles = {}
    if cache_path.exists():
        with gzip.open(cache_path, "rt", encoding="utf-8") as f:
            cached_bundles = json.load(f)

    # 1. Load 4-Family Expanded ELI5 Audit (9 scorers across 20 sources, plus gemma-31b when audited)
    eli5_detok_dir = Path("results/audits/hc3-50-comparison-open-frontier-detok-20261101")
    exp_dir = Path("results/audits/hc3-50-expanded-open-frontier-detok-20261101")
    gemma31b_eli5_dir = Path("results/audits/hc3-50-expanded-gemma31b-detok-20261102")
    exp_sources_20 = ["human"] + FRONTIER_PROP + LARGE_OW + SMALL_OW_EXPANDED_9
    all_eli5_scorers = EXPANDED_9_SCORERS + ["gemma-31b"]
    all_eli5_sources = exp_sources_20 + ["gemma-31b"]

    exp_records = (
        deserialize_records(cached_bundles["exp_records"])
        if "exp_records" in cached_bundles
        else defaultdict(dict)
    )
    if exp_dir.exists():
        merge_records(exp_records, load_suite_records(exp_dir, all_eli5_scorers, all_eli5_sources))
    if gemma31b_eli5_dir.exists():
        merge_records(exp_records, load_suite_records(gemma31b_eli5_dir, all_eli5_scorers, all_eli5_sources))

    has_gemma31b_eli5 = bool(exp_records.get(("gemma-31b", "human")))
    out["has_gemma31b_eli5"] = has_gemma31b_eli5

    core_sources_16 = ["human"] + FRONTIER_PROP + LARGE_OW + SMALL_OW_CORE_5
    core_records = exp_records

    core_tier_sources = {
        "human": ["human"],
        "frontier_prop": FRONTIER_PROP,
        "large_ow": LARGE_OW,
        "small_ow": SMALL_OW_CORE_5,
    }
    out["eli5_core_4sc_tiers"] = {}
    for cond in [
        ("realistic", 0.99),
        ("realistic", 0.95),
        ("realistic", 0.90),
        ("strict", 0.99),
        ("strict_full", 0.99),
        ("default", "shipped"),
        ("default", "card_nonthinking"),
        ("default", "unscaled_T1"),
        "prefix150_0.99",
    ]:
        label = cond if isinstance(cond, str) else f"{cond[0]}_{cond[1]}"
        out["eli5_core_4sc_tiers"][label] = compute_tier_table_row(
            core_records, CORE_4_SCORERS, core_tier_sources, cond
        )

    if has_gemma31b_eli5:
        core_5bf16_scorers = CORE_4_SCORERS + ["gemma-31b"]
        core_tier_sources_17 = {
            "human": ["human"],
            "frontier_prop": FRONTIER_PROP,
            "large_ow": LARGE_OW,
            "small_ow": SMALL_OW_CORE_5 + ["gemma-31b"],
        }
        out["eli5_core_5bf16_tiers"] = {}
        for cond in [("realistic", 0.99), ("realistic", 0.95), ("realistic", 0.90), ("strict_full", 0.99), ("default", "shipped")]:
            label = f"{cond[0]}_{cond[1]}"
            out["eli5_core_5bf16_tiers"][label] = compute_tier_table_row(
                core_records, core_5bf16_scorers, core_tier_sources_17, cond
            )

    # Family-decomposed tier means for Figure 5 (Qwen bf16 = qwen-4b, qwen-9b; Gemma bf16 = gemma-e4b, gemma-12b)
    out["eli5_family_decomposed"] = {"qwen": {}, "gemma": {}}
    for fam, fam_scorers in [("qwen", ["qwen-4b", "qwen-9b"]), ("gemma", ["gemma-e4b", "gemma-12b"])]:
        for cond in [
            ("realistic", 0.90),
            ("realistic", 0.95),
            ("realistic", 0.99),
            ("strict_full", 0.99),
            ("default", "shipped"),
        ]:
            label = f"{cond[0]}_{cond[1]}"
            out["eli5_family_decomposed"][fam][label] = compute_tier_table_row(
                core_records, fam_scorers, core_tier_sources, cond
            )["tier_means"]

    # Per-source cells and human-minus-source deltas across all conditions (Tables 4, 5, 6, 10, 11)
    out["eli5_core_matrix_cells"] = {}
    core_matrix_scorers = CORE_5_SCORERS + (["gemma-31b"] if has_gemma31b_eli5 else [])
    core_matrix_sources = core_sources_16 + (["gemma-31b"] if has_gemma31b_eli5 else [])
    for sc in core_matrix_scorers:
        out["eli5_core_matrix_cells"][sc] = {}
        for src in core_matrix_sources:
            cell = core_records[(sc, src)]
            if not cell:
                continue
            out["eli5_core_matrix_cells"][sc][src] = {
                f"{m}_{p}": float(np.mean([v[(m, p)] for v in cell.values()]))
                for m in ("realistic", "strict", "strict_full")
                for p in (0.9, 0.95, 0.99)
            }
            for var in ("shipped", "card_nonthinking", "unscaled_T1"):
                out["eli5_core_matrix_cells"][sc][src][f"default_{var}"] = float(
                    np.mean([v[("default", var)] for v in cell.values()])
                )
            out["eli5_core_matrix_cells"][sc][src]["prefix150_0.99"] = float(
                np.mean([v["prefix150_0.99"] for v in cell.values()])
            )

    out["eli5_core_source_deltas_4sc"] = {}
    for cond in [
        ("realistic", 0.99),
        ("realistic", 0.95),
        ("realistic", 0.90),
        ("strict_full", 0.99),
        ("default", "shipped"),
        ("default", "card_nonthinking"),
        ("default", "unscaled_T1"),
        "prefix150_0.99",
    ]:
        label = cond if isinstance(cond, str) else f"{cond[0]}_{cond[1]}"
        out["eli5_core_source_deltas_4sc"][label] = {}
        for src in core_matrix_sources:
            if not all(core_records[(sc, src)] for sc in CORE_4_SCORERS):
                continue
            common_pids = sorted(
                set.intersection(
                    *(set(core_records[(sc, "human")].keys()) & set(core_records[(sc, src)].keys()) for sc in CORE_4_SCORERS)
                )
            )
            h_vec = [float(np.mean([core_records[(sc, "human")][pid][cond] for sc in CORE_4_SCORERS])) for pid in common_pids]
            g_vec = [float(np.mean([core_records[(sc, src)][pid][cond] for sc in CORE_4_SCORERS])) for pid in common_pids]
            boot = paired_bootstrap(h_vec, g_vec, seed=42, replicates=10000) if src != "human" else {"delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": len(common_pids)}
            src_mean_all = float(np.mean([np.mean([v[cond] for v in core_records[(sc, src)].values()]) for sc in CORE_4_SCORERS]))
            hum_mean_all = float(np.mean([np.mean([v[cond] for v in core_records[(sc, "human")].values()]) for sc in CORE_4_SCORERS]))
            out["eli5_core_source_deltas_4sc"][label][src] = {
                "source_mean_4sc": src_mean_all,
                "human_mean_4sc": hum_mean_all,
                "unweighted_diff": hum_mean_all - src_mean_all,
                **boot,
            }

    # 2. Sampled T>0 Decoding Ablation (4 bf16 scorers, 15 sources, detokenized human reference)
    sampled_dir = Path("results/audits/hc3-50-sampled-open-frontier-20261030")
    sampled_sources_15 = ["human"] + FRONTIER_PROP + LARGE_OW + SMALL_OW_BF16_4
    sampled_records = (
        deserialize_records(cached_bundles["sampled_records"])
        if "sampled_records" in cached_bundles
        else defaultdict(dict)
    )
    if sampled_dir.exists():
        merge_records(
            sampled_records,
            load_suite_records(
                sampled_dir,
                CORE_4_SCORERS,
                sampled_sources_15,
                override_human_dir=eli5_detok_dir if eli5_detok_dir.exists() else None,
            ),
        )
    sampled_tier_sources = {
        "human": ["human"],
        "frontier_prop": FRONTIER_PROP,
        "large_ow": LARGE_OW,
        "small_ow": SMALL_OW_BF16_4,
    }
    out["eli5_sampled_4sc_tiers"] = {}
    for cond in [
        ("realistic", 0.99),
        ("realistic", 0.95),
        ("realistic", 0.90),
        ("strict_full", 0.99),
        ("default", "shipped"),
    ]:
        label = f"{cond[0]}_{cond[1]}"
        out["eli5_sampled_4sc_tiers"][label] = compute_tier_table_row(
            sampled_records, CORE_4_SCORERS, sampled_tier_sources, cond
        )
    out["eli5_sampled_table12_cells"] = {
        sc: {
            src: float(np.mean([v[("realistic", 0.99)] for v in sampled_records[(sc, src)].values()]))
            for src in ["human"] + SMALL_OW_BF16_4 + FRONTIER_PROP
        }
        for sc in CORE_4_SCORERS
    }

    # 3. 4-Family Expanded ELI5 Audit (9 scorers across 20 sources, plus 10-scorer / 8-bf16 when gemma-31b present)
    exp_tier_sources = {
        "human": ["human"],
        "frontier_prop": FRONTIER_PROP,
        "large_ow": LARGE_OW,
        "small_ow": SMALL_OW_EXPANDED_9,
    }
    out["eli5_4family_9sc_tiers"] = {}
    out["eli5_4family_7bf16_tiers"] = {}
    for cond in [
        ("realistic", 0.99),
        ("realistic", 0.95),
        ("realistic", 0.90),
        ("strict_full", 0.99),
        ("default", "shipped"),
    ]:
        label = f"{cond[0]}_{cond[1]}"
        out["eli5_4family_9sc_tiers"][label] = compute_tier_table_row(
            exp_records, EXPANDED_9_SCORERS, exp_tier_sources, cond
        )
        out["eli5_4family_7bf16_tiers"][label] = compute_tier_table_row(
            exp_records, EXPANDED_7_BF16_SCORERS, exp_tier_sources, cond
        )

    if has_gemma31b_eli5:
        exp_10_scorers = EXPANDED_9_SCORERS + ["gemma-31b"]
        exp_8bf16_scorers = EXPANDED_7_BF16_SCORERS + ["gemma-31b"]
        exp_tier_sources_21 = {
            "human": ["human"],
            "frontier_prop": FRONTIER_PROP,
            "large_ow": LARGE_OW,
            "small_ow": SMALL_OW_EXPANDED_9 + ["gemma-31b"],
        }
        out["eli5_4family_10sc_tiers"] = {}
        out["eli5_4family_8bf16_tiers"] = {}
        for cond in [("realistic", 0.99), ("realistic", 0.95), ("realistic", 0.90), ("strict_full", 0.99), ("default", "shipped")]:
            label = f"{cond[0]}_{cond[1]}"
            out["eli5_4family_10sc_tiers"][label] = compute_tier_table_row(
                exp_records, exp_10_scorers, exp_tier_sources_21, cond
            )
            out["eli5_4family_8bf16_tiers"][label] = compute_tier_table_row(
                exp_records, exp_8bf16_scorers, exp_tier_sources_21, cond
            )

    table8_scorers = EXPANDED_9_SCORERS + (["gemma-31b"] if has_gemma31b_eli5 else [])
    table8_sources = exp_sources_20 + (["gemma-31b"] if has_gemma31b_eli5 else [])
    out["eli5_4family_table8_cells"] = {
        sc: {
            src: {
                "realistic_0.99": float(np.mean([v[("realistic", 0.99)] for v in exp_records[(sc, src)].values()])),
                "realistic_0.95": float(np.mean([v[("realistic", 0.95)] for v in exp_records[(sc, src)].values()])),
                "realistic_0.9": float(np.mean([v[("realistic", 0.9)] for v in exp_records[(sc, src)].values()])),
                "strict_full_0.99": float(np.mean([v[("strict_full", 0.99)] for v in exp_records[(sc, src)].values()])),
                "default_shipped": float(np.mean([v[("default", "shipped")] for v in exp_records[(sc, src)].values()])),
            }
            for src in table8_sources
            if exp_records[(sc, src)]
        }
        for sc in table8_scorers
    }

    # Same-family vs cross-family advantage & orthographic sensitivity check (Figure 7)
    out["same_family_advantage"] = {"eli5_qwen_gemma": {}, "eli5_all_4_families": {}}
    qwen_gemma_6sc = ["qwen-4b", "qwen-9b", "qwen-27b-int8", "qwen-32b-int8", "gemma-e4b", "gemma-12b"]
    for p in (0.99, 0.95, 0.90):
        out["same_family_advantage"]["eli5_qwen_gemma"][f"realistic_{p}"] = compute_same_vs_cross_family(
            exp_records, qwen_gemma_6sc, exp_sources_20, ("realistic", p)
        )
        out["same_family_advantage"]["eli5_qwen_gemma"][f"ortho_clean_{p}"] = compute_same_vs_cross_family(
            exp_records, qwen_gemma_6sc, exp_sources_20, ("ortho_clean", p)
        )
        out["same_family_advantage"]["eli5_all_4_families"][f"realistic_{p}"] = compute_same_vs_cross_family(
            exp_records, table8_scorers, table8_sources, ("realistic", p)
        )
        out["same_family_advantage"]["eli5_all_4_families"][f"ortho_clean_{p}"] = compute_same_vs_cross_family(
            exp_records, table8_scorers, table8_sources, ("ortho_clean", p)
        )

    # 4. Wikipedia CS/AI (4 bf16 scorers across 15 sources, plus expanded 4-family suite when present)
    wiki_dir = Path("results/audits/hc3-wiki-50-comparison-frontier-20261030")
    wiki_sources_15 = ["human"] + FRONTIER_PROP + LARGE_OW + SMALL_OW_BF16_4
    wiki_records = (
        deserialize_records(cached_bundles["wiki_records"])
        if "wiki_records" in cached_bundles
        else defaultdict(dict)
    )
    if wiki_dir.exists():
        merge_records(wiki_records, load_suite_records(wiki_dir, CORE_4_SCORERS, wiki_sources_15))
    wiki_tier_sources = {
        "human": ["human"],
        "frontier_prop": FRONTIER_PROP,
        "large_ow": LARGE_OW,
        "small_ow": SMALL_OW_BF16_4,
    }
    out["wiki_4sc_tiers"] = {}
    for cond in [
        ("realistic", 0.99),
        ("realistic", 0.95),
        ("realistic", 0.90),
        ("strict_full", 0.99),
        ("default", "shipped"),
    ]:
        label = f"{cond[0]}_{cond[1]}"
        out["wiki_4sc_tiers"][label] = compute_tier_table_row(
            wiki_records, CORE_4_SCORERS, wiki_tier_sources, cond
        )
    out["wiki_table9_cells"] = {
        sc: {
            src: float(np.mean([v[("realistic", 0.99)] for v in wiki_records[(sc, src)].values()]))
            for src in wiki_sources_15
        }
        for sc in CORE_4_SCORERS
    }
    out["same_family_advantage"]["wiki_qwen_gemma"] = {}
    for p in (0.99, 0.95, 0.90):
        out["same_family_advantage"]["wiki_qwen_gemma"][f"realistic_{p}"] = compute_same_vs_cross_family(
            wiki_records, CORE_4_SCORERS, wiki_sources_15, ("realistic", p)
        )
        out["same_family_advantage"]["wiki_qwen_gemma"][f"ortho_clean_{p}"] = compute_same_vs_cross_family(
            wiki_records, CORE_4_SCORERS, wiki_sources_15, ("ortho_clean", p)
        )

    wiki_exp_dir = Path("results/audits/hc3-wiki-50-expanded-frontier-20261102")
    wiki_gemma31b_dir = Path("results/audits/hc3-wiki-50-expanded-gemma31b-20261102")
    candidate_wiki_scorers = [
        "qwen-4b",
        "qwen-9b",
        "qwen-27b-int8",
        "gemma-e4b",
        "gemma-12b",
        "gemma-31b",
        "llama-3.2-1b",
        "llama-3.1-8b",
        "mistral-7b",
    ]
    all_wiki_sources = ["human"] + FRONTIER_PROP + LARGE_OW + candidate_wiki_scorers
    wiki_exp_records = (
        deserialize_records(cached_bundles["wiki_exp_records"])
        if "wiki_exp_records" in cached_bundles
        else defaultdict(dict)
    )
    if wiki_exp_dir.exists():
        merge_records(wiki_exp_records, load_suite_records(wiki_exp_dir, candidate_wiki_scorers, all_wiki_sources))
    if wiki_gemma31b_dir.exists():
        merge_records(wiki_exp_records, load_suite_records(wiki_gemma31b_dir, candidate_wiki_scorers, all_wiki_sources))

    wiki_exp_scorers = [sc for sc in candidate_wiki_scorers if wiki_exp_records.get((sc, "human"))]
    if "mistral-7b" in wiki_exp_scorers:
        wiki_exp_sources = ["human"] + FRONTIER_PROP + LARGE_OW + wiki_exp_scorers
        wiki_exp_tier_sources = {
            "human": ["human"],
            "frontier_prop": FRONTIER_PROP,
            "large_ow": LARGE_OW,
            "small_ow": wiki_exp_scorers,
        }
        out["wiki_expanded_scorers"] = wiki_exp_scorers
        out["wiki_expanded_8sc_tiers"] = {}
        out["wiki_expanded_7bf16_tiers"] = {}
        wiki_bf16_scorers = [sc for sc in wiki_exp_scorers if not sc.endswith("-int8")]
        for cond in [
            ("realistic", 0.99),
            ("realistic", 0.95),
            ("realistic", 0.90),
            ("strict_full", 0.99),
            ("default", "shipped"),
        ]:
            label = f"{cond[0]}_{cond[1]}"
            out["wiki_expanded_8sc_tiers"][label] = compute_tier_table_row(
                wiki_exp_records, wiki_exp_scorers, wiki_exp_tier_sources, cond
            )
            out["wiki_expanded_7bf16_tiers"][label] = compute_tier_table_row(
                wiki_exp_records, wiki_bf16_scorers, wiki_exp_tier_sources, cond
            )
        out["wiki_expanded_cells"] = {
            sc: {
                src: float(np.mean([v[("realistic", 0.99)] for v in wiki_exp_records[(sc, src)].values()]))
                for src in wiki_exp_sources
                if wiki_exp_records[(sc, src)]
            }
            for sc in wiki_exp_scorers
        }
        out["same_family_advantage"]["wiki_all_4_families"] = {}
        for p in (0.99, 0.95, 0.90):
            out["same_family_advantage"]["wiki_all_4_families"][f"realistic_{p}"] = compute_same_vs_cross_family(
                wiki_exp_records, wiki_exp_scorers, wiki_exp_sources, ("realistic", p)
            )
            out["same_family_advantage"]["wiki_all_4_families"][f"ortho_clean_{p}"] = compute_same_vs_cross_family(
                wiki_exp_records, wiki_exp_scorers, wiki_exp_sources, ("ortho_clean", p)
            )

    # 5. Top-100 vs Top-1000 vs Full-Vocabulary Validation (Table 7 & Figure 6)
    top1000_dir = Path("results/audits/hc3-50-top1000-validation")
    top1000_human_detok = Path("results/audits/hc3-50-top1000-detok-human-20261101")
    top1000_exp_dir = Path("results/audits/hc3-50-top1000-expanded-detok-20261102")
    top1000_gemma31b_dir = Path("results/audits/hc3-50-top1000-gemma31b-detok-20261102")
    top1000_records = (
        deserialize_records(cached_bundles["top1000_records"])
        if "top1000_records" in cached_bundles
        else defaultdict(dict)
    )
    if top1000_dir.exists():
        merge_records(
            top1000_records,
            load_suite_records(
                top1000_dir,
                ["qwen-9b", "gemma-12b"],
                core_sources_16,
                override_human_dir=top1000_human_detok if top1000_human_detok.exists() else None,
            ),
        )
    for extra_dir in (top1000_exp_dir, top1000_gemma31b_dir):
        if extra_dir.exists():
            merge_records(
                top1000_records,
                load_suite_records(
                    extra_dir,
                    ["qwen-9b", "gemma-12b", "gemma-31b", "llama-3.1-8b", "mistral-7b"],
                    all_eli5_sources,
                ),
            )

    top1000_scorers = [
        sc
        for sc in ("qwen-9b", "gemma-12b", "gemma-31b", "llama-3.1-8b", "mistral-7b")
        if top1000_records.get((sc, "human"))
    ]

    out["top1000_validation"] = {}
    for sc in top1000_scorers:
        out["top1000_validation"][sc] = {}
        groups_to_report = {
            "human": ["human"],
            "gpt-6-astra": ["gpt-6-astra"],
            "claude-opus-5.5": ["claude-opus-5.5"],
            "gemini-4-argon": ["gemini-4-argon"],
            "frontier_prop_mean": FRONTIER_PROP,
            "large_ow_mean": LARGE_OW,
            "small_ow_5_mean": SMALL_OW_CORE_5,
            "small_ow_4bf16_mean": SMALL_OW_BF16_4,
        }
        if sc in ("llama-3.1-8b", "mistral-7b", "gemma-31b"):
            groups_to_report["small_ow_9_mean"] = [
                s for s in SMALL_OW_EXPANDED_9 if top1000_records.get((sc, s)) and exp_records.get((sc, s))
            ]
        for label, srcs in groups_to_report.items():
            avail_srcs = [s for s in srcs if top1000_records.get((sc, s)) and exp_records.get((sc, s))]
            if not avail_srcs:
                continue
            m100 = float(np.mean([np.mean([v["tail_mass"] for v in exp_records[(sc, s)].values()]) for s in avail_srcs]))
            m1000 = float(np.mean([np.mean([v["tail_mass"] for v in top1000_records[(sc, s)].values()]) for s in avail_srcs]))
            miss100 = float(np.mean([np.mean([v["top_miss"] for v in exp_records[(sc, s)].values()]) for s in avail_srcs]))
            miss1000 = float(np.mean([np.mean([v["top_miss"] for v in top1000_records[(sc, s)].values()]) for s in avail_srcs]))
            wcs100 = float(np.mean([np.mean([v[("realistic", 0.99)] for v in exp_records[(sc, s)].values()]) for s in avail_srcs]))
            wcs1000 = float(np.mean([np.mean([v[("realistic", 0.99)] for v in top1000_records[(sc, s)].values()]) for s in avail_srcs]))
            wcsfull = float(np.mean([np.mean([v[("strict_full", 0.99)] for v in exp_records[(sc, s)].values()]) for s in avail_srcs]))
            out["top1000_validation"][sc][label] = {
                "tail_mass_100": m100,
                "tail_mass_1000": m1000,
                "miss_100": miss100,
                "miss_1000": miss1000,
                "wcs_100": wcs100,
                "wcs_1000": wcs1000,
                "wcs_full": wcsfull,
                "gap_closed_pct": (wcs1000 - wcs100) / (wcsfull - wcs100) * 100.0 if wcsfull != wcs100 else 100.0,
            }

    # Save updated prompt cache so downstream/other servers always have full per-prompt records
    updated_bundles = {
        "exp_records": serialize_records(exp_records),
        "sampled_records": serialize_records(sampled_records),
        "wiki_records": serialize_records(wiki_records),
        "wiki_exp_records": serialize_records(wiki_exp_records),
        "top1000_records": serialize_records(top1000_records),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_path, "wt", encoding="utf-8", compresslevel=9) as f:
        json.dump(updated_bundles, f)

    # 6. Appendix A: Zipf-Adjusted Lexical Category and Loss-Rate Taxonomy
    out_path = Path("results/reports/revised_nov2_recomputed.json")
    existing_out = json.loads(out_path.read_text()) if out_path.exists() else {}
    if exp_dir.exists() and eli5_detok_dir.exists():
        out["zipf_9sc_20src"] = analyze_zipf_subset(exp_dir, EXPANDED_9_SCORERS, exp_sources_20, p=0.99)
        out["zipf_4sc_16src"] = analyze_zipf_subset(eli5_detok_dir, CORE_4_SCORERS, core_sources_16, p=0.99)
        out["zipf_5sc_16src"] = analyze_zipf_subset(eli5_detok_dir, CORE_5_SCORERS, core_sources_16, p=0.99)
    elif "zipf_9sc_20src" in existing_out and "zipf_4sc_16src" in existing_out:
        out["zipf_9sc_20src"] = existing_out["zipf_9sc_20src"]
        out["zipf_4sc_16src"] = existing_out["zipf_4sc_16src"]
        out["zipf_5sc_16src"] = existing_out["zipf_5sc_16src"]

    out_path.write_text(json.dumps(out, indent=2))
    print(f"Wrote complete recomputed data to {out_path}")


if __name__ == "__main__":
    main()
