#!/usr/bin/env python3
"""
Track which words get lost under the Word Coverage Score (WCS) definition,
including Zipf-adjusted (frequency-normalized) loss rates and lexical category
stratification (Content/Lexical words vs. Function/Stopwords vs. Punctuation).

In WCS:
A word is segmented by the word policy (whitespace by default).
A word is LOST if any of its constituent subtokens fails to fall within
the top-p nucleus of the scorer (i.e. cumulative mass before > p or target not in top-100).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from wcsmatrix.metrics import word_groups

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "in", "on", "at", "to", "for", "of", "with", "by",
    "from", "up", "about", "into", "over", "after", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall", "should", "can", "could",
    "may", "might", "must", "it", "its", "they", "them", "their", "we", "us", "our", "you", "your",
    "i", "me", "my", "he", "him", "his", "she", "her", "that", "this", "these", "those", "which",
    "who", "whom", "what", "where", "when", "why", "how", "as", "so", "than", "then", "there", "here",
    "not", "no", "nor", "only", "own", "same", "too", "very", "just", "also",
}

TIER_MAP = {
    "human": "human",
    "gpt-6-astra": "frontier_proprietary",
    "claude-opus-5.5": "frontier_proprietary",
    "gemini-4-argon": "frontier_proprietary",
    "deepseek-v4.1-flash-1017": "large_open_weight",
    "glm-5.3": "large_open_weight",
    "minimax-m3": "large_open_weight",
    "kimi-k3": "large_open_weight",
    "mimo-v2.6-pro": "large_open_weight",
    "deepseek-v4-pro-0813": "large_open_weight",
    "qwen3.8-2.4t-a95b": "large_open_weight",
}


def classify_word(raw_word: str) -> tuple[str, str]:
    clean = re.sub(r"^[^\w]+|[^\w]+$", "", raw_word.lower())
    if not clean:
        return "standalone_punct_or_symbol", raw_word.lower()
    if clean in STOPWORDS:
        return "function_stopword", clean
    return "content_lexical", clean


def analyze_lost_words(audit_dir: Path, p: float = 0.99, mode: str = "realistic", policy: str = "whitespace"):
    lost_records = []
    word_counters_by_pair = defaultdict(Counter)
    word_counters_by_source = defaultdict(Counter)
    total_words_by_source = Counter()

    # Zipf / rate-normalized tracking per tier and per source (excluding self-scoring diagonal)
    tier_cat_total = defaultdict(Counter)
    tier_cat_lost = defaultdict(Counter)
    tier_word_total = defaultdict(Counter)
    tier_word_lost = defaultdict(Counter)
    tier_word_cat = {}

    npz_paths = sorted(audit_dir.glob("*/*/*.npz"))
    print(f"Analyzing {len(npz_paths)} audits in {audit_dir}...")

    for path in npz_paths:
        meta_path = path.with_suffix(".json")
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text())
        scorer = meta["scorer"]
        source = meta["source"]
        prompt_id = meta["prompt_id"]
        text = meta["text"]
        tier = TIER_MAP.get(source, "small_open_weight")
        is_off_diag = scorer != source

        with np.load(path, allow_pickle=False) as arrays:
            top_probs = arrays["top_probs"]
            target_top_index = arrays["target_top_index"]
            offsets = arrays["offsets"].tolist()
            mass_before = arrays.get("mass_before")

        probs = np.asarray(top_probs, dtype=np.float64)
        indices = np.asarray(target_top_index, dtype=int)

        if mode == "realistic":
            probs = probs / probs.sum(axis=-1, keepdims=True)
            before = probs.cumsum(axis=-1) - probs
            valid = (indices >= 0) & (indices < probs.shape[1])
            survives = valid & (before[np.arange(len(indices)), indices.clip(0, probs.shape[1] - 1)] <= p)
        elif mode == "strict":
            before = probs.cumsum(axis=-1) - probs
            valid = (indices >= 0) & (indices < probs.shape[1])
            survives = valid & (before[np.arange(len(indices)), indices.clip(0, probs.shape[1] - 1)] <= p)
        elif mode == "strict_full":
            survives = np.asarray(mass_before) <= p
        else:
            raise ValueError(mode)

        pattern = r"\S+" if policy == "whitespace" else r"\b\w+(?:['’-]\w+)*\b"
        matches = list(re.finditer(pattern, text, flags=re.UNICODE))
        groups = word_groups(text, offsets, policy)

        if len(matches) != len(groups):
            continue

        for match, g in zip(matches, groups):
            word_str = match.group()
            total_words_by_source[source] += 1
            word_survives = bool(np.all(survives[g]))

            cat, norm_w = classify_word(word_str)
            if is_off_diag:
                tier_cat_total[tier][cat] += 1
                tier_word_total[tier][norm_w] += 1
                tier_word_cat[norm_w] = cat
                if not word_survives:
                    tier_cat_lost[tier][cat] += 1
                    tier_word_lost[tier][norm_w] += 1

            if not word_survives:
                failing_tokens = [tok_idx for tok_idx in g if not survives[tok_idx]]
                first_fail = failing_tokens[0]
                fail_reason = "outside_top100" if indices[first_fail] < 0 else "outside_nucleus_p"

                lost_records.append(
                    {
                        "scorer": scorer,
                        "source": source,
                        "prompt_id": prompt_id,
                        "word": word_str,
                        "lexical_category": cat,
                        "char_start": match.start(),
                        "char_end": match.end(),
                        "failing_token_idx": first_fail,
                        "reason": fail_reason,
                    }
                )
                word_counters_by_pair[(source, scorer)][word_str.lower()] += 1
                word_counters_by_source[source][word_str.lower()] += 1

    cat_summary_rows = []
    for tier in ("human", "frontier_proprietary", "large_open_weight", "small_open_weight"):
        tot_tier_lost = sum(tier_cat_lost[tier].values())
        for cat in ("content_lexical", "function_stopword", "standalone_punct_or_symbol"):
            tot_c = tier_cat_total[tier][cat]
            lost_c = tier_cat_lost[tier][cat]
            if tot_c > 0:
                cat_summary_rows.append(
                    {
                        "tier": tier,
                        "lexical_category": cat,
                        "lost_words": lost_c,
                        "total_words": tot_c,
                        "loss_rate": round(lost_c / tot_c, 6),
                        "share_of_tier_losses": round(lost_c / tot_tier_lost, 6) if tot_tier_lost else 0.0,
                    }
                )

    zipf_rows = []
    for tier in ("human", "frontier_proprietary", "large_open_weight", "small_open_weight"):
        baseline_rate = sum(tier_cat_lost[tier].values()) / max(1, sum(tier_cat_total[tier].values()))
        for w, tot_w in tier_word_total[tier].items():
            if tot_w < 33:
                continue
            lost_w = tier_word_lost[tier][w]
            rate_w = lost_w / tot_w
            zipf_rows.append(
                {
                    "tier": tier,
                    "word": w,
                    "lexical_category": tier_word_cat.get(w, "content_lexical"),
                    "lost_count": lost_w,
                    "total_occurrences": tot_w,
                    "loss_rate": round(rate_w, 6),
                    "relative_risk_vs_tier_mean": round(rate_w / baseline_rate, 4) if baseline_rate else 0.0,
                }
            )
    zipf_rows.sort(key=lambda r: (r["tier"], -r["loss_rate"], -r["lost_count"]))

    return (
        lost_records,
        word_counters_by_pair,
        word_counters_by_source,
        total_words_by_source,
        cat_summary_rows,
        zipf_rows,
    )


def main():
    parser = argparse.ArgumentParser(description="Track lost words and Zipf-adjusted loss rates in WCS audits")
    parser.add_argument(
        "--suite",
        default="results/audits/hc3-50-expanded-open-frontier-detok-20261101",
        help="Path to audit suite directory",
    )
    parser.add_argument("--p", type=float, default=0.99)
    parser.add_argument("--mode", default="realistic")
    parser.add_argument(
        "--output",
        default="results/reports/hc3-50-expanded-open-frontier-detok-20261101/wcs_lost_words.csv",
    )
    args = parser.parse_args()

    audit_dir = Path(args.suite)
    records, by_pair, by_source, totals, cat_rows, zipf_rows = analyze_lost_words(
        audit_dir, p=args.p, mode=args.mode
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scorer",
                "source",
                "prompt_id",
                "word",
                "lexical_category",
                "char_start",
                "char_end",
                "failing_token_idx",
                "reason",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    cat_path = out_path.with_name("wcs_lexical_category_summary.csv")
    with open(cat_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(cat_rows[0]))
        writer.writeheader()
        writer.writerows(cat_rows)

    zipf_path = out_path.with_name("wcs_lost_words_zipf_adjusted.csv")
    with open(zipf_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(zipf_rows[0]))
        writer.writeheader()
        writer.writerows(zipf_rows)

    print(f"\nWrote {len(records)} lost word occurrences to {out_path}")
    print(f"Wrote lexical category summary to {cat_path}")
    print(f"Wrote Zipf-adjusted rate table to {zipf_path}")


if __name__ == "__main__":
    main()
