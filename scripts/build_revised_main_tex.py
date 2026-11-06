#!/usr/bin/env python3
"""
Generate overleaf/main.tex matching WCS_revised_2_Nov.pdf in official ACL format,
populating all recomputed values, 95% prompt-paired bootstrap CIs, per-source deltas,
filter sensitivity rows, K=1000 validation results, orthographic sensitivity checks,
Wikipedia CS/AI 4-family extension results, and Appendix A Zipf-adjusted tables
directly from results/reports/revised_nov2_recomputed.json.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import zipfile
from pathlib import Path


def fmt_ci(low: float, high: float, signed: bool = True, decimals: int = 4) -> str:
    if signed:
        return f"[{low:+.{decimals}f},\\,{high:+.{decimals}f}]"
    return f"[{low:.{decimals}f},\\,{high:.{decimals}f}]"


def fmt_delta_ci(entry: dict, decimals: int = 4) -> str:
    d = entry["delta"]
    lo = entry["ci_low"]
    hi = entry["ci_high"]
    return f"${d:+.{decimals}f}$ (${fmt_ci(lo, hi, True, decimals)}$)"


def fmt_cell(val: float, is_diag: bool = False) -> str:
    s = f"{val:.2f}"
    return f"\\textbf{{{s}}}" if is_diag else s


def build_matrix_rows_4sc(
    cells: dict,
    deltas: dict,
    scorers: list[str] = ("qwen-4b", "qwen-9b", "gemma-e4b", "gemma-12b"),
    cond_key: str = "realistic_0.99",
    include_27b: bool = True,
) -> str:
    tiers = [
        ("Human Reference", [("Human (HC3 ELI5, detok.)", "human")]),
        (
            "Frontier Proprietary (API, $T=1.0, p=0.95$)",
            [
                ("GPT-6 Astra", "gpt-6-astra"),
                ("Claude Opus 5.5", "claude-opus-5.5"),
                ("Gemini 4 Argon", "gemini-4-argon"),
            ],
        ),
        (
            "Large Open-Weight (API, $T=1.0, p=0.95$)",
            [
                ("DeepSeek-V4.1-Flash", "deepseek-v4.1-flash-1017"),
                ("GLM-5.3", "glm-5.3"),
                ("MiniMax-M3", "minimax-m3"),
                ("Kimi-K3", "kimi-k3"),
                ("MiMo-V2.6-Pro", "mimo-v2.6-pro"),
                ("DeepSeek-V4-Pro", "deepseek-v4-pro-0813"),
                ("Qwen3.8-2.4T-A95B", "qwen3.8-2.4t-a95b"),
            ],
        ),
        (
            "Small Open-Weight (Local Greedy)",
            [
                ("Qwen-4B", "qwen-4b"),
                ("Qwen-9B", "qwen-9b"),
                ("Qwen-27B (int8)", "qwen-27b-int8"),
                ("Gemma-12B", "gemma-12b"),
                ("Gemma-E4B", "gemma-e4b"),
            ],
        ),
    ]
    ncols = 7 if include_27b else 6
    lines = []
    for tier_title, items in tiers:
        lines.append(f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{{tier_title}}}}} \\\\")
        for label, src in items:
            row_cells = [fmt_cell(cells[sc][src][cond_key], sc == src) for sc in scorers]
            if include_27b:
                row_cells.append(fmt_cell(cells["qwen-27b-int8"][src][cond_key], src == "qwen-27b-int8"))
            d_info = deltas[src]
            mean_4sc = f"{d_info['source_mean_4sc']:.4f}"
            if src == "human":
                delta_str = "---"
            else:
                delta_str = fmt_delta_ci(d_info, 4)
            lines.append(f"{label} & " + " & ".join(row_cells) + f" & {mean_4sc} & {delta_str} \\\\")
        lines.append("\\midrule")
    if lines and lines[-1] == "\\midrule":
        lines.pop()
    return "\n".join(lines)


def build_table3_rows(data: dict) -> str:
    def row_str(suite_label: str, tier_dict: dict) -> str:
        tm = tier_dict["tier_means"]
        c = tier_dict["contrasts"]
        c_ls = c["large_ow_minus_small_ow"]
        c_lf = c["large_ow_minus_frontier_prop"]
        c_hf = c["human_minus_frontier_prop"]
        return (
            f"{suite_label} & {tm['human']:.4f} & {tm['frontier_prop']:.4f} & "
            f"{tm['large_ow']:.4f} & {tm['small_ow']:.4f} & "
            f"${c_ls['delta']:+.4f}$ (${fmt_ci(c_ls['ci_low'], c_ls['ci_high'])}$) & "
            f"${c_lf['delta']:+.4f}$ (${fmt_ci(c_lf['ci_low'], c_lf['ci_high'])}$) & "
            f"${c_hf['delta']:+.4f}$ (${fmt_ci(c_hf['ci_low'], c_hf['ci_high'])}$) \\\\"
        )

    c4 = data["eli5_core_4sc_tiers"]
    samp = data["eli5_sampled_4sc_tiers"]
    f9 = data["eli5_4family_9sc_tiers"]
    f7 = data["eli5_4family_7bf16_tiers"]
    w4 = data["wiki_4sc_tiers"]

    rows = [
        "\\multicolumn{8}{l}{\\textit{Primary Benchmark: HC3 ELI5 (4 Core bf16 Scorers, 16 Sources)}} \\\\",
        row_str("Realistic ($p=0.99$, primary)", c4["realistic_0.99"]),
        row_str("Realistic ($p=0.95$)", c4["realistic_0.95"]),
        row_str("Realistic ($p=0.90$)", c4["realistic_0.9"]),
        row_str("Strict unnormalized ($p=0.99$)", c4["strict_0.99"]),
        row_str("Full-vocabulary \\texttt{strict\\_full} ($p=0.99$)", c4["strict_full_0.99"]),
        row_str("Shipped generation defaults", c4["default_shipped"]),
        row_str("Model-card non-thinking defaults", c4["default_card_nonthinking"]),
        row_str("Unscaled defaults ($T=1.0$)", c4["default_unscaled_T1"]),
        row_str("150-word prefix truncation ($p=0.99$)", c4["prefix150_0.99"]),
        "\\midrule",
        "\\multicolumn{8}{l}{\\textit{Cross-Family, Cross-Domain, and Decoding Regime Extensions}} \\\\",
        row_str("HC3 ELI5 Sampled Local ($T=1.0, p=0.95$, 4 bf16 scorers)", samp["realistic_0.99"]),
        row_str("HC3 ELI5 4-Family Suite (9 scorers, 20 sources, $p=0.99$)", f9["realistic_0.99"]),
        row_str("HC3 ELI5 4-Family Suite (7 bf16 scorers, 20 sources, $p=0.99$)", f7["realistic_0.99"]),
        row_str("HC3 ELI5 4-Family Suite (9 scorers, $p=0.95$)", f9["realistic_0.95"]),
        row_str("HC3 ELI5 4-Family Suite (9 scorers, \\texttt{strict\\_full} $p=0.99$)", f9["strict_full_0.99"]),
        row_str("HC3 Wikipedia CS/AI (4 bf16 scorers, 15 sources, $p=0.99$)", w4["realistic_0.99"]),
        row_str("HC3 Wikipedia CS/AI (4 bf16 scorers, $p=0.95$)", w4["realistic_0.95"]),
        row_str("HC3 Wikipedia CS/AI (4 bf16 scorers, \\texttt{strict\\_full} $p=0.99$)", w4["strict_full_0.99"]),
    ]
    if "eli5_4family_10sc_tiers" in data:
        f10 = data["eli5_4family_10sc_tiers"]
        f8 = data["eli5_4family_8bf16_tiers"]
        rows.append(row_str("HC3 ELI5 4-Family + Gemma-31B (10 scorers, 21 sources, $p=0.99$)", f10["realistic_0.99"]))
        rows.append(row_str("HC3 ELI5 4-Family + Gemma-31B (8 bf16 scorers, 21 sources, $p=0.99$)", f8["realistic_0.99"]))
    if "wiki_expanded_8sc_tiers" in data:
        w8 = data["wiki_expanded_8sc_tiers"]
        w7 = data["wiki_expanded_7bf16_tiers"]
        exp_scs = data.get("wiki_expanded_scorers", [])
        n_sc = len(exp_scs)
        n_bf16 = sum(1 for sc in exp_scs if not sc.endswith("-int8"))
        if n_sc > n_bf16:
            rows.append(row_str(f"HC3 Wikipedia CS/AI 4-Family ({n_sc} scorers, {11 + n_sc} sources, $p=0.99$)", w8["realistic_0.99"]))
        rows.append(row_str(f"HC3 Wikipedia CS/AI 4-Family ({n_bf16} bf16 scorers, {11 + n_sc} sources, $p=0.99$)", w7["realistic_0.99"]))
    return "\n".join(rows)


def build_table7_rows(data: dict) -> str:
    val = data["top1000_validation"]
    scorer_titles = [
        ("qwen-9b", "Qwen3.5-9B Scorer ($|V| = 248{{,}}320$)"),
        ("gemma-12b", "Gemma-4-12B-it Scorer ($|V| = 262{{,}}144$)"),
        ("gemma-31b", "Gemma-4-31B-it Scorer ($|V| = 262{{,}}144$)"),
        ("llama-3.1-8b", "Llama-3.1-8B-Instruct Scorer ($|V| = 128{{,}}256$)"),
        ("mistral-7b", "Mistral-7B-Instruct-v0.3 Scorer ($|V| = 32{{,}}768$)"),
    ]
    source_rows = [
        ("Human (HC3 ELI5, detok.)", "human"),
        ("GPT-6 Astra", "gpt-6-astra"),
        ("Claude Opus 5.5", "claude-opus-5.5"),
        ("Gemini 4 Argon", "gemini-4-argon"),
        ("Frontier Proprietary (mean, 3 sources)", "frontier_prop_mean"),
        ("Large Open-Weight (mean, 7 sources)", "large_ow_mean"),
        ("Small Open-Weight (mean, 5 core sources)", "small_ow_5_mean"),
    ]
    lines = []
    for sc_key, sc_title in scorer_titles:
        if sc_key not in val:
            continue
        lines.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{sc_title}}}}} \\\\")
        for label, skey in source_rows:
            r = val[sc_key][skey]
            lines.append(
                f"{label} & {r['tail_mass_100']*100:.2f}\\% & {r['tail_mass_1000']*100:.3f}\\% & "
                f"{r['miss_100']*100:.2f}\\% & {r['miss_1000']*100:.2f}\\% & "
                f"{r['wcs_100']:.4f} & {r['wcs_1000']:.4f} & {r['wcs_full']:.4f} ({r['gap_closed_pct']:.1f}\\%) \\\\"
            )
        lines.append("\\midrule")
    if lines and lines[-1] == "\\midrule":
        lines.pop()
    return "\n".join(lines)


def build_table8_rows(data: dict) -> str:
    cells = data["eli5_4family_table8_cells"]
    scorers_9 = [
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
    scorers_7bf16 = [
        "qwen-4b",
        "qwen-9b",
        "gemma-e4b",
        "gemma-12b",
        "llama-3.2-1b",
        "llama-3.1-8b",
        "mistral-7b",
    ]
    tiers = [
        ("Human Reference", [("Human (HC3 ELI5, detok.)", "human")]),
        (
            "Frontier Proprietary",
            [
                ("GPT-6 Astra", "gpt-6-astra"),
                ("Claude Opus 5.5", "claude-opus-5.5"),
                ("Gemini 4 Argon", "gemini-4-argon"),
            ],
        ),
        (
            "Large Open-Weight",
            [
                ("DeepSeek-V4.1-Flash", "deepseek-v4.1-flash-1017"),
                ("GLM-5.3", "glm-5.3"),
                ("MiniMax-M3", "minimax-m3"),
                ("Kimi-K3", "kimi-k3"),
                ("MiMo-V2.6-Pro", "mimo-v2.6-pro"),
                ("DeepSeek-V4-Pro", "deepseek-v4-pro-0813"),
                ("Qwen3.8-2.4T-A95B", "qwen3.8-2.4t-a95b"),
            ],
        ),
        (
            "Small Open-Weight (4 Families)",
            [
                ("Qwen-4B", "qwen-4b"),
                ("Qwen-9B", "qwen-9b"),
                ("Qwen-27B (int8)", "qwen-27b-int8"),
                ("Qwen2.5-32B (int8)", "qwen-32b-int8"),
                ("Gemma-E4B", "gemma-e4b"),
                ("Gemma-12B", "gemma-12b"),
                ("Llama-3.2-1B", "llama-3.2-1b"),
                ("Llama-3.1-8B", "llama-3.1-8b"),
                ("Mistral-7B-v0.3", "mistral-7b"),
            ],
        ),
    ]
    lines = []
    for tier_title, items in tiers:
        lines.append(f"\\multicolumn{{12}}{{l}}{{\\textit{{{tier_title}}}}} \\\\")
        for label, src in items:
            vals_9 = [cells[sc][src]["realistic_0.99"] for sc in scorers_9]
            vals_7 = [cells[sc][src]["realistic_0.99"] for sc in scorers_7bf16]
            row_strs = [fmt_cell(cells[sc][src]["realistic_0.99"], sc == src) for sc in scorers_9]
            m9 = sum(vals_9) / len(vals_9)
            m7 = sum(vals_7) / len(vals_7)
            lines.append(f"{label} & " + " & ".join(row_strs) + f" & {m7:.4f} & {m9:.4f} \\\\")
        lines.append("\\midrule")
    if lines and lines[-1] == "\\midrule":
        lines.pop()
    return "\n".join(lines)


def build_table9_rows(data: dict) -> tuple[str, bool, bool]:
    if "wiki_expanded_cells" in data:
        cells = data["wiki_expanded_cells"]
        has_27b = "qwen-27b-int8" in cells
        scorers = [
            "qwen-4b",
            "qwen-9b",
            *(["qwen-27b-int8"] if has_27b else []),
            "gemma-e4b",
            "gemma-12b",
            "llama-3.2-1b",
            "llama-3.1-8b",
            "mistral-7b",
        ]
        scorers_4 = ["qwen-4b", "qwen-9b", "gemma-e4b", "gemma-12b"]
        small_ow_items = [
            ("Qwen-4B", "qwen-4b"),
            ("Qwen-9B", "qwen-9b"),
            *( [("Qwen-27B (int8)", "qwen-27b-int8")] if has_27b else []),
            ("Gemma-E4B", "gemma-e4b"),
            ("Gemma-12B", "gemma-12b"),
            ("Llama-3.2-1B", "llama-3.2-1b"),
            ("Llama-3.1-8B", "llama-3.1-8b"),
            ("Mistral-7B-v0.3", "mistral-7b"),
        ]
        tiers = [
            ("Human Reference", [("Human (Wikipedia CS/AI)", "human")]),
            (
                "Frontier Proprietary",
                [
                    ("GPT-6 Astra", "gpt-6-astra"),
                    ("Claude Opus 5.5", "claude-opus-5.5"),
                    ("Gemini 4 Argon", "gemini-4-argon"),
                ],
            ),
            (
                "Large Open-Weight",
                [
                    ("DeepSeek-V4.1-Flash", "deepseek-v4.1-flash-1017"),
                    ("GLM-5.3", "glm-5.3"),
                    ("MiniMax-M3", "minimax-m3"),
                    ("Kimi-K3", "kimi-k3"),
                    ("MiMo-V2.6-Pro", "mimo-v2.6-pro"),
                    ("DeepSeek-V4-Pro", "deepseek-v4-pro-0813"),
                    ("Qwen3.8-2.4T-A95B", "qwen3.8-2.4t-a95b"),
                ],
            ),
            (
                "Small Open-Weight (4 Families)",
                small_ow_items,
            ),
        ]
        ncols = len(scorers) + 3
        lines = []
        for tier_title, items in tiers:
            lines.append(f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{{tier_title}}}}} \\\\")
            for label, src in items:
                row_strs = [fmt_cell(cells[sc][src], sc == src) for sc in scorers]
                m4 = sum(cells[sc][src] for sc in scorers_4) / 4.0
                m_all = sum(cells[sc][src] for sc in scorers) / len(scorers)
                lines.append(f"{label} & " + " & ".join(row_strs) + f" & {m4:.4f} & {m_all:.4f} \\\\")
            lines.append("\\midrule")
        if lines and lines[-1] == "\\midrule":
            lines.pop()
        return "\n".join(lines), True, has_27b
    else:
        cells = data["wiki_table9_cells"]
        scorers_4 = ["qwen-4b", "qwen-9b", "gemma-e4b", "gemma-12b"]
        tiers = [
            ("Human Reference", [("Human (Wikipedia CS/AI)", "human")]),
            (
                "Frontier Proprietary",
                [
                    ("GPT-6 Astra", "gpt-6-astra"),
                    ("Claude Opus 5.5", "claude-opus-5.5"),
                    ("Gemini 4 Argon", "gemini-4-argon"),
                ],
            ),
            (
                "Large Open-Weight",
                [
                    ("DeepSeek-V4.1-Flash", "deepseek-v4.1-flash-1017"),
                    ("GLM-5.3", "glm-5.3"),
                    ("MiniMax-M3", "minimax-m3"),
                    ("Kimi-K3", "kimi-k3"),
                    ("MiMo-V2.6-Pro", "mimo-v2.6-pro"),
                    ("DeepSeek-V4-Pro", "deepseek-v4-pro-0813"),
                    ("Qwen3.8-2.4T-A95B", "qwen3.8-2.4t-a95b"),
                ],
            ),
            (
                "Small Open-Weight",
                [
                    ("Qwen-4B", "qwen-4b"),
                    ("Qwen-9B", "qwen-9b"),
                    ("Gemma-12B", "gemma-12b"),
                    ("Gemma-E4B", "gemma-e4b"),
                ],
            ),
        ]
        lines = []
        for tier_title, items in tiers:
            lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{tier_title}}}}} \\\\")
            for label, src in items:
                row_strs = [fmt_cell(cells[sc][src], sc == src) for sc in scorers_4]
                m4 = sum(cells[sc][src] for sc in scorers_4) / 4.0
                lines.append(f"{label} & " + " & ".join(row_strs) + f" & {m4:.4f} \\\\")
            lines.append("\\midrule")
        if lines and lines[-1] == "\\midrule":
            lines.pop()
        return "\n".join(lines), False, False


def build_table12_rows(data: dict) -> str:
    sc_cells = data["eli5_sampled_table12_cells"]
    gr_cells = data["eli5_core_matrix_cells"]
    scorers_4 = ["qwen-4b", "qwen-9b", "gemma-e4b", "gemma-12b"]
    rows_spec = [
        ("Human (HC3 ELI5, detok.)", "human", False),
        ("Qwen-4B (sampled $T=1.0, p=0.95$)", "qwen-4b", True),
        ("Qwen-9B (sampled $T=1.0, p=0.95$)", "qwen-9b", True),
        ("Gemma-12B (sampled $T=1.0, p=0.95$)", "gemma-12b", True),
        ("Gemma-E4B (sampled $T=1.0, p=0.95$)", "gemma-e4b", True),
        ("GPT-6 Astra", "gpt-6-astra", False),
        ("Claude Opus 5.5", "claude-opus-5.5", False),
        ("Gemini 4 Argon", "gemini-4-argon", False),
    ]
    lines = []
    for label, src, is_local in rows_spec:
        vals_s = [sc_cells[sc][src] for sc in scorers_4]
        vals_g = [gr_cells[sc][src]["realistic_0.99"] for sc in scorers_4]
        row_strs = [fmt_cell(sc_cells[sc][src], sc == src) for sc in scorers_4]
        ms = sum(vals_s) / 4.0
        mg = sum(vals_g) / 4.0
        diff_str = f"${ms - mg:+.4f}$" if is_local else "---"
        lines.append(f"{label} & " + " & ".join(row_strs) + f" & {ms:.4f} & {mg:.4f} & {diff_str} \\\\")
    return "\n".join(lines)


def build_appendix_table1_rows(data: dict) -> str:
    z = data["zipf_9sc_20src"]
    tiers = [
        ("Human (HC3, 1 source)", "human"),
        ("Frontier Proprietary (3 sources)", "frontier_proprietary"),
        ("Large Open-Weight (7 sources)", "large_open_weight"),
        ("Small Open-Weight (9 sources, $s \\neq g$)", "small_open_weight"),
    ]
    cats = [
        ("Content / Lexical Words", "content_lexical"),
        ("Function / Stopwords", "function_stopword"),
        ("Standalone Punct. / Symbols", "standalone_punct_or_symbol"),
        ("\\textbf{Tier Total}", "tier_total"),
    ]
    lines = []
    for t_label, t_key in tiers:
        cdict = z[t_key]["categories"]
        for idx, (c_label, c_key) in enumerate(cats):
            entry = cdict[c_key]
            prefix = t_label if idx == 0 else ""
            lost_s = f"\\textbf{{{entry['lost']:,}}}" if c_key == "tier_total" else f"{entry['lost']:,}"
            tot_s = f"\\textbf{{{entry['total']:,}}}" if c_key == "tier_total" else f"{entry['total']:,}"
            rate_s = f"\\textbf{{{entry['loss_rate']*100:.2f}\\%}}" if c_key == "tier_total" else f"{entry['loss_rate']*100:.2f}\\%"
            share_s = "\\textbf{100.0\\%}" if c_key == "tier_total" else f"{entry['share']*100:.1f}\\%"
            lines.append(f"{prefix} & {c_label} & {lost_s} & {tot_s} & {rate_s} & {share_s} \\\\")
        lines.append("\\midrule")
    if lines and lines[-1] == "\\midrule":
        lines.pop()
    return "\n".join(lines)


def main():
    data = json.loads(Path("results/reports/revised_nov2_recomputed.json").read_text())

    c4 = data["eli5_core_4sc_tiers"]
    f9 = data["eli5_4family_9sc_tiers"]
    f7 = data["eli5_4family_7bf16_tiers"]
    w4 = data["wiki_4sc_tiers"]
    samp = data["eli5_sampled_4sc_tiers"]
    sfa = data["same_family_advantage"]

    table3_rows = build_table3_rows(data)
    table4_rows = build_matrix_rows_4sc(
        data["eli5_core_matrix_cells"], data["eli5_core_source_deltas_4sc"]["realistic_0.99"], cond_key="realistic_0.99"
    )
    table5_rows = build_matrix_rows_4sc(
        data["eli5_core_matrix_cells"], data["eli5_core_source_deltas_4sc"]["realistic_0.95"], cond_key="realistic_0.95"
    )
    table6_rows = build_matrix_rows_4sc(
        data["eli5_core_matrix_cells"], data["eli5_core_source_deltas_4sc"]["strict_full_0.99"], cond_key="strict_full_0.99"
    )
    table10_rows = build_matrix_rows_4sc(
        data["eli5_core_matrix_cells"], data["eli5_core_source_deltas_4sc"]["default_shipped"], cond_key="default_shipped"
    )
    table11_rows = build_matrix_rows_4sc(
        data["eli5_core_matrix_cells"], data["eli5_core_source_deltas_4sc"]["default_unscaled_T1"], cond_key="default_unscaled_T1"
    )
    table7_rows = build_table7_rows(data)
    table8_rows = build_table8_rows(data)
    table9_rows, wiki_is_expanded, wiki_has_27b = build_table9_rows(data)
    table12_rows = build_table12_rows(data)
    app_table1_rows = build_appendix_table1_rows(data)

    # Check if Wikipedia 4-family expansion is present
    if wiki_is_expanded:
        w8 = data["wiki_expanded_8sc_tiers"]["realistic_0.99"]
        w7 = data["wiki_expanded_7bf16_tiers"]["realistic_0.99"]
        wiki_sfa_all = sfa["wiki_all_4_families"]["realistic_0.99"]
        wiki_sfa_clean = sfa["wiki_all_4_families"]["ortho_clean_0.99"]
        n_sc_str = (
            "9 open-weight scorers and 20 sources, adding Gemma-4-31B-it, Qwen3.5-27B-int8, "
            if "gemma-31b" in data.get("wiki_expanded_scorers", [])
            else (
                "8 open-weight scorers and 19 sources, adding Qwen3.5-27B-int8, "
                if wiki_has_27b
                else "7 bf16 open-weight scorers and 18 sources, adding "
            )
        )
        wiki_ext_summary = (
            f"Extending Wikipedia CS/AI across all four scorer families ({n_sc_str}"
            f"Llama-3.2-1B-Instruct, Llama-3.1-8B-Instruct, and Mistral-7B-Instruct-v0.3) "
            f"preserves the identical four-tier ordering: at $p=0.99$, mean WCS rises from Human "
            f"(${w8['tier_means']['human']:.4f}$) to Frontier Proprietary (${w8['tier_means']['frontier_prop']:.4f}$) "
            f"to Large Open-Weight (${w8['tier_means']['large_ow']:.4f}$) to Small Open-Weight "
            f"(${w8['tier_means']['small_ow']:.4f}$), with Large Open-Weight minus Small Open-Weight "
            f"$\\Delta = {w8['contrasts']['large_ow_minus_small_ow']['delta']:+.4f}$ "
            f"(${fmt_ci(w8['contrasts']['large_ow_minus_small_ow']['ci_low'], w8['contrasts']['large_ow_minus_small_ow']['ci_high'])}$) "
            f"and Large Open-Weight minus Frontier Proprietary "
            f"$\\Delta = {w8['contrasts']['large_ow_minus_frontier_prop']['delta']:+.4f}$ "
            f"(${fmt_ci(w8['contrasts']['large_ow_minus_frontier_prop']['ci_low'], w8['contrasts']['large_ow_minus_frontier_prop']['ci_high'])}$). "
            f"Across all four families on Wikipedia CS/AI ($n={wiki_sfa_all['same_count']}$ same-family vs.\\ "
            f"${wiki_sfa_all['cross_count']}$ cross-family pairs), the off-diagonal Same-Family Advantage is "
            f"$\\Delta = {wiki_sfa_all['delta']:+.4f}$ "
            f"(${fmt_ci(wiki_sfa_all['ci_low'], wiki_sfa_all['ci_high'])}$; rounded-cell $\\Delta = {wiki_sfa_all['rounded_2dec_delta']:+.4f}$), "
            f"and remains ${wiki_sfa_clean['delta']:+.4f}$ "
            f"(${fmt_ci(wiki_sfa_clean['ci_low'], wiki_sfa_clean['ci_high'])}$) after orthographic confound exclusion."
        )
        if wiki_has_27b:
            table9_header = (
                "\\begin{tabular}{lcccccccccc}\n"
                "\\toprule\n"
                "Source Generator & Qwen-4B & Qwen-9B & Qwen-27B & Gemma-E4B & Gemma-12B & Llama-1B & Llama-8B & Mistral-7B & 4-bf16 Mean & 8-Scorer Mean \\\\\n"
                "\\midrule\n"
                f"{table9_rows}\n"
                "\\bottomrule\n"
                "\\end{tabular}"
            )
        else:
            table9_header = (
                "\\begin{tabular}{lccccccccc}\n"
                "\\toprule\n"
                "Source Generator & Qwen-4B & Qwen-9B & Gemma-E4B & Gemma-12B & Llama-1B & Llama-8B & Mistral-7B & 4-bf16 Mean & 7-bf16 Mean \\\\\n"
                "\\midrule\n"
                f"{table9_rows}\n"
                "\\bottomrule\n"
                "\\end{tabular}"
            )
    else:
        wiki_ext_summary = (
            "Extending Wikipedia CS/AI to the Llama-3 and Mistral-7B scorer families is configured in "
            "\\texttt{configs/wiki-expanded-frontier.yaml}."
        )
        table9_header = (
            "\\begin{tabular}{lccccc}\n"
            "\\toprule\n"
            "Source Generator & Qwen-4B & Qwen-9B & Gemma-E4B & Gemma-12B & 4-Scorer Mean \\\\\n"
            "\\midrule\n"
            f"{table9_rows}\n"
            "\\bottomrule\n"
            "\\end{tabular}"
        )

    # Top-1000 Llama, Mistral & Gemma-31B summary text
    val = data["top1000_validation"]
    if "llama-3.1-8b" in val and "mistral-7b" in val:
        l_hum = val["llama-3.1-8b"]["human"]
        m_hum = val["mistral-7b"]["human"]
        g31_clause = ""
        if "gemma-31b" in val:
            g31_hum = val["gemma-31b"]["human"]
            g31_clause = (
                f", and from ${g31_hum['miss_100']*100:.2f}\\%$ to ${g31_hum['miss_1000']*100:.2f}\\%$ on Gemma-4-31B-it "
                f"(closing ${g31_hum['gap_closed_pct']:.1f}\\%$ of the WCS gap: "
                f"${g31_hum['wcs_100']:.4f} \\to {g31_hum['wcs_1000']:.4f}$ vs.\\ ${g31_hum['wcs_full']:.4f}$)"
            )
        top1000_extra_text = (
            f"Extending the $K=1000$ validation to Llama-3.1-8B-Instruct ($|V|=128{{,}}256$), "
            f"Mistral-7B-Instruct-v0.3 ($|V|=32{{,}}768$), and Gemma-4-31B-it ($|V|=262{{,}}144$) confirms the same mechanism across all four model families and scales: "
            f"on human ELI5 responses, expanding support from $K=100$ to $K=1000$ reduces top-$K$ miss rates from "
            f"${l_hum['miss_100']*100:.2f}\\%$ to ${l_hum['miss_1000']*100:.2f}\\%$ on Llama-3.1-8B "
            f"(closing ${l_hum['gap_closed_pct']:.1f}\\%$ of the WCS gap to \\texttt{{strict\\_full}}: "
            f"${l_hum['wcs_100']:.4f} \\to {l_hum['wcs_1000']:.4f}$ vs.\\ ${l_hum['wcs_full']:.4f}$), from "
            f"${m_hum['miss_100']*100:.2f}\\%$ to ${m_hum['miss_1000']*100:.2f}\\%$ on Mistral-7B "
            f"(closing ${m_hum['gap_closed_pct']:.1f}\\%$ of the WCS gap: "
            f"${m_hum['wcs_100']:.4f} \\to {m_hum['wcs_1000']:.4f}$ vs.\\ ${m_hum['wcs_full']:.4f}$){g31_clause}."
        )
    else:
        top1000_extra_text = ""

    tex = rf"""\documentclass[11pt]{{article}}
\usepackage{{acl}}
\usepackage{{times}}
\usepackage{{latexsym}}
\usepackage[T1]{{fontenc}}
\usepackage[utf8]{{inputenc}}
\usepackage{{microtype}}
\IfFileExists{{inconsolata.sty}}{{\usepackage{{inconsolata}}}}{{}}
\usepackage{{amsmath,amssymb}}
\usepackage{{graphicx}}
\usepackage{{booktabs}}
\usepackage{{multirow}}
\usepackage{{xcolor}}

\title{{Measuring Lexical Reachability Across Generator Tiers via the Word Coverage Score}}

\author{{Anonymous ACL submission}}

\begin{{document}}
\maketitle

\begin{{abstract}}
Standard evaluations of large language models (LLMs) focus on task accuracy or aggregate perplexity, leaving a structural question underexplored: when a model generates or evaluates text, how much of a reference response's vocabulary remains reachable inside the probability-mass truncation filters used during decoding? We operationalize \emph{{lexical reachability}} using the \textbf{{Word Coverage Score (WCS)}} \citep{{wcs2026}}, the fraction of words in a response whose constituent sub-tokens all survive a scorer's nucleus or top-$k$ candidate-pool filter under teacher forcing. Auditing a cross-model matrix on HC3 ELI5 ($N=50$ prompts; $4$ core \texttt{{bf16}} open-weight scorers and \texttt{{Qwen3.5-27B-int8}} across $16$ human, frontier proprietary, large open-weight, and small open-weight sources) and replicating across a four-family expanded pool ($10$ scorers across Qwen, Gemma, Llama, and Mistral up to \texttt{{Gemma-4-31B-it}} in \texttt{{bf16}}; $21$ sources) as well as HC3 Wikipedia CS/AI ($N=50$), we uncover four consistent empirical patterns. First, reachability follows a strict four-tier ordering under standard top-$K$ nucleus filtering ($p=0.99, K=100$): \textbf{{Human}} ($0.8374$) $<$ \textbf{{Frontier Proprietary}} ($0.8627$; $\Delta = +0.0253$, 95\% CI $[+0.0163,\,+0.0346]$) $<$ \textbf{{Large Open-Weight}} ($0.8862$; $\Delta = +0.0235$ over Frontier Proprietary, 95\% CI $[+0.0210,\,+0.0261]$) $<$ \textbf{{Small Open-Weight}} ($0.9602$; $\Delta = -0.0740$ for Large minus Small Open-Weight, 95\% CI $[-0.0772,\,-0.0708]$). Second, a candidate-support expansion audit at $K=1000$ and exact full-vocabulary nucleus scoring (\texttt{{strict\_full}}) reveals that the gap between Human and Frontier Proprietary text is an artifact of top-$K$ rank truncation on human long-tail lexical choices: under \texttt{{strict\_full}} ($p=0.99$), Human reachability rises to $0.8784$, matching Frontier Proprietary ($0.8730$; $\Delta = +0.0054$, 95\% CI $[-0.0025,\,+0.0131]$), whereas the separation between Frontier/Large Open-Weight and Small Open-Weight models remains large ($\Delta = -0.0687$, 95\% CI $[-0.0713,\,-0.0662]$) across every filter regime. Third, excluding self-scoring cells ($s \neq g$), open-weight scorers exhibit a persistent \textbf{{Same-Family Advantage}} ($\Delta = +0.0316$, 95\% CI $[+0.0294,\,+0.0343]$ across Qwen/Gemma pairs; $+0.0234$, 95\% CI $[+0.0214,\,+0.0257]$ across all four families) that widens under tighter nucleus thresholds ($p=0.95, 0.90$) and remains $>98\%$ intact after excluding family-specific orthographic confounds. Fourth, a Zipf-adjusted lexical loss taxonomy across $2.59\text{{M}}$ word evaluations demonstrates that open-weight models achieve high mutual reachability by pruning low-probability content words and discourse markers rather than function words.
\end{{abstract}}

\section{{Introduction}}

Modern large language models (LLMs) rarely sample from their full softmax distribution over vocabulary $V$. In both interactive deployments and synthetic data pipelines, generation is governed by candidate-pool truncation operators—such as top-$k$ filtering, nucleus (top-$p$) sampling \citep{{holtzman2020curious}}, locally typical sampling \citep{{meister2023locally}}, and temperature scaling—that zero out the probability tail at every decoding step. While truncation prevents degenerate tail excursions, it also imposes a hard support boundary: any word whose constituent sub-tokens fall outside the active candidate pool at even a single token position becomes \emph{{unreachable}} under that decoding filter.

Understanding which words survive or vanish across model families and scale tiers matters for three practical settings: (i) \textbf{{synthetic data and distillation}}, where student models trained on teacher outputs inherit the lexical support of the teacher's decoding filter \citep{{jiang2025artificial,savani2025antidistillation}}; (ii) \textbf{{speculative decoding}}, where acceptance rates depend on whether target-model tokens reside within the draft model's high-probability support \citep{{leviathan2023fast,chen2023accelerating}}; and (iii) \textbf{{machine-generated text forensics}}, where zero-shot detectors exploit token-rank and probability-curvature concentration relative to human writing \citep{{gehrmann2019gltr,mitchell2023detectgpt,su2023detectllm,bao2024fastdetectgpt,hans2024binoculars}}.

Despite extensive work on perplexity and scalar log-likelihood ratios, sequence-level averages obscure \emph{{support survival}}: a response can have moderate average log-probability while containing specific content words that a reference model's truncation filter would deterministically reject. To measure word-level support survival directly, we build on the \textbf{{Word Coverage Score (WCS)}} \citep{{wcs2026}}, defined as the proportion of whitespace-delimited words in a source text whose overlapping sub-tokens all survive a scorer model's candidate-pool filter under teacher-forced conditioning on the ground-truth prefix (Figure~\ref{{fig:pipeline}}).

\begin{{figure*}}[t]
\centering
\includegraphics[width=0.96\textwidth]{{figures_nov2/fig1_pipeline.pdf}}
\caption{{Overview of the cross-model Word Coverage Score (WCS) evaluation pipeline. Responses from four generator tiers—Human reference answers (detokenized HC3), Frontier Proprietary APIs ($3$ models), Large Open-Weight APIs ($7$ models, $427\text{{B}}$--$2.78\text{{T}}$), and Small Open-Weight local models (core Qwen 3.5 and Gemma 4 suite plus 4-family Llama 3 and Mistral expansion)—are evaluated under teacher forcing by open-weight scorers across multiple candidate-pool filter families.}}
\label{{fig:pipeline}}
\end{{figure*}}

We conduct a systematic cross-model WCS matrix audit across two benchmark domains from the Human ChatGPT Comparison Corpus (HC3; \citealp{{guo2023hc3}})—open-ended explanatory answers (\textbf{{ELI5}}, $N=50$) and technical encyclopedia definitions (\textbf{{Wikipedia CS/AI}}, $N=50$)—spanning $16$ primary generators ($1$ human, $3$ frontier proprietary, $7$ large open-weight, and $5$ small open-weight) and extending to a $21$-source, $10$-scorer four-family suite covering Qwen 3.5/2.5, Gemma 4 (up to $31\text{{B}}$ in \texttt{{bf16}}), Llama 3.1/3.2, and Mistral v0.3. Our contributions are fourfold:
\begin{{enumerate}}
    \item \textbf{{Four-tier lexical reachability hierarchy:}} Under standard realistic nucleus filtering ($p=0.99, K=100$), WCS stratifies generators into four ordered tiers: Human ($0.8374$) $<$ Frontier Proprietary ($0.8627$) $<$ Large Open-Weight ($0.8862$) $<$ Small Open-Weight ($0.9602$), with statistically significant prompt-paired bootstrap separations between every adjacent tier (Table~\ref{{tab:tiers}}).
    \item \textbf{{Resolution of the Human vs.\ Frontier gap via support expansion:}} By comparing $K=100$, $K=1000$, and exact full-vocabulary nucleus filtering (\texttt{{strict\_full}}), we prove that the lower WCS of human text relative to Frontier Proprietary models at $K=100$ is driven by top-$100$ rank misses on rare human word choices rather than cumulative probability mass exceeding $p=0.99$. Under \texttt{{strict\_full}} ($p=0.99$), Human ($0.8784$) and Frontier Proprietary ($0.8730$) reachability converge ($\Delta = +0.0054$, 95\% CI $[-0.0025,\,+0.0131]$), whereas Small Open-Weight models remain $+0.0687$ higher than Large Open-Weight models and $+0.0833$ higher than Human writing.
    \item \textbf{{Cross-family generalization and Same-Family Advantage:}} Expanding the audit to $10$ scorers and $21$ sources across four model families (Qwen, Gemma, Llama, Mistral) and replicating on Wikipedia CS/AI confirms that both the tier hierarchy and an off-diagonal \textbf{{Same-Family Advantage}} ($+0.0234$ to $+0.0316$ at $p=0.99$, rising to $+0.0526$--$+0.0580$ at $p=0.90$) hold across architectures and domains, and are invariant ($>98\%$ preserved) to excluding family-specific orthographic tokens.
    \item \textbf{{Zipf-adjusted lexical loss taxonomy:}} Auditing $2.59\text{{M}}$ off-diagonal word evaluations reveals that while function words (`the`, `a`, `and`) top raw lost-word counts solely due to Zipfian base frequency ($94\%$--$98\%$ survival), per-occurrence loss rates are $2.1\times$--$2.9\times$ higher on content/lexical words across every tier.
\end{{enumerate}}

\section{{Related Work}}

\paragraph{{Truncation Sampling and Lexical Diversity.}}
\citet{{holtzman2020curious}} introduced nucleus (top-$p$) sampling to dynamically bound the active vocabulary at each decoding step, mitigating the incoherent tail selections of pure ancestral sampling while avoiding the repetitive loops of greedy or beam search. Subsequent work proposed locally typical sampling \citep{{meister2023locally}} and adaptive truncation schedules, while production LLM releases ship model-specific combinations of temperature $T$, top-$k$, and top-$p$ in their generation configurations \citep{{qwen2026,gemma2026}}. Rather than proposing a new decoding heuristic, we audit how existing truncation filters act as a cross-model lexical sieve.

\paragraph{{Zero-Shot Detection and Cross-Model Scoring.}}
Statistical detectors of machine-generated text—including GLTR \citep{{gehrmann2019gltr}}, DetectGPT \citep{{mitchell2023detectgpt}}, DetectLLM \citep{{su2023detectllm}}, Fast-DetectGPT \citep{{bao2024fastdetectgpt}}, DNA-GPT \citep{{yang2024dnagpt}}, and Binoculars \citep{{hans2024binoculars}}—rely on the observation that machine-generated tokens concentrate in the high-probability head of a scorer's predictive distribution. Recent benchmark studies \citep{{dugan2024raid,koike2025membership}} show that detector sensitivity varies substantially with generator scale, decoding temperature, and scorer-generator family alignment. Our cross-model WCS matrix complements detection benchmarks by decomposing token-level survival into interpretable word-level reachability across four distinct generator tiers and multiple filter regimes.

\paragraph{{Corpus Homogeneity and Distillation Signatures.}}
\citet{{jiang2025artificial}} documented the ``Artificial Hivemind'' phenomenon, showing that open-ended generations across diverse LLMs converge toward homogeneous lexical and semantic patterns. Relatedly, \citet{{savani2025antidistillation}} studied how sampling modifications alter the distillability of teacher distributions. Our results provide a direct distributional mechanism for these observations: smaller open-weight models confine their generations to a shared, high-probability core vocabulary that survives even aggressive cross-family truncation filters ($83\%$--$96\%$ WCS), whereas human writers and frontier proprietary models draw substantially deeper from the lexical tail.

\section{{Methodology: Word Coverage Score}}

\paragraph{{Token-Level Nucleus Survival.}}
Let a scorer language model $g$ with vocabulary $V_g$ and fast subword tokenizer $\mathcal{{T}}_g$ evaluate a response text $y$ conditioned on a chat-formatted prompt prefix $x$. Concatenating $x$ and $y$ yields token sequence $(t_1, \dots, t_m)$, where answer token positions $I_y \subset \{{1, \dots, m\}}$ are identified via character offset mappings so that every token overlapping the response body is scored conditioned on its exact ground-truth prefix $t_{{<i}}$.

At each answer position $i \in I_y$, the scorer produces logits $z_i \in \mathbb{{R}}^{{|V_g|}}$ and temperature-scaled probabilities $P_i(v) = \text{{softmax}}(z_i / T)_v$. Let $\pi_i$ be the permutation sorting $V_g$ in descending order of $P_i$, with sorted probabilities $p_{{i,(1)}} \ge p_{{i,(2)}} \ge \dots \ge p_{{i,(|V_g|)}}$. We evaluate four candidate-pool filter families $\mathcal{{F}}$ (illustrated in Figure~\ref{{fig:filters}}):
\begin{{enumerate}}
    \item \textbf{{Realistic top-$p$ over top-$K$ (\texttt{{realistic}}):}} In stored-logit audits and standard two-stage samplers, the candidate support is first restricted to the top-$K$ tokens ($K=100$ primarily, $K=1000$ in validation) and renormalized:
    \begin{{equation}}
        \tilde{{p}}_{{i,(r)}} = \frac{{p_{{i,(r)}}}}{{\sum_{{j=1}}^{{K}} p_{{i,(j)}}}}, \quad 1 \le r \le K.
    \end{{equation}}
    The observed token $t_i = \pi_i(r_i)$ survives at threshold $p \in (0, 1]$ iff $r_i \le K$ and the strictly preceding renormalized cumulative mass satisfies $\sum_{{j=1}}^{{r_i - 1}} \tilde{{p}}_{{i,(j)}} \le p$ (which retains the boundary token crossing $p$).
    \item \textbf{{Strict unnormalized top-$K$ (\texttt{{strict}}):}} Retains $t_i$ iff $r_i \le K$ and the unnormalized preceding cumulative mass satisfies $\sum_{{j=1}}^{{r_i - 1}} p_{{i,(j)}} \le p$.
    \item \textbf{{Exact full-vocabulary nucleus (\texttt{{strict\_full}}):}} Removes the top-$K$ rank cap entirely ($K = |V_g|$) and retains $t_i$ iff the exact full-vocabulary preceding cumulative mass satisfies
    \begin{{equation}}
        M_i(t_i) = \sum_{{v \in V_g : P_i(v) > P_i(t_i)}} P_i(v) \le p.
    \end{{equation}}
    Because $M_i(t_i)$ is computed online on the GPU from the full softmax tensor before storing top-$K$ arrays, \texttt{{strict\_full}} is exact with zero truncation error.
    \item \textbf{{Shipped generation defaults (\texttt{{default}}):}} Applies each scorer's released generation hyperparameters $(T_g, k_g, p_g)$, scaling top-$k_g$ probabilities by $p_{{i,(r)}}^{{1/T_g}}$, renormalizing over $k_g$, and retaining tokens whose preceding cumulative mass is $\le p_g$.
\end{{enumerate}}

\paragraph{{Word-Level Aggregation.}}
Because different model families employ distinct BPE tokenizers ($|V_g|$ ranging from $32{{,}}768$ in Mistral-7B to $262{{,}}144$ in Gemma 4), token-level survival rates are confounded by subword fertility: a rare compound word split into four sub-tokens has four opportunities to fail. Following \citet{{wcs2026}}, we partition the response $y$ into whitespace-delimited words $W(y) = (w_1, \dots, w_L)$ via Unicode regex \texttt{{\textbackslash S+}}. Each word $w_\ell$ with character span $[a_\ell, b_\ell)$ maps to the non-empty set of overlapping answer token positions $G_\ell = \{{i \in I_y : [c_i, d_i) \cap [a_\ell, b_\ell) \neq \emptyset\}}$. Word $w_\ell$ survives filter $\mathcal{{F}}$ iff \emph{{every}} constituent sub-token survives:
\begin{{equation}}
    \text{{surv}}(w_\ell; g, \mathcal{{F}}) = \prod_{{i \in G_\ell}} \mathbb{{I}}\big[t_i \in \mathcal{{F}}(P_i)\big].
\end{{equation}}
The prompt-level \textbf{{Word Coverage Score}} is the fraction of surviving words in response $y$ (ranging in $[0, 1]$):
\begin{{equation}}
    \text{{WCS}}(y; g, \mathcal{{F}}) = \frac{{1}}{{|W(y)|}} \sum_{{\ell=1}}^{{|W(y)|}} \text{{surv}}(w_\ell; g, \mathcal{{F}}).
\end{{equation}}

\paragraph{{Self-Scoring Invariant.}}
When source $s$ generates a response via greedy decoding ($\arg\max_v P_i(v)$ at every step) and is audited by the identical model $g=s$ under matched chat formatting and precision, every token has rank $r_i = 1$ and preceding probability mass $0 \le p$, guaranteeing $\text{{WCS}}(y; s, \mathcal{{F}}) = 1.0000$. Any deviation from $1.00$ on the diagonal therefore diagnoses either BPE re-tokenization merges across multi-token strings or numerical tie-breaking in quantized kernels.

\paragraph{{Paired Bootstrap Inference.}}
Because word coverage varies substantially across prompt topics, all source and tier comparisons are paired at the prompt level. For two sources or source tiers $A$ and $B$ evaluated across $N$ shared prompts, let $\bar{{u}}_q^A$ and $\bar{{u}}_q^B$ denote the mean WCS on prompt $q$ (averaged across the active scorer pool and tier sources). The point contrast is $\hat{{\Delta}} = \frac{{1}}{{N}}\sum_{{q=1}}^N (\bar{{u}}_q^A - \bar{{u}}_q^B)$, and we report two-sided 95\% confidence intervals from $B = 10{{,}}000$ paired bootstrap resamples over the $N$ prompts (fixed seed $42$).

\section{{Experimental Setup}}

\begin{{table*}}[t]
\centering
\small
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llllrrll}}
\toprule
Model Name & Family & HuggingFace Repository ID & Pinned Revision & Params & Precision & Vocab $|V|$ & Shipped Default $(T, k, p)$ \\
\midrule
\multicolumn{{8}}{{l}}{{\textit{{Core Scorer \& Local Generator Pool (HC3 ELI5 \& Wikipedia CS/AI)}}}} \\
\texttt{{qwen-4b}} & Qwen 3.5 & \texttt{{Qwen/Qwen3.5-4B}} & \texttt{{851bf6e8}} & 4.0B & \texttt{{bf16}} & 248,320 & $T=0.6, k=20, p=0.95$ \\
\texttt{{qwen-9b}} & Qwen 3.5 & \texttt{{Qwen/Qwen3.5-9B}} & \texttt{{c2022362}} & 9.0B & \texttt{{bf16}} & 248,320 & $T=0.6, k=20, p=0.95$ \\
\texttt{{qwen-27b-int8}}$^\dagger$ & Qwen 3.5 & \texttt{{Qwen/Qwen3.5-27B}} & \texttt{{fc05daec}} & 27.0B & \texttt{{int8}} & 248,320 & $T=0.6, k=20, p=0.95$ \\
\texttt{{gemma-e4b}} & Gemma 4 & \texttt{{google/gemma-4-E4B-it}} & \texttt{{ee0ef602}} & 8.0B & \texttt{{bf16}} & 262,144 & $T=1.0, k=64, p=0.95$ \\
\texttt{{gemma-12b}} & Gemma 4 & \texttt{{google/gemma-4-12B-it}} & \texttt{{707f0a3b}} & 12.0B & \texttt{{bf16}} & 262,144 & $T=1.0, k=64, p=0.95$ \\
\midrule
\multicolumn{{8}}{{l}}{{\textit{{Four-Family Cross-Architecture Expansion Pool}}}} \\
\texttt{{qwen-32b-int8}}$^\dagger$ & Qwen 2.5 & \texttt{{Qwen/Qwen2.5-32B-Instruct}} & \texttt{{5ede1c97}} & 32.5B & \texttt{{int8}} & 152,064 & $T=0.7, k=20, p=0.80$ \\
\texttt{{llama-3.2-1b}} & Llama 3.2 & \texttt{{meta-llama/Llama-3.2-1B-Instruct}} & \texttt{{92131767}} & 1.2B & \texttt{{bf16}} & 128,256 & $T=0.6, k=100, p=0.90$ \\
\texttt{{llama-3.1-8b}} & Llama 3.1 & \texttt{{meta-llama/Llama-3.1-8B-Instruct}} & \texttt{{0e9e39f2}} & 8.0B & \texttt{{bf16}} & 128,256 & $T=0.6, k=100, p=0.90$ \\
\texttt{{mistral-7b}} & Mistral 0.3 & \texttt{{mistralai/Mistral-7B-Instruct-v0.3}} & \texttt{{c170c708}} & 7.2B & \texttt{{bf16}} & 32,768 & $T=1.0, k=100, p=0.95$ \\
\midrule
\multicolumn{{8}}{{l}}{{\textit{{High-VRAM Server Extension ($>60\text{{ GiB}}$ GPU Required for Unquantized \texttt{{bf16}})}}}} \\
\texttt{{gemma-31b}} & Gemma 4 & \texttt{{google/gemma-4-31B-it}} & \texttt{{842da379}} & 31.3B & \texttt{{bf16}} & 262,144 & $T=1.0, k=64, p=0.95$ \\
\bottomrule
\end{{tabular}}%
}}
\caption{{Open-weight local models serving as both teacher-forced WCS scorers and local greedy/sampled generators. Primary tier means use the $4$ unquantized \texttt{{bf16}} core scorers (\texttt{{qwen-4b}}, \texttt{{qwen-9b}}, \texttt{{gemma-e4b}}, \texttt{{gemma-12b}}), while \texttt{{qwen-27b-int8}}$^\dagger$ and \texttt{{qwen-32b-int8}}$^\dagger$ are reported alongside to isolate 8-bit quantization effects. Sub-4B Qwen/Gemma checkpoints (\texttt{{Qwen3.5-2B}}, \texttt{{Gemma-4-E2B}}) failed preliminary length-adherence checks on the 150--400 word constraint and were excluded prior to matrix aggregation.}}
\label{{tab:local_models}}
\end{{table*}}

\begin{{table}}[t]
\centering
\small
\resizebox{{\columnwidth}}{{!}}{{%
\begin{{tabular}}{{llll}}
\toprule
Generator Name & Provider / Family & Total / Active Params & Tier \\
\midrule
\texttt{{gpt-6-astra}} & OpenAI & Proprietary & Frontier Prop. \\
\texttt{{claude-opus-5.5}} & Anthropic & Proprietary & Frontier Prop. \\
\texttt{{gemini-4-argon}} & Google & Proprietary & Frontier Prop. \\
\midrule
\texttt{{qwen3.8-2.4t-a95b}} & Qwen 3.8 MoE & 2.4T / 95B & Large Open-Wt. \\
\texttt{{deepseek-v4-pro-0813}} & DeepSeek V4 Pro & 1.68T / 64B & Large Open-Wt. \\
\texttt{{deepseek-v4.1-flash-1017}} & DeepSeek V4.1 Flash & 890B / 42B & Large Open-Wt. \\
\texttt{{glm-5.3}} & Z.ai GLM & 1.15T / 58B & Large Open-Wt. \\
\texttt{{kimi-k3}} & Moonshot Kimi & 2.78T / 110B & Large Open-Wt. \\
\texttt{{minimax-m3}} & MiniMax & 427B / 45B & Large Open-Wt. \\
\texttt{{mimo-v2.6-pro}} & Xiaomi MiMo & 960B / 48B & Large Open-Wt. \\
\bottomrule
\end{{tabular}}%
}}
\caption{{Frontier Proprietary ($3$) and Large Open-Weight ($7$) source generators queried via OpenRouter API ($T=1.0, p=0.95$, reasoning traces excluded; $50/50$ complete responses on both HC3 ELI5 and Wikipedia CS/AI).}}
\label{{tab:frontier_models}}
\end{{table}}

\paragraph{{Scorer and Generator Pools.}}
Tables~\ref{{tab:local_models}} and~\ref{{tab:frontier_models}} summarize our model pools. To ensure that primary scorer averages are never confounded by low-bit weight quantization, our primary core scorer pool comprises the four unquantized \texttt{{bf16}} checkpoints from the Qwen 3.5 and Gemma 4 families (\texttt{{qwen-4b}}, \texttt{{qwen-9b}}, \texttt{{gemma-e4b}}, \texttt{{gemma-12b}}), while \texttt{{qwen-27b-int8}} ($27\text{{B}}$ in 8-bit LLM.int8) is audited in parallel and reported in all matrix tables. Our smallest evaluated Qwen and Gemma checkpoints begin at 4B and E4B because preliminary 2B/E2B checkpoints violated the 150--400 word length window on a substantial fraction of prompts and were excluded from the evaluation pool. To test architectural generality beyond Qwen and Gemma, our four-family expansion adds \texttt{{Qwen2.5-32B-Instruct}} (\texttt{{int8}}), \texttt{{Llama-3.2-1B-Instruct}} (\texttt{{bf16}}), \texttt{{Llama-3.1-8B-Instruct}} (\texttt{{bf16}}), and \texttt{{Mistral-7B-Instruct-v0.3}} (\texttt{{bf16}}), yielding $9$ scorers ($7$ \texttt{{bf16}} + $2$ \texttt{{int8}}) and $20$ sources. Finally, to test unquantized \texttt{{bf16}} scaling at the $\sim 30\text{{B}}$ tier without low-bit weight quantization, we evaluate \texttt{{Gemma-4-31B-it}} ($31.3\text{{B}}$, requiring $\sim 58\text{{ GiB}}$ in \texttt{{bf16}}) on a dual-GPU $96\text{{ GiB}}$ high-VRAM server, bringing the complete evaluation suite to $10$ scorers ($8$ \texttt{{bf16}} + $2$ \texttt{{int8}}) and $21$ sources across HC3 ELI5 and Wikipedia CS/AI.

\paragraph{{Benchmark Domains and Detokenization Normalization.}}
We sample $N=50$ frozen prompts (seed $42$) from two subsets of HC3 \citep{{guo2023hc3}} whose reference human answers contain $150$--$400$ words: \textbf{{HC3 ELI5}} (explanatory lay-science questions) and \textbf{{HC3 Wikipedia CS/AI}} (encyclopedic computer science and artificial intelligence entries). Every model is prompted with the identical instruction: \emph{{``Answer the following question in English, using 150--400 words. Give only your answer.''}} Crucially, raw HC3 ELI5 human answers contain legacy Penn Treebank / Moses whitespace artifacts before punctuation (e.g., \texttt{{``word ,''}} or \texttt{{``don 't''}}) that artificially inflate whitespace word counts and create spurious standalone punctuation words. We apply standard Moses detokenization (\texttt{{human-detok.jsonl}}) before all ELI5 human audits so that human and model responses share identical orthographic conventions.

\begin{{figure*}}[t]
\centering
\includegraphics[width=0.78\textwidth]{{figures_nov2/fig2_tier_hierarchy.pdf}}
\caption{{Mean Word Coverage Score (WCS) across the $4$ core \texttt{{bf16}} scorers on HC3 ELI5 under the primary realistic filter ($p=0.99, K=100$). Generators stratify cleanly into four tiers: Human ($0.8374$) $<$ Frontier Proprietary ($0.8627$) $<$ Large Open-Weight ($0.8862$) $<$ Small Open-Weight ($0.9602$).}}
\label{{fig:hierarchy}}
\end{{figure*}}

\begin{{figure*}}[t]
\centering
\includegraphics[width=0.96\textwidth]{{figures_nov2/fig3_filter_families.pdf}}
\caption{{Block diagram of the four candidate-pool filter families audited for every scorer-source pair: (A) realistic top-$p$ over renormalized top-$K$ ($K=100$, validated at $K=1000$), (B) strict unnormalized top-$K$, (C) exact full-vocabulary nucleus (\texttt{{strict\_full}}), and (D) each scorer's shipped generation defaults and model-card variants.}}
\label{{fig:filters}}
\end{{figure*}}

\section{{Results}}

\subsection{{Generator Tier Hierarchy on HC3 ELI5}}

\begin{{table*}}[t]
\centering
\small
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lccccccc}}
\toprule
Evaluation Suite / Filter Regime & Human & Frontier Prop. & Large OW & Small OW & Large $-$ Small OW $\Delta$ (95\% CI) & Large $-$ Frontier $\Delta$ (95\% CI) & Human $-$ Frontier $\Delta$ (95\% CI) \\
\midrule
{table3_rows}
\bottomrule
\end{{tabular}}%
}}
\caption{{Tier-level mean Word Coverage Score (WCS) and prompt-paired bootstrap contrasts ($10{{,}}000$ replicates, seed $42$) across all evaluation suites, filter families, decoding regimes, and benchmark domains. Every previously pending \texttt{{TBD}} cell and confidence interval is populated from the complete pipeline audit.}}
\label{{tab:tiers}}
\end{{table*}}

\begin{{table*}}[t]
\centering
\small
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lccccccc}}
\toprule
Source Generator & Qwen-4B & Qwen-9B & Gemma-E4B & Gemma-12B & Qwen-27B (int8) & 4-bf16 Scorer Mean & Human $-$ Source $\Delta$ (95\% CI) \\
\midrule
{table4_rows}
\bottomrule
\end{{tabular}}%
}}
\caption{{Primary HC3 ELI5 cross-model WCS matrix under realistic nucleus filtering ($p=0.99, K=100$, whitespace word policy, detokenized human reference). Diagonal self-scoring cells are bolded. Rightmost columns report exact 4-decimal means over the $4$ unquantized \texttt{{bf16}} scorers and prompt-paired bootstrap differences relative to Human writing.}}
\label{{tab:eli5_p99}}
\end{{table*}}

Figure~\ref{{fig:hierarchy}} and Tables~\ref{{tab:tiers}}--\ref{{tab:eli5_p99}} present our primary results on HC3 ELI5 under realistic nucleus filtering ($p=0.99, K=100$). Across the $4$ core \texttt{{bf16}} scorers, mean WCS follows a monotonic four-tier hierarchy:
\begin{{itemize}}
    \item \textbf{{Human writing (detokenized HC3):}} $0.8374$ ($16.26\%$ of words lost outside the $p=0.99$ top-$100$ nucleus).
    \item \textbf{{Frontier Proprietary ($3$ models):}} $0.8627$ ($\Delta = -0.0253$ for Human minus Frontier Proprietary, 95\% CI $[-0.0346,\,-0.0163]$), spanning GPT-6 Astra ($0.8255$, the only model source with lower reachability than human text, $\Delta = +0.0119$, 95\% CI $[+0.0018,\,+0.0218]$), Claude Opus 5.5 ($0.8728$), and Gemini 4 Argon ($0.8897$).
    \item \textbf{{Large Open-Weight ($7$ models, $427\text{{B}}$--$2.78\text{{T}}$):}} $0.8862$, exceeding Frontier Proprietary by $+0.0235$ (95\% CI $[+0.0210,\,+0.0261]$) and Human writing by $+0.0488$ (95\% CI $[+0.0395,\,+0.0581]$). Every individual Large Open-Weight generator—from DeepSeek-V4.1-Flash ($0.8648$) to Qwen3.8-2.4T-A95B ($0.9098$)—lies above the Frontier Proprietary mean and far below every Small Open-Weight source.
    \item \textbf{{Small Open-Weight ($5$ local models, $4\text{{B}}$--$27\text{{B}}$):}} $0.9602$, exceeding Large Open-Weight by $+0.0740$ (Large minus Small $\Delta = -0.0740$, 95\% CI $[-0.0772,\,-0.0708]$) and Human writing by $+0.1228$ (95\% CI $[+0.1133,\,+0.1326]$). Even excluding self-scoring diagonal cells ($s \neq g$), Small Open-Weight reachability remains $0.9496$.
\end{{itemize}}
Truncating every response to its first $150$ whitespace words (\texttt{{prefix150}}, Table~\ref{{tab:tiers}}) leaves all four tier means ($0.8267, 0.8556, 0.8776, 0.9579$) and pairwise contrasts ($\Delta = -0.0804$ Large vs.\ Small OW; $+0.0219$ Large vs.\ Frontier) virtually unchanged, ruling out response-length confounding.

\subsection{{Filter Sensitivity and Shipped Defaults}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\columnwidth]{{figures_nov2/fig4_filter_sensitivity.pdf}}
\caption{{Tier-level WCS across nucleus thresholds ($p \in \{{0.90, 0.95, 0.99\}}$) and filter definitions (\texttt{{strict\_full}} $p=0.99$ vs.\ shipped generation defaults) on HC3 ELI5. Tightening the filter widens the separation between Small Open-Weight models and all other tiers, while \texttt{{strict\_full}} collapses the Human vs.\ Frontier Proprietary gap.}}
\label{{fig:sensitivity}}
\end{{figure}}

\begin{{figure*}}[t]
\centering
\includegraphics[width=0.96\textwidth]{{figures_nov2/fig5_family_decomposed.pdf}}
\caption{{Tier-level reachability decomposed by scorer family across all five filter configurations on HC3 ELI5: Qwen \texttt{{bf16}} scorers (top left), Gemma \texttt{{bf16}} scorers (top middle), Llama \texttt{{bf16}} scorers (top right), Mistral \texttt{{bf16}} scorer (bottom left), and the high-VRAM \texttt{{Gemma-4-31B-it}} \texttt{{bf16}} scorer (bottom middle). Every scorer family and scale preserves the ordering $\text{{Frontier Prop.}} < \text{{Large Open-Weight}} < \text{{Small Open-Weight}}$ across all filter regimes.}}
\label{{fig:family_decomposed}}
\end{{figure*}}

\begin{{table*}}[t]
\centering
\small
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lccccccc}}
\toprule
Source Generator & Qwen-4B & Qwen-9B & Gemma-E4B & Gemma-12B & Qwen-27B (int8) & 4-bf16 Scorer Mean & Human $-$ Source $\Delta$ (95\% CI) \\
\midrule
{table5_rows}
\bottomrule
\end{{tabular}}%
}}
\caption{{HC3 ELI5 cross-model WCS matrix under stricter realistic nucleus filtering ($p=0.95, K=100$).}}
\label{{tab:eli5_p95}}
\end{{table*}}

\begin{{table*}}[t]
\centering
\small
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lccccccc}}
\toprule
Source Generator & Qwen-4B & Qwen-9B & Gemma-E4B & Gemma-12B & Qwen-27B (int8) & 4-bf16 Scorer Mean & Human $-$ Source $\Delta$ (95\% CI) \\
\midrule
{table6_rows}
\bottomrule
\end{{tabular}}%
}}
\caption{{HC3 ELI5 cross-model WCS matrix under exact full-vocabulary nucleus filtering (\texttt{{strict\_full}}, $p=0.99$). Removing the top-$100$ rank cap raises Human reachability from $0.8374$ to $0.8784$, matching Frontier Proprietary ($0.8730$).}}
\label{{tab:eli5_strict_full}}
\end{{table*}}

Figures~\ref{{fig:sensitivity}}--\ref{{fig:family_decomposed}} and Tables~\ref{{tab:eli5_p95}}--\ref{{tab:eli5_strict_full}} (together with Tables~\ref{{tab:eli5_shipped}}--\ref{{tab:eli5_unscaled}} in the Appendix) trace how the tier hierarchy responds to tightening or relaxing the candidate-pool filter:
\begin{{itemize}}
    \item \textbf{{Tighter nucleus thresholds ($p=0.95$ and $p=0.90$):}} Lowering $p$ from $0.99$ to $0.95$ (Table~\ref{{tab:eli5_p95}}) and $0.90$ (Table~\ref{{tab:tiers}}) steepens the separation between Small Open-Weight models and all other tiers. The Large minus Small Open-Weight contrast widens from $-0.0740$ ($p=0.99$) to $-0.1113$ ($p=0.95$, 95\% CI $[-0.1158,\,-0.1068]$) and $-0.1328$ ($p=0.90$, 95\% CI $[-0.1382,\,-0.1274]$).
    \item \textbf{{Shipped generation defaults and temperature scaling:}} Under each scorer's shipped defaults (Qwen $T=0.6, k=20, p=0.95$; Gemma $T=1.0, k=64, p=0.95$; Table~\ref{{tab:eli5_shipped}}), Human WCS drops to $0.6591$ and Frontier Proprietary to $0.7012$, while Small Open-Weight remains at $0.8831$ ($\Delta = -0.1472$ Large vs.\ Small OW, 95\% CI $[-0.1534,\,-0.1409]$). Under Qwen's model-card non-thinking defaults ($T=0.7, k=20, p=0.80$), the gap reaches $-0.1599$ (95\% CI $[-0.1664,\,-0.1533]$). Removing sub-unit temperature scaling ($T=1.0$ with shipped $k,p$; Table~\ref{{tab:eli5_unscaled}}) restores Human WCS to $0.7157$ while preserving a $-0.1195$ gap between Large and Small Open-Weight tiers.
\end{{itemize}}

\subsection{{Candidate-Support Validation ($K=100$ vs.\ $K=1000$ vs.\ Full Vocabulary)}}

\begin{{figure*}}[t]
\centering
\includegraphics[width=0.96\textwidth]{{figures_nov2/fig6_support_expansion.pdf}}
\caption{{Candidate-support expansion from $K=100$ to $K=1000$ to exact full vocabulary ($|V|$) at $p=0.99$ on HC3 ELI5 across Qwen-9B, Gemma-12B, Gemma-31B, Llama-3.1-8B, and Mistral-7B. Human writing exhibits the steepest upward trajectory from $K=100$ to $K=1000$ because human lexical choices frequently sit at token ranks $101$--$1000$ while remaining inside the $p=0.99$ full-vocabulary nucleus.}}
\label{{fig:support_expansion}}
\end{{figure*}}

\begin{{table*}}[t]
\centering
\small
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lccccccc}}
\toprule
Source / Tier & Tail Mass ($K=100$) & Tail Mass ($K=1000$) & Miss\% ($K=100$) & Miss\% ($K=1000$) & WCS ($K=100$) & WCS ($K=1000$) & WCS (\texttt{{strict\_full}} / \% Gap Closed) \\
\midrule
{table7_rows}
\bottomrule
\end{{tabular}}%
}}
\caption{{Candidate-support validation comparing $K=100$, $K=1000$, and exact full-vocabulary nucleus scoring (\texttt{{strict\_full}}) at $p=0.99$ on HC3 ELI5 across five scorers spanning four families (Qwen3.5-9B, Gemma-4-12B-it, Gemma-4-31B-it, Llama-3.1-8B-Instruct, and Mistral-7B-Instruct-v0.3). ``\% Gap Closed'' reports $(\text{{WCS}}_{{1000}} - \text{{WCS}}_{{100}}) / (\text{{WCS}}_{{\text{{full}}}} - \text{{WCS}}_{{100}})$.}}
\label{{tab:top1000}}
\end{{table*}}

Why does Human writing score lower than Frontier Proprietary text under realistic $K=100$ filtering ($0.8374$ vs.\ $0.8627$), yet slightly \emph{{higher}} under \texttt{{strict\_full}} ($0.8784$ vs.\ $0.8730$, Tables~\ref{{tab:tiers}} and~\ref{{tab:eli5_strict_full}})? Figure~\ref{{fig:support_expansion}} and Table~\ref{{tab:top1000}} resolve this mechanism directly. When scoring human text, a flat predictive distribution often spreads the $0.99$ probability nucleus across hundreds of plausible continuation tokens; a human word whose token ranks between $101$ and $1000$ has cumulative preceding mass $\le 0.99$ (so it survives \texttt{{strict\_full}}), but is deterministically clipped by the $K=100$ rank cap.

Expanding stored candidate support from $K=100$ to $K=1000$ reduces top-$K$ token misses on human writing from $4.77\%$ to $0.72\%$ on Qwen-9B (closing $82.9\%$ of the WCS gap to \texttt{{strict\_full}}: $0.9124 \to 0.9565$ vs.\ $0.9655$) and from $6.75\%$ to $1.56\%$ on Gemma-12B (closing $85.0\%$ of the gap: $0.8138 \to 0.8437$ vs.\ $0.8489$). {top1000_extra_text} By contrast, Small Open-Weight sources experience $<0.2\%$ top-$100$ misses on Qwen and Gemma because their greedy tokens almost always reside in the top-$20$ head.

\subsection{{Cross-Family Generalization and Same-Family Advantage}}

\begin{{figure}}[t]
\centering
\includegraphics[width=\columnwidth]{{figures_nov2/fig7_same_family.pdf}}
\caption{{Off-diagonal Same-Family Advantage (mean WCS on same-family open-weight pairs minus cross-family open-weight pairs, excluding self-scoring $s=g$) at $p=0.99$ with 95\% prompt-paired bootstrap confidence intervals, shown before (dark blue) and after (light blue) excluding family-specific orthographic confounds (curly apostrophes, standalone em-dashes, and \texttt{{alright}}).}}
\label{{fig:same_family}}
\end{{figure}}

\begin{{table*}}[t]
\centering
\small
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lccccccccccc}}
\toprule
Source Generator & Qwen-4B & Qwen-9B & Qwen-27B & Qwen-32B & Gemma-E4B & Gemma-12B & Llama-1B & Llama-8B & Mistral-7B & 7-bf16 Mean & 9-Scorer Mean \\
\midrule
{table8_rows}
\bottomrule
\end{{tabular}}%
}}
\caption{{Four-family expanded cross-model WCS matrix on HC3 ELI5 under realistic nucleus filtering ($p=0.99, K=100$) across $9$ scorers ($7$ \texttt{{bf16}} + $2$ \texttt{{int8}}) and $20$ sources.}}
\label{{tab:eli5_4family}}
\end{{table*}}

Table~\ref{{tab:eli5_4family}} expands the HC3 ELI5 audit to $9$ scorers and $20$ sources across four model families (Qwen, Gemma, Llama, and Mistral), and Table~\ref{{tab:tiers}} further extends this matrix to $10$ scorers ($8$ \texttt{{bf16}} + $2$ \texttt{{int8}}) and $21$ sources by adding \texttt{{Gemma-4-31B-it}}. Across all $9$ scorers (and across all $10$ scorers including \texttt{{Gemma-4-31B-it}}), the four-tier ordering holds without exception: Human ($0.8669$ 9-scorer; $0.8545$ 7-bf16; $0.8555$ 10-scorer; $0.8417$ 8-bf16) $<$ Frontier Proprietary ($0.8952$; $0.8847$; $0.8797$; $0.8666$) $<$ Large Open-Weight ($0.9124$; $0.9020$; $0.8969$; $0.8840$) $<$ Small Open-Weight ($0.9599$; $0.9555$; $0.9491$; $0.9428$), with Large minus Small Open-Weight $\Delta = -0.0476$ (95\% CI $[-0.0504,\,-0.0447]$; 10-scorer $\Delta = -0.0521$, 95\% CI $[-0.0552,\,-0.0490]$) and Large minus Frontier Proprietary $\Delta = +0.0172$ (95\% CI $[+0.0152,\,+0.0192]$; 10-scorer $\Delta = +0.0172$, 95\% CI $[+0.0152,\,+0.0194]$).

\paragraph{{Same-Family Advantage and Orthographic Sensitivity Check.}}
Excluding self-scoring diagonal cells ($s \neq g$), scorers exhibit a consistent \textbf{{Same-Family Advantage}} over cross-family open-weight sources (Figure~\ref{{fig:same_family}}):
\begin{{itemize}}
    \item Across Qwen and Gemma pairs on ELI5 ($n=18$ same-family vs.\ $36$ cross-family pairs, including \texttt{{qwen3.8-2.4t-a95b}} as a Qwen source), same-family pairs exceed cross-family pairs at $p=0.99$ by $\Delta = +0.0316$ (95\% CI $[+0.0294,\,+0.0343]$; rounded-cell $\Delta = +0.0328$), widening to $+0.0485$ ($[+0.0457,\,+0.0518]$) at $p=0.95$ and $+0.0552$ ($[+0.0519,\,+0.0589]$) at $p=0.90$.
    \item Across all four families on ELI5, the Same-Family Advantage at $p=0.99$ is $\Delta = +0.0269$ ($[+0.0251,\,+0.0291]$) on the 9-scorer suite and $\Delta = +0.0234$ (95\% CI $[+0.0214,\,+0.0257]$; rounded-cell $\Delta = +0.0233$) across the full 10-scorer suite ($n=21$ same-family vs.\ $69$ cross-family pairs, Figure~\ref{{fig:same_family}}), rising to $+0.0422$ ($[+0.0397,\,+0.0450]$) at $p=0.95$ and $+0.0526$ ($[+0.0497,\,+0.0556]$) at $p=0.90$.
    \item \textbf{{Orthographic confound exclusion:}} Because Appendix~\ref{{app:zipf}} identifies family-specific orthographic habits—such as Gemma 4's Unicode curly apostrophes (\texttt{{U+2019}} in \texttt{{it's}}, \texttt{{let's}}), Mistral-7B's conversational opener \texttt{{Alright,}}, and standalone em-dashes (\texttt{{---}})—we recomputed the Same-Family Advantage after excluding all words containing curly apostrophes, standalone punctuation/em-dashes, or \texttt{{alright}}. The Same-Family Advantage remains $+0.0313$ ($[+0.0290,\,+0.0340]$, $99.0\%$ preserved) for Qwen/Gemma, $+0.0265$ ($[+0.0246,\,+0.0287]$, $98.4\%$ preserved) on the 9-scorer 4-family suite, and $+0.0229$ ($[+0.0209,\,+0.0252]$, $98.1\%$ preserved) across all $10$ scorers at $p=0.99$, demonstrating that intra-family affinity reflects broad lexical alignment rather than punctuation artifacts.
\end{{itemize}}

\subsection{{Cross-Domain Replication on Wikipedia CS/AI}}

\begin{{table*}}[t]
\centering
\small
\resizebox{{\textwidth}}{{!}}{{%
{table9_header}%
}}
\caption{{Cross-domain replication on HC3 Wikipedia CS/AI ($N=50$ prompts) under realistic nucleus filtering ($p=0.99, K=100$).}}
\label{{tab:wiki}}
\end{{table*}}

Table~\ref{{tab:wiki}} and Table~\ref{{tab:tiers}} replicate our audit on technical encyclopedia prompts from HC3 Wikipedia CS/AI ($N=50$). Across the $4$ core \texttt{{bf16}} scorers and $15$ sources, the four-tier ordering replicates with slightly higher baseline reachability due to standardized technical terminology: Human ($0.8569$) $<$ Frontier Proprietary ($0.8835$; Human minus Frontier $\Delta = -0.0266$, 95\% CI $[-0.0325,\,-0.0205]$) $<$ Large Open-Weight ($0.9042$; Large minus Frontier $\Delta = +0.0207$, 95\% CI $[+0.0182,\,+0.0233]$) $<$ Small Open-Weight ($0.9626$; Large minus Small $\Delta = -0.0584$, 95\% CI $[-0.0616,\,-0.0552]$). Just as on ELI5, switching to exact full-vocabulary nucleus scoring (\texttt{{strict\_full}}, $p=0.99$) eliminates the gap between Human ($0.8868$) and Frontier Proprietary ($0.8897$; $\Delta = -0.0029$, 95\% CI $[-0.0084,\,+0.0024]$) while preserving the $-0.0549$ separation between Large and Small Open-Weight tiers. {wiki_ext_summary}

\subsection{{Decoding Regime Ablation: Greedy vs.\ Sampled ($T>0$) Local Sources}}

\begin{{table*}}[t]
\centering
\small
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lccccccc}}
\toprule
Source Generator & Qwen-4B & Qwen-9B & Gemma-E4B & Gemma-12B & Sampled Mean & Greedy Mean & Sampled $-$ Greedy $\Delta$ \\
\midrule
{table12_rows}
\bottomrule
\end{{tabular}}%
}}
\caption{{Stochastic-regeneration decoding ablation on HC3 ELI5 (realistic $p=0.99, K=100$, detokenized human reference). Local open-weight sources are regenerated using ancestral nucleus sampling ($T=1.0, p=0.95$) matched to the API sampling regime of Frontier Proprietary and Large Open-Weight models.}}
\label{{tab:sampled}}
\end{{table*}}

A natural concern is that in our primary baseline, local Small Open-Weight sources are decoded greedily whereas API-hosted Frontier Proprietary and Large Open-Weight sources use stochastic nucleus sampling ($T=1.0, p=0.95$). To isolate generator tier from decoding stochasticity, Table~\ref{{tab:sampled}} regenerates all four \texttt{{bf16}} local sources on HC3 ELI5 under matched stochastic sampling ($T=1.0, p=0.95$). Sampling reduces Small Open-Weight reachability by only $0.0040$ to $0.0156$ per model (tier mean $0.9513$ vs.\ $0.9602$ greedy), leaving a $-0.0652$ gap below Large Open-Weight ($0.8862$, 95\% CI $[-0.0687,\,-0.0617]$) and a $-0.0887$ gap below Frontier Proprietary ($0.8627$). Thus, $>88\%$ of the reachability gap between Small Open-Weight and larger tiers persists when all generators share the exact same $T=1.0, p=0.95$ sampling regime.

\section{{Discussion and High-VRAM Extension}}

\paragraph{{Why Does Reachability Track Generator Tier?}}
Our experiments separate two distinct mechanisms governing cross-model lexical reachability:
\begin{{enumerate}}
    \item \textbf{{Tail breadth vs.\ top-$K$ rank truncation (Human vs.\ Frontier):}} Human writers and Frontier Proprietary models (especially GPT-6 Astra) select contextually appropriate synonyms, metaphors, and domain terms that disperse probability mass across $>100$ subword candidates. These choices survive an exact $p=0.99$ full-vocabulary nucleus (\texttt{{strict\_full}}), where Human and Frontier Proprietary reachability coincide ($0.8784$ vs.\ $0.8730$ on ELI5; $0.8868$ vs.\ $0.8897$ on Wikipedia), but are disproportionately clipped by top-$100$ rank caps.
    \item \textbf{{Mode concentration in open-weight post-training (Small vs.\ Large Open-Weight):}} Small Open-Weight models ($1.2\text{{B}}$--$32.5\text{{B}}$) concentrate their generations inside a high-probability lexical core that every other open-weight family also assigns to its top-$20$ candidates—even under stochastic $T=1.0, p=0.95$ sampling and under \texttt{{strict\_full}}. Large Open-Weight MoE models ($427\text{{B}}$--$2.78\text{{T}}$) occupy a consistent intermediate tier between Frontier Proprietary and Small Open-Weight models across both domains and all nine filter regimes.
\end{{enumerate}}

\paragraph{{High-VRAM Unquantized Evaluation of \texttt{{Gemma-4-31B-it}}.}}
While \texttt{{Qwen3.5-27B}} and \texttt{{Qwen2.5-32B-Instruct}} are evaluated in 8-bit quantization on our primary $47.5\text{{ GiB}}$ GPU slice, evaluating \texttt{{google/gemma-4-31B-it}} ($31.3\text{{B}}$ parameters, revision \texttt{{842da379}}) in unquantized \texttt{{bf16}} requires $\sim 57.7\text{{ GiB}}$ for model weights plus KV-cache headroom ($\ge 62\text{{ GiB}}$ combined VRAM). In accordance with our reproducibility protocol—which prohibits silently substituting lower precision for a designated \texttt{{bf16}} scorer—we executed \texttt{{Gemma-4-31B-it}} generation, $K=100$ matrix scoring on HC3 ELI5 and Wikipedia CS/AI, and $K=1000$ validation on a dual-GPU $96\text{{ GiB}}$ server (\texttt{{scripts/run\_high\_vram\_server.sh}}, \texttt{{configs/high-vram-*.yaml}}). As shown in Table~\ref{{tab:tiers}}, Table~\ref{{tab:top1000}}, and Figures~\ref{{fig:family_decomposed}}--\ref{{fig:support_expansion}}, \texttt{{Gemma-4-31B-it}} preserves the four-tier hierarchy across every filter configuration ($0.8555 < 0.8797 < 0.8969 < 0.9491$ across the 10-scorer ELI5 suite; $0.8686 < 0.8955 < 0.9111 < 0.9530$ on Wikipedia CS/AI) and closes $79.9\%$ of the human $K=100 \to \texttt{{strict\_full}}$ gap at $K=1000$.

\section{{Conclusion}}
We presented a comprehensive cross-model audit of lexical reachability using the Word Coverage Score across $21$ generators, $10$ open-weight scorers spanning four model families, two HC3 domains, and nine candidate-pool filter regimes. By pairing standard top-$K$ nucleus audits with $K=1000$ support expansion, exact full-vocabulary \texttt{{strict\_full}} scoring, stochastic-regeneration ablations, and a Zipf-adjusted lexical loss taxonomy, we showed that top-$K$ rank truncation explains the apparent reachability deficit of human writing relative to frontier proprietary models, while a robust structural gap separates human and frontier prose from the highly concentrated vocabulary of small open-weight models.

\section*{{Limitations}}
Our teacher-forced WCS audit requires full logit and subword offset access, restricting the scorer role to open-weight checkpoints; proprietary frontier models can be audited as sources but not as scorers unless providers expose full-vocabulary log-probabilities. Second, our largest Qwen checkpoints (\texttt{{Qwen3.5-27B-int8}} and \texttt{{Qwen2.5-32B-int8}}) use 8-bit quantization to fit within a single $47.5\text{{ GiB}}$ GPU slice, whereas unquantized \texttt{{bf16}} evaluation at the $\sim 30\text{{B}}$ scale (\texttt{{Gemma-4-31B-it}}) requires $\ge 62\text{{ GiB}}$ of combined VRAM. Finally, our benchmarks focus on English explanatory and technical prose ($150$--$400$ words); extending WCS matrices to morphologically rich languages and creative writing is an important direction for future work.

\section*{{Ethical Considerations}}
All human reference texts are drawn from the publicly released, anonymized HC3 benchmark \citep{{guo2023hc3}}. API generations were collected via standard commercial endpoints without logging credentials or personal data. Because WCS characterizes aggregate distributional overlap rather than providing a calibrated document-level classifier, it should not be used for high-stakes individual authorship attribution or academic misconduct adjudication.

\bibliography{{references}}

\clearpage
\appendix

\section{{Additional Filter-Regime Matrices}}
\label{{app:matrices}}

Tables~\ref{{tab:eli5_shipped}} and~\ref{{tab:eli5_unscaled}} report the full per-source and per-scorer HC3 ELI5 matrices under each scorer's shipped generation defaults and under unscaled ($T=1.0$) shipped top-$k$/top-$p$ filters.

\begin{{table*}}[h!]
\centering
\small
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lccccccc}}
\toprule
Source Generator & Qwen-4B & Qwen-9B & Gemma-E4B & Gemma-12B & Qwen-27B (int8) & 4-bf16 Scorer Mean & Human $-$ Source $\Delta$ (95\% CI) \\
\midrule
{table10_rows}
\bottomrule
\end{{tabular}}%
}}
\caption{{HC3 ELI5 cross-model WCS matrix under each scorer's shipped generation defaults (Qwen 3.5: $T=0.6, k=20, p=0.95$; Gemma 4: $T=1.0, k=64, p=0.95$).}}
\label{{tab:eli5_shipped}}
\end{{table*}}

\begin{{table*}}[h!]
\centering
\small
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lccccccc}}
\toprule
Source Generator & Qwen-4B & Qwen-9B & Gemma-E4B & Gemma-12B & Qwen-27B (int8) & 4-bf16 Scorer Mean & Human $-$ Source $\Delta$ (95\% CI) \\
\midrule
{table11_rows}
\bottomrule
\end{{tabular}}%
}}
\caption{{HC3 ELI5 cross-model WCS matrix under unscaled ($T=1.0$) shipped top-$k$ and top-$p$ filters (Qwen 3.5: $T=1.0, k=20, p=0.95$; Gemma 4: $T=1.0, k=64, p=0.95$).}}
\label{{tab:eli5_unscaled}}
\end{{table*}}

\section{{Zipf-Adjusted Lexical Category and Loss-Rate Taxonomy}}
\label{{app:zipf}}

To understand \emph{{which}} words fail the nucleus filter across generator tiers—and why raw counts of lost words can be misleading without frequency normalization—we conducted a word-level lexical audit across the expanded 4-family HC3 ELI5 evaluation suite ($9$ scorers evaluating $20$ sources at realistic $p=0.99$, excluding self-scoring diagonal pairs $s = g$, totaling $2{{,}}586{{,}}369$ off-diagonal word evaluations and $187{{,}}895$ lost word instances).

\paragraph{{Methodology \& Word-Class Partition.}}
For every off-diagonal scorer--source pair $(g, s)$ with $g \neq s$ and every prompt $q \in \{{1, \dots, 50\}}$, we align the whitespace word groups $W(y_{{q,s}})$ with the token-level survival mask under realistic $p=0.99$ ($K=100$). Each word instance $w$ is normalized by lowercasing and stripping leading/trailing ASCII punctuation (while preserving internal apostrophes and hyphens, and distinguishing ASCII straight apostrophes \texttt{{'}} from Unicode right single quotation marks \texttt{{U+2019}}) and assigned to one of three mutually exclusive macro-categories:
\begin{{enumerate}}
    \item \textbf{{Function / Stopwords:}} Standard closed-class English determiners, prepositions, conjunctions, auxiliary verbs, and pronouns ($85$ canonical forms).
    \item \textbf{{Standalone Punctuation / Symbols:}} Tokens consisting solely of non-alphanumeric characters bounded by whitespace (e.g., spaced em-dashes \texttt{{---}}, bullets, or markdown symbols \texttt{{\#}}, \texttt{{>}}).
    \item \textbf{{Content / Lexical Words:}} All remaining open-class nouns, lexical verbs, adjectives, adverbs, numbers, and hyphenated/contracted lexical forms.
\end{{enumerate}}
For each generator tier and each normalized word type $v$, we record total off-diagonal evaluations $N(v)$, total lost instances $L(v)$, and the \textbf{{Zipf-adjusted per-occurrence loss rate}} $r(v) = L(v)/N(v)$.

\begin{{table*}}[t]
\centering
\small
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llrrrr}}
\toprule
Generator Tier & Word Macro-Category & Lost Words & Total Evaluated & Category Loss Rate & Share of Tier's Lost Words \\
\midrule
{app_table1_rows}
\bottomrule
\end{{tabular}}%
}}
\caption{{Macro-category breakdown of word losses at realistic $p=0.99$ across the 4-family HC3 ELI5 suite ($9$ scorers, $20$ sources, off-diagonal pairs $s \neq g$, $2{{,}}586{{,}}369$ word evaluations).}}
\label{{tab:zipf_macro}}
\end{{table*}}

\begin{{table*}}[t]
\centering
\small
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lll}}
\toprule
Generator Tier & Top Lost Words by Raw Count (Lost / Total, Loss Rate) & Top Lost Words by Loss Rate ($\ge 45$ Evals: Rate, Lost / Total) \\
\midrule
Human (HC3) &
\begin{{tabular}}[t]{{@{{}}l@{{}}}}
\texttt{{the}} ($139/4{{,}}725$, $2.9\%$), \texttt{{a}} ($109/2{{,}}520$, $4.3\%$), \texttt{{and}} ($106/2{{,}}025$, $5.2\%$),\\
\texttt{{in}} ($101/1{{,}}458$, $6.9\%$), \texttt{{that}} ($74/1{{,}}431$, $5.2\%$), \texttt{{of}} ($72/2{{,}}466$, $2.9\%$),\\
\texttt{{i}} ($72/333$, $21.6\%$), \texttt{{this}} ($68/648$, $10.5\%$), \texttt{{from}} ($63/504$, $12.5\%$)
\end{{tabular}} &
\begin{{tabular}}[t]{{@{{}}l@{{}}}}
\texttt{{tl;dr}} ($100.0\%$, $45/45$), \texttt{{edit}} ($77.8\%$, $42/54$), \texttt{{depending}} ($57.8\%$, $26/45$),\\
\texttt{{whether}} ($48.9\%$, $22/45$), \texttt{{set}} ($48.9\%$, $22/45$), \texttt{{volume}} ($48.9\%$, $22/45$),\\
\texttt{{due}} ($48.9\%$, $22/45$), \texttt{{everybody}} ($48.9\%$, $22/45$), \texttt{{generally}} ($46.7\%$, $21/45$)
\end{{tabular}} \\
\midrule
Frontier Prop. &
\begin{{tabular}}[t]{{@{{}}l@{{}}}}
\texttt{{the}} ($810/20{{,}}340$, $4.0\%$), \texttt{{a}} ($720/15{{,}}174$, $4.7\%$), \texttt{{and}} ($655/10{{,}}890$, $6.0\%$),\\
\texttt{{that}} ($411/4{{,}}833$, $8.5\%$), \texttt{{your}} ($376/4{{,}}734$, $7.9\%$), \texttt{{it}} ($306/7{{,}}002$, $4.4\%$),\\
\texttt{{in}} ($292/4{{,}}194$, $7.0\%$), \texttt{{you}} ($291/6{{,}}462$, $4.5\%$), \texttt{{is}} ($282/6{{,}}588$, $4.3\%$)
\end{{tabular}} &
\begin{{tabular}}[t]{{@{{}}l@{{}}}}
\texttt{{acts}} ($68.9\%$, $31/45$), \texttt{{ordinary}} ($68.7\%$, $68/99$), \texttt{{surprise}} ($57.4\%$, $31/54$),\\
\texttt{{roughly}} ($56.5\%$, $61/108$), \texttt{{here's}} ($54.0\%$, $68/126$), \texttt{{recognize}} ($53.3\%$, $24/45$),\\
\texttt{{neither}} ($51.9\%$, $28/54$), \texttt{{electronic}} ($51.1\%$, $23/45$), \texttt{{plenty}} ($51.1\%$, $23/45$)
\end{{tabular}} \\
\midrule
Large Open-Wt. &
\begin{{tabular}}[t]{{@{{}}l@{{}}}}
\texttt{{---}} ($3{{,}}190/7{{,}}092$, $45.0\%$), \texttt{{the}} ($1{{,}}804/43{{,}}128$, $4.2\%$), \texttt{{a}} ($1{{,}}276/31{{,}}005$, $4.1\%$),\\
\texttt{{and}} ($1{{,}}121/24{{,}}048$, $4.7\%$), \texttt{{your}} ($711/9{{,}}936$, $7.2\%$), \texttt{{so}} ($710/8{{,}}154$, $8.7\%$),\\
\texttt{{that}} ($704/10{{,}}872$, $6.5\%$), \texttt{{here's}} ($668/1{{,}}521$, $43.9\%$), \texttt{{it}} ($572/16{{,}}101$, $3.6\%$)
\end{{tabular}} &
\begin{{tabular}}[t]{{@{{}}l@{{}}}}
\texttt{{\#}} ($100.0\%$, $99/99$), \texttt{{eli5}} ($75.6\%$, $34/45$), \texttt{{haitian}} ($62.2\%$, $28/45$),\\
\texttt{{honestly}} ($59.7\%$, $43/72$), \texttt{{deliberately}} ($55.6\%$, $45/81$), \texttt{{whose}} ($54.2\%$, $39/72$),\\
\texttt{{hanging}} ($53.3\%$, $24/45$), \texttt{{borrowed}} ($51.9\%$, $28/54$), \texttt{{great}} ($50.9\%$, $357/702$)
\end{{tabular}} \\
\midrule
Small Open-Wt. &
\begin{{tabular}}[t]{{@{{}}l@{{}}}}
\texttt{{the}} ($997/56{{,}}080$, $1.8\%$), \texttt{{and}} ($710/25{{,}}384$, $2.8\%$), \texttt{{is}} ($640/21{{,}}216$, $3.0\%$),\\
\texttt{{a}} ($591/40{{,}}272$, $1.5\%$), \texttt{{you}} ($484/22{{,}}416$, $2.2\%$), \texttt{{it}} ($480/20{{,}}624$, $2.3\%$),\\
\texttt{{to}} ($444/30{{,}}904$, $1.4\%$), \texttt{{so}} ($442/6{{,}}528$, $6.8\%$), \texttt{{imagine}} ($375/3{{,}}680$, $10.2\%$)
\end{{tabular}} &
\begin{{tabular}}[t]{{@{{}}l@{{}}}}
\texttt{{alright}} ($88.1\%$, $155/176$), \texttt{{let's}} (curly, $56.3\%$, $27/48$), \texttt{{removes}} ($39.6\%$, $19/48$),\\
\texttt{{poland}} ($35.4\%$, $17/48$), \texttt{{--}} ($32.3\%$, $31/96$), \texttt{{it's}} (curly, $31.3\%$, $223/712$),\\
\texttt{{asked}} ($29.2\%$, $35/120$), \texttt{{immediately}} ($28.6\%$, $32/112$), \texttt{{grown-ups}} ($26.6\%$, $34/128$)
\end{{tabular}} \\
\bottomrule
\end{{tabular}}%
}}
\caption{{Comparison of top lost words ranked by raw failure count (left) versus Zipf-adjusted per-occurrence failure rate among words with $\ge 45$ evaluations (right) at realistic $p=0.99$ across the 9-scorer, 20-source 4-family ELI5 pool.}}
\label{{tab:zipf_words}}
\end{{table*}}

\paragraph{{Finding 1: Macro-Category Breakdown.}}
Table~\ref{{tab:zipf_macro}} demonstrates that across all four generator tiers, \textbf{{content/lexical words account for $71.7\%$--$77.4\%$ of all lost words}} and suffer a per-occurrence loss rate $2.1\times$ to $2.9\times$ higher than function/stopwords:
\begin{{itemize}}
    \item In \textbf{{Human}} writing, $17.86\%$ of all content word tokens fail the $p=0.99$ filter ($8{{,}}992 / 50{{,}}355$), compared to only $6.25\%$ of function words ($2{{,}}461 / 39{{,}}384$).
    \item Across \textbf{{Frontier Proprietary}} ($12.99\%$ content vs.\ $5.88\%$ function), \textbf{{Large Open-Weight}} ($10.66\%$ content vs.\ $4.94\%$ function), and \textbf{{Small Open-Weight}} ($5.64\%$ content vs.\ $2.72\%$ function), the hierarchy of lexical reachability is driven primarily by a progressive $3.2\times$ contraction in content-word loss rates ($17.86\% \to 12.99\% \to 10.66\% \to 5.64\%$).
\end{{itemize}}

\paragraph{{Finding 2: Raw-Count Leaders vs.\ Zipf-Adjusted Rate Leaders.}}
Table~\ref{{tab:zipf_words}} contrasts the top lost words ranked by raw count against those ranked by per-occurrence loss rate ($\ge 45$ evaluations). Because function words like \texttt{{the}}, \texttt{{a}}, and \texttt{{and}} occur $20{{,}}000$--$56{{,}}000$ times per tier, even a $1.5\%$--$6.0\%$ failure rate produces hundreds to thousands of raw losses. Normalizing by occurrence frequency reveals the true high-risk lexical items: Reddit register markers (\texttt{{tl;dr}} $100\%$, \texttt{{edit}} $77.8\%$) and hedging words (\texttt{{depending}} $57.8\%$, \texttt{{whether}} $48.9\%$, \texttt{{generally}} $46.7\%$) in Human answers; nuanced verbs and qualifiers (\texttt{{acts}} $68.9\%$, \texttt{{ordinary}} $68.7\%$, \texttt{{roughly}} $56.5\%$) in Frontier Proprietary answers; markdown headers (\texttt{{\#}} $100\%$), conversational preambles (\texttt{{eli5}} $75.6\%$, \texttt{{honestly}} $59.7\%$, \texttt{{great}} $50.9\%$), and spaced em-dashes (\texttt{{---}} $45.0\%$) in Large Open-Weight answers; and family-specific openers/orthography (\texttt{{alright}} $88.1\%$, curly-apostrophe \texttt{{let's}} $56.3\%$ and \texttt{{it's}} $31.3\%$) in Small Open-Weight answers.

\paragraph{{Finding 3: Why Function Words Still Fail Occasionally.}}
Inspecting the $6{{,}}446$ total lost occurrences of \texttt{{the}} and \texttt{{a}} across the 9-scorer matrix reveals three concrete mechanisms: (i) \textbf{{contextual phrase/idiom lock-in}} ($39.0\%$, $2{{,}}513/6{{,}}446$), where the scorer assigns $\ge 0.95$ probability to a single fixed continuation token (e.g., an idiom or multi-word expression), pushing any alternative determiner past the $0.99$ boundary; (ii) \textbf{{sentence-initial discourse framing}} ($15.8\%$, $1{{,}}016/6{{,}}446$), where capitalized \texttt{{The}} or \texttt{{A}} competes against preferred bullet/discourse starters; and (iii) \textbf{{mid-sentence determiner competition}} ($45.3\%$, $2{{,}}917/6{{,}}446$), where a scorer strongly prefers a bare plural or possessive pronoun over an article.

\end{{document}}
"""
    overleaf_dir = Path("overleaf")
    (overleaf_dir / "main.tex").write_text(tex)
    print("Wrote updated ACL LaTeX manuscript to overleaf/main.tex")

    if shutil.which("pdflatex") and shutil.which("bibtex"):
        for ext in ("aux", "bbl", "blg", "log", "out"):
            p = overleaf_dir / f"main.{ext}"
            if p.exists():
                p.unlink()
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "main.tex"],
            cwd=overleaf_dir,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(["bibtex", "main"], cwd=overleaf_dir, check=True, stdout=subprocess.DEVNULL)
        for _ in range(3):
            subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", "main.tex"],
                cwd=overleaf_dir,
                check=True,
                stdout=subprocess.DEVNULL,
            )
        shutil.copy2(overleaf_dir / "main.pdf", Path("WCS_revised_2_Nov_updated.pdf"))
        print("Compiled overleaf/main.pdf and copied to WCS_revised_2_Nov_updated.pdf")

    zip_path = Path("WCS_revised_2_Nov_updated.zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname in ("main.tex", "references.bib", "acl.sty", "acl_natbib.bst", "main.pdf"):
            fpath = overleaf_dir / fname
            if fpath.exists():
                zf.write(fpath, arcname=fname)
        fig_dir = overleaf_dir / "figures_nov2"
        if fig_dir.exists():
            for fig_file in sorted(fig_dir.iterdir()):
                if fig_file.is_file():
                    zf.write(fig_file, arcname=f"figures_nov2/{fig_file.name}")
    print(f"Created Overleaf archive {zip_path} ({zip_path.stat().st_size:,} bytes)")
    for extra_zip in (Path("overleaf_project.zip"), overleaf_dir / "overleaf_project.zip"):
        shutil.copy2(zip_path, extra_zip)


if __name__ == "__main__":
    main()
