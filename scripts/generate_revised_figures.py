#!/usr/bin/env python3
"""
Generate all 7 publication-grade figures (PDF + PNG) for WCS_revised_2_Nov_updated:
- fig1_pipeline.{pdf,png}: Architecture/evaluation pipeline schematic (matching WCS_revised_2_Nov Fig 1)
- fig2_tier_hierarchy.{pdf,png}: Horizontal dot-plot of all 16 core ELI5 sources + tier bands and means
- fig3_filter_families.{pdf,png}: Candidate-pool filter families block diagram (matching WCS_revised_2_Nov Fig 3)
- fig4_filter_sensitivity.{pdf,png}: Tier-level WCS across filter configurations (matching WCS_revised_2_Nov Fig 4)
- fig5_family_decomposed.{pdf,png}: 5-panel family-decomposed tier trajectories (Qwen, Gemma, Llama, Mistral, Gemma-31B)
- fig6_support_expansion.{pdf,png}: 5-panel K=100 -> K=1000 -> Full |V| trajectories (Qwen-9B, Gemma-12B, Llama-8B, Mistral-7B, Gemma-31B)
- fig7_same_family.{pdf,png}: Same-family minus cross-family WCS advantage (All words vs. Orthographic-clean)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

OUT_DIR = Path("overleaf/figures_nov2")

TIER_COLORS = {
    "human": "#2b2b2b",
    "frontier_prop": "#c44e52",
    "large_ow": "#4c72b0",
    "small_ow": "#55a868",
}


def save_both(fig: plt.Figure, stem: str, tight: bool = False):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.savefig(OUT_DIR / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
        fig.savefig(OUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    else:
        fig.savefig(OUT_DIR / f"{stem}.pdf")
        fig.savefig(OUT_DIR / f"{stem}.png", dpi=300)
    plt.close(fig)


def _pt_box(
    ax,
    x0: float,
    y0: float,
    w: float,
    h: float,
    text: str,
    fc: str,
    ec: str,
    tc: str = "#1a1a1a",
    fontsize: float = 6.9,
    weight: str = "normal",
    lw: float = 1.25,
    ls: str = "-",
    pad: float = 2.0,
):
    box = mpatches.FancyBboxPatch(
        (x0 + pad, y0 + pad),
        w - 2 * pad,
        h - 2 * pad,
        boxstyle=f"round,pad={pad},rounding_size=3.2",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
    )
    ax.add_patch(box)
    ax.text(
        x0 + w / 2.0,
        y0 + h / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=tc,
        fontweight=weight,
        linespacing=1.18,
    )


def draw_fig1_pipeline(data: dict):
    # Exact point-coordinate geometry from WCS_revised_2_Nov.pdf (507.6 x 172.8 pt = 7.05 x 2.40 in)
    fig = plt.figure(figsize=(7.05, 2.40))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 507.6)
    ax.set_ylim(0, 172.8)
    ax.axis("off")

    ax.text(
        253.8,
        158.5,
        "Evaluation pipeline: every source response is scored by every scorer with the Word Coverage Score (WCS)",
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color="#111111",
    )

    # Column headers
    _pt_box(ax, 11.1, 128.0, 140.7, 16.5, "Generators (sources)", "#33475b", "#33475b", tc="white", fontsize=8.0, weight="bold", lw=0)
    _pt_box(ax, 204.2, 128.0, 140.7, 16.5, "Scorers (WCS)", "#33475b", "#33475b", tc="white", fontsize=8.0, weight="bold", lw=0)
    _pt_box(ax, 376.5, 128.0, 119.9, 16.5, "Outputs", "#33475b", "#33475b", tc="white", fontsize=8.0, weight="bold", lw=0)

    # Column 1: 5 source boxes
    _pt_box(ax, 11.1, 104.2, 140.7, 21.0, "Human writing\n(HC3, detokenized)", "#f0f0f0", "#444444", fontsize=6.8)
    _pt_box(ax, 11.1, 81.2, 140.7, 21.0, "Frontier Proprietary\n3 models (GPT-6, Claude 5.5, Gemini 4)", "#f9ecec", "#c44e52", fontsize=6.5)
    _pt_box(ax, 11.1, 58.2, 140.7, 21.0, "Large Open-Weight\n7 models (427B–2.78T MoE)", "#edf2f8", "#4c72b0", fontsize=6.7)
    _pt_box(ax, 11.1, 35.2, 140.7, 21.0, "Small Open-Weight (Core)\n5 models (Qwen 3.5, Gemma 4)", "#eef6f0", "#55a868", fontsize=6.7)
    _pt_box(ax, 11.1, 12.2, 140.7, 21.0, "4-Family & High-VRAM Suite\nLlama 3.1/3.2, Mistral 7B, Gemma-31B", "#fbf6e9", "#b98a2f", fontsize=6.5)

    # Connector from Column 1 to Column 2
    y_mids = [114.7, 91.7, 68.7, 45.7, 22.7]
    for ym in y_mids:
        ax.plot([152.2, 160.0], [ym, ym], color="#7a8a99", lw=1.15)
    ax.plot([160.0, 160.0], [min(y_mids), max(y_mids)], color="#7a8a99", lw=1.15)
    ax.annotate(
        "",
        xy=(203.2, 74.0),
        xytext=(160.0, 74.0),
        arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.28", color="#4d4d4d", lw=1.4),
    )
    ax.text(181.8, 90.0, "50 prompts\nper domain", ha="center", va="center", fontsize=6.0, color="#333333", linespacing=1.15)
    ax.text(181.8, 56.0, "responses\n150–400\nwords", ha="center", va="center", fontsize=6.0, color="#333333", linespacing=1.15)

    # Column 2: Scorers
    _pt_box(
        ax,
        204.2,
        104.2,
        140.7,
        21.0,
        "5 core open-weight scorers\nQwen 3.5 (4B–27B), Gemma 4 (E4B, 12B)",
        "#eaf1f8",
        "#3a6ea5",
        fontsize=6.4,
    )
    _pt_box(
        ax,
        204.2,
        81.2,
        140.7,
        21.0,
        "5 expanded scorers (4 families)\nLlama (1B, 8B), Mistral-7B, Gemma-31B",
        "#fbf6e9",
        "#b98a2f",
        fontsize=6.4,
    )
    _pt_box(
        ax,
        204.2,
        16.0,
        140.7,
        62.0,
        "teacher-forced pass\nper position: candidate-pool\nfilter (Figure 3)\nword covered iff every\nsub-token survives the filter\nWCS = mean word coverage",
        "#eaf1f8",
        "#3a6ea5",
        fontsize=6.4,
    )

    # Connector from Column 2 to Column 3
    ax.annotate(
        "",
        xy=(375.5, 74.0),
        xytext=(345.8, 74.0),
        arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.28", color="#4d4d4d", lw=1.4),
    )

    # Column 3: Outputs
    _pt_box(ax, 376.5, 96.5, 119.9, 28.0, "scorer-by-source\nWCS matrix\n(Figure 2, Table 4)", "#f4f6f8", "#5b6b7c", fontsize=6.6)
    _pt_box(ax, 376.5, 62.5, 119.9, 28.0, "tier means and contrasts\n(Table 3)", "#f4f6f8", "#5b6b7c", fontsize=6.6)
    _pt_box(ax, 376.5, 28.5, 119.9, 28.0, "filter-family and\ncandidate-support audits\n(Figures 3–7)", "#f4f6f8", "#5b6b7c", fontsize=6.6)

    save_both(fig, "fig1_pipeline", tight=False)


def draw_fig2_tier_hierarchy(data: dict):
    # Matches WCS_revised_2_Nov.pdf Figure 2 (figsize=(6.00, 3.95))
    src_deltas = data["eli5_core_source_deltas_4sc"]["realistic_0.99"]
    items = [
        ("Human (HC3 ELI5)", src_deltas["human"]["source_mean_4sc"], "human"),
        ("GPT-6 Astra", src_deltas["gpt-6-astra"]["source_mean_4sc"], "frontier_prop"),
        ("Claude Opus 5.5", src_deltas["claude-opus-5.5"]["source_mean_4sc"], "frontier_prop"),
        ("Gemini 4 Argon", src_deltas["gemini-4-argon"]["source_mean_4sc"], "frontier_prop"),
        ("DeepSeek-V4.1-Flash", src_deltas["deepseek-v4.1-flash-1017"]["source_mean_4sc"], "large_ow"),
        ("GLM-5.3", src_deltas["glm-5.3"]["source_mean_4sc"], "large_ow"),
        ("MiniMax-M3", src_deltas["minimax-m3"]["source_mean_4sc"], "large_ow"),
        ("Kimi-K3", src_deltas["kimi-k3"]["source_mean_4sc"], "large_ow"),
        ("MiMo-V2.6-Pro", src_deltas["mimo-v2.6-pro"]["source_mean_4sc"], "large_ow"),
        ("DeepSeek-V4-Pro", src_deltas["deepseek-v4-pro-0813"]["source_mean_4sc"], "large_ow"),
        ("Qwen3.8-2.4T-A95B", src_deltas["qwen3.8-2.4t-a95b"]["source_mean_4sc"], "large_ow"),
        ("Qwen-4B", src_deltas["qwen-4b"]["source_mean_4sc"], "small_ow"),
        ("Qwen-9B", src_deltas["qwen-9b"]["source_mean_4sc"], "small_ow"),
        ("Qwen-27B (int8)", src_deltas["qwen-27b-int8"]["source_mean_4sc"], "small_ow"),
        ("Gemma-12B", src_deltas["gemma-12b"]["source_mean_4sc"], "small_ow"),
        ("Gemma-E4B", src_deltas["gemma-e4b"]["source_mean_4sc"], "small_ow"),
    ]

    y_pos = [
        17.5,  # Human
        15.8, 14.8, 13.8,  # Frontier Prop
        12.1, 11.1, 10.1, 9.1, 8.1, 7.1, 6.1,  # Large OW
        4.4, 3.4, 2.4, 1.4, 0.4,  # Small OW
    ]

    fig = plt.figure(figsize=(6.00, 3.95))
    ax = fig.add_axes([0.274, 0.139, 0.680, 0.755])

    ax.axhspan(16.95, 18.05, color="#ebebeb", alpha=0.85, zorder=0)
    ax.axhspan(13.25, 16.35, color="#f9e6e6", alpha=0.75, zorder=0)
    ax.axhspan(5.55, 12.65, color="#e6eef8", alpha=0.75, zorder=0)
    ax.axhspan(-0.15, 4.95, color="#e6f4ea", alpha=0.75, zorder=0)

    ax.grid(axis="x", linestyle=":", color="#b0b0b0", alpha=0.7, zorder=1)

    human_val = items[0][1]
    ax.axvline(human_val, color="#2b2b2b", linestyle="--", linewidth=1.1, zorder=2)
    ax.text(
        human_val + 0.002,
        17.65,
        f"Human writing ({human_val:.4f})",
        fontsize=8.2,
        color="#2b2b2b",
        va="center",
    )

    tier_means = data["eli5_core_4sc_tiers"]["realistic_0.99"]["tier_means"]
    tier_spans = [
        ("frontier_prop", 13.25, 16.35, 16.46, "left"),
        ("large_ow", 5.55, 12.65, 12.76, "left"),
        ("small_ow", -0.15, 4.95, 5.06, "right"),
    ]
    for t_key, y0, y1, y_label, align in tier_spans:
        tm = tier_means[t_key]
        ax.plot([tm, tm], [y0, y1], color=TIER_COLORS[t_key], linewidth=1.5, zorder=3)
        x_txt = tm + 0.0025 if align == "left" else tm - 0.0025
        ax.text(
            x_txt,
            y_label,
            f"tier mean {tm:.4f}",
            fontsize=7.8,
            color=TIER_COLORS[t_key],
            style="italic",
            ha=align,
            va="bottom",
        )

    for (label, val, tier), y in zip(items, y_pos):
        ax.scatter([val], [y], color=TIER_COLORS[tier], s=38, zorder=4)
        if tier != "human":
            ax.text(val + 0.0025, y, f"{val:.4f}", fontsize=7.8, color="#222222", va="center")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([it[0] for it in items], fontsize=9.0)
    ax.set_xlim(0.79, 1.005)
    ax.set_ylim(-0.4, 18.4)
    ax.set_xticks(np.arange(0.80, 1.01, 0.02))
    ax.tick_params(axis="x", labelsize=8.8)
    ax.set_xlabel("Mean WCS (4 bf16 scorers)", fontsize=9.5)
    ax.set_title("Lexical reachability by source tier and generator", fontsize=10.8, fontweight="bold", pad=10)

    save_both(fig, "fig2_tier_hierarchy", tight=False)


def draw_fig3_filter_families(data: dict):
    # Exact point-coordinate geometry from WCS_revised_2_Nov.pdf Figure 3 (507.6 x 244.8 pt = 7.05 x 3.40 in)
    fig = plt.figure(figsize=(7.05, 3.40))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 507.6)
    ax.set_ylim(0, 244.8)
    ax.axis("off")

    ax.text(
        253.8,
        231.0,
        "Candidate-pool filters evaluated in the audit (block diagram: each block is one stage of the filter)",
        ha="center",
        va="center",
        fontsize=8.6,
        fontweight="bold",
        color="#111111",
    )

    rows = [
        (
            "A",
            175.0,
            "Realistic top-p\nover top-K\n(main configuration)",
            [
                (121.0, 68.0, "top-K window\nK = 100", "#ebf2f8", "#4c72b0"),
                (199.5, 74.0, "renormalize\nwithin window", "#eef6f0", "#55a868"),
                (284.0, 82.0, "nucleus until\ncumulative mass = p\np = 0.90 / 0.95 / 0.99", "#f2ecf8", "#7c5cad"),
            ],
            "validation:\nK = 1000 and exact\nfull vocabulary",
        ),
        (
            "B",
            127.0,
            "Strict unnormalized\ntop-K",
            [
                (121.0, 68.0, "top-K window\nK = 100", "#ebf2f8", "#4c72b0"),
                (199.5, 74.0, "no\nrenormalization", "#f5f5f5", "#8a8a8a"),
                (284.0, 82.0, "nucleus over\nraw mass\np = 0.99", "#f2ecf8", "#7c5cad"),
            ],
            "",
        ),
        (
            "C",
            79.0,
            "Exact full-vocabulary\ntop-p (strict_full)",
            [
                (121.0, 68.0, "full\nvocabulary |V|", "#ebf2f8", "#4c72b0"),
                (199.5, 82.0, "nucleus over full\nvocabulary\np = 0.99", "#f2ecf8", "#7c5cad"),
            ],
            "",
        ),
        (
            "D",
            31.0,
            "Shipped generation\ndefaults (scorer\ndefault filters)",
            [
                (121.0, 53.5, "temperature\nscaling T", "#fcf0e6", "#dd8452"),
                (184.0, 52.0, "top-k\nwindow", "#ebf2f8", "#4c72b0"),
                (245.5, 53.5, "renormalize", "#eef6f0", "#55a868"),
                (308.5, 57.5, "nucleus p", "#f2ecf8", "#7c5cad"),
            ],
            "Qwen 3.5: T = 0.6, k = 20, p = 0.95\nGemma 4: T = 1.0, k = 64, p = 0.95\nLlama: T = 0.6, k = 100, p = 0.90\nMistral: T = 1.0, k = 100, p = 0.95",
        ),
    ]

    for letter, yc, title, blocks, note in rows:
        _pt_box(ax, 11.1, yc - 9.0, 24.5, 18.0, letter, "#33475b", "#33475b", tc="white", fontsize=8.4, weight="bold", lw=0, pad=1.5)
        ax.text(39.0, yc, title, ha="left", va="center", fontsize=6.8, color="#1a1a1a", linespacing=1.16)

        prev_right = None
        for bx, bw, btxt, fc, ec in blocks:
            _pt_box(ax, bx, yc - 17.5, bw, 35.0, btxt, fc, ec, fontsize=6.8, lw=1.25, pad=1.8)
            if prev_right is not None:
                ax.annotate(
                    "",
                    xy=(bx - 0.8, yc),
                    xytext=(prev_right + 0.8, yc),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.22", color="#4d4d4d", lw=1.25),
                )
            prev_right = bx + bw

        if note:
            ax.text(373.5, yc, note, ha="left", va="center", fontsize=6.2, style="italic", color="#333333", linespacing=1.18)

    save_both(fig, "fig3_filter_families", tight=False)


def draw_fig4_filter_sensitivity(data: dict):
    # Matches WCS_revised_2_Nov.pdf Figure 4 (figsize=(3.55, 2.75)):
    # Left panel: Realistic p=0.90, p=0.95, p=0.99
    # Right panel: Strict full p=0.99, Shipped default
    fig = plt.figure(figsize=(3.55, 2.75))
    ax1 = fig.add_axes([0.165, 0.285, 0.435, 0.645])
    ax2 = fig.add_axes([0.645, 0.285, 0.325, 0.645])

    tiers_4sc = data["eli5_core_4sc_tiers"]
    series_meta = [
        ("human", "Human", TIER_COLORS["human"], "o"),
        ("frontier_prop", "Frontier Prop.", TIER_COLORS["frontier_prop"], "s"),
        ("large_ow", "Large Open-Wt.", TIER_COLORS["large_ow"], "^"),
        ("small_ow", "Small Open-Wt.", TIER_COLORS["small_ow"], "D"),
    ]

    conds_left = ["realistic_0.9", "realistic_0.95", "realistic_0.99"]
    conds_right = ["strict_full_0.99", "default_shipped"]

    for t_key, label, col, marker in series_meta:
        y_left = [tiers_4sc[c]["tier_means"][t_key] for c in conds_left]
        y_right = [tiers_4sc[c]["tier_means"][t_key] for c in conds_right]
        ax1.plot([0, 1, 2], y_left, color=col, marker=marker, markersize=4.5, lw=1.4, label=label)
        ax2.plot([0, 1], y_right, color=col, marker=marker, markersize=4.5, lw=1.4)

    for ax in (ax1, ax2):
        ax.set_ylim(0.60, 1.005)
        ax.set_yticks(np.arange(0.60, 1.01, 0.05))
        ax.grid(True, linestyle=":", color="#b0b0b0", alpha=0.65)
        ax.tick_params(axis="both", labelsize=6.5)

    ax1.set_xlim(-0.25, 2.25)
    ax1.set_xticks([0, 1, 2])
    ax1.set_xticklabels(["Realistic\np=0.90", "Realistic\np=0.95", "Realistic\np=0.99"], fontsize=6.4)
    ax1.set_ylabel("Mean WCS (4 bf16 scorers)", fontsize=7.2)

    ax2.set_xlim(-0.32, 1.32)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Strict full\np=0.99", "Shipped\ndefault"], fontsize=6.4)
    ax2.tick_params(labelleft=False)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.54, 0.01),
        ncol=2,
        fontsize=6.5,
        frameon=False,
        columnspacing=1.4,
        handletextpad=0.5,
    )

    save_both(fig, "fig4_filter_sensitivity", tight=False)


def draw_fig5_family_decomposed(data: dict):
    # Equal-height 2x3 panel layout (figsize=(7.15, 3.65)), matching WCS_revised_2_Nov.pdf Figure 5
    fig = plt.figure(figsize=(7.15, 3.65))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 1.0],
        hspace=0.48,
        wspace=0.22,
        left=0.068,
        right=0.985,
        top=0.86,
        bottom=0.13,
    )

    ax_qwen = fig.add_subplot(gs[0, 0])
    ax_gemma = fig.add_subplot(gs[0, 1])
    ax_leg = fig.add_subplot(gs[0, 2])
    ax_leg.axis("off")

    ax_llama = fig.add_subplot(gs[1, 0])
    ax_mistral = fig.add_subplot(gs[1, 1])
    ax_gemma31 = fig.add_subplot(gs[1, 2])

    conds = ["realistic_0.9", "realistic_0.95", "realistic_0.99", "strict_full_0.99", "default_shipped"]
    x_labels = [
        "realistic\np = 0.90",
        "realistic\np = 0.95",
        "realistic\np = 0.99",
        "strict_full\np = 0.99",
        "shipped\ndefault",
    ]
    x = np.arange(len(conds))

    t8 = data["eli5_4family_table8_cells"]
    tier_srcs = {
        "human": ["human"],
        "frontier_prop": ["gpt-6-astra", "claude-opus-5.5", "gemini-4-argon"],
        "large_ow": [
            "deepseek-v4.1-flash-1017",
            "glm-5.3",
            "minimax-m3",
            "kimi-k3",
            "mimo-v2.6-pro",
            "deepseek-v4-pro-0813",
            "qwen3.8-2.4t-a95b",
        ],
        "small_ow": [
            "qwen-4b",
            "qwen-9b",
            "qwen-27b-int8",
            "qwen-32b-int8",
            "gemma-e4b",
            "gemma-12b",
            "llama-3.2-1b",
            "llama-3.1-8b",
            "mistral-7b",
        ],
    }

    def get_fam_series(scorers: list[str], src_Map: dict):
        res = {t: [] for t in ("human", "frontier_prop", "large_ow", "small_ow")}
        for c in conds:
            for t, srcs in src_Map.items():
                vals = [t8[sc][s][c] for sc in scorers for s in srcs if sc in t8 and s in t8[sc]]
                res[t].append(float(np.mean(vals)))
        return res

    qwen_data = data["eli5_family_decomposed"]["qwen"]
    gemma_data = data["eli5_family_decomposed"]["gemma"]
    llama_series = get_fam_series(["llama-3.2-1b", "llama-3.1-8b"], tier_srcs)
    mistral_series = get_fam_series(["mistral-7b"], tier_srcs)
    gemma31_series = get_fam_series(["gemma-31b"], tier_srcs)

    series_meta = [
        ("human", "Human", TIER_COLORS["human"], "o"),
        ("frontier_prop", "Frontier Proprietary", TIER_COLORS["frontier_prop"], "s"),
        ("large_ow", "Large Open-Weight", TIER_COLORS["large_ow"], "^"),
        ("small_ow", "Small Open-Weight", TIER_COLORS["small_ow"], "D"),
    ]

    for ax, fam_dict, title in [
        (ax_qwen, qwen_data, "Qwen scorers (4B, 9B bf16)"),
        (ax_gemma, gemma_data, "Gemma scorers (E4B, 12B bf16)"),
    ]:
        for t_key, label, col, marker in series_meta:
            ys = [fam_dict[c][t_key] for c in conds]
            ax.plot(x, ys, color=col, marker=marker, markersize=4.2, lw=1.3, label=label)
        ax.set_title(title, fontsize=8.4, fontweight="bold", pad=4)

    for ax, fam_ser, title in [
        (ax_llama, llama_series, "Llama scorers (1B, 8B bf16)"),
        (ax_mistral, mistral_series, "Mistral-7B-v0.3 (bf16)"),
        (ax_gemma31, gemma31_series, "Gemma-4-31B-it (31B bf16)"),
    ]:
        for t_key, label, col, marker in series_meta:
            ax.plot(x, fam_ser[t_key], color=col, marker=marker, markersize=4.2, lw=1.3, label=label)
        ax.set_title(title, fontsize=8.4, fontweight="bold", pad=4)

    for ax in (ax_qwen, ax_gemma, ax_llama, ax_mistral, ax_gemma31):
        ax.set_ylim(0.55, 1.015)
        ax.set_yticks([0.6, 0.7, 0.8, 0.9, 1.0])
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=6.2)
        ax.tick_params(axis="y", labelsize=7.2)
        ax.grid(True, linestyle=":", color="#b0b0b0", alpha=0.65)

    ax_qwen.set_ylabel("Mean WCS", fontsize=8.0)
    ax_llama.set_ylabel("Mean WCS", fontsize=8.0)

    handles, labels = ax_qwen.get_legend_handles_labels()
    ax_leg.legend(
        handles,
        labels,
        loc="center",
        fontsize=8.0,
        frameon=True,
        facecolor="#fafafa",
        edgecolor="#cccccc",
        title="Source Tier",
        title_fontsize=8.2,
    )

    fig.suptitle(
        "Tier-level reachability across filter configurations, decomposed by scorer family (HC3 ELI5)",
        fontsize=9.2,
        fontweight="bold",
        y=0.965,
    )
    save_both(fig, "fig5_family_decomposed", tight=False)


def draw_fig6_support_expansion(data: dict):
    # Equal-height 2x3 panel layout (figsize=(7.50, 3.55)), matching WCS_revised_2_Nov.pdf Figure 6
    # Plots the 4 core convergence curves (Human, GPT-6 Astra, Claude Opus 5.5, Large Open-Wt. mean)
    # so the y-axis zooms cleanly into the human/frontier convergence zone on all 5 scorers!
    fig = plt.figure(figsize=(7.50, 3.55))
    gs = fig.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 1.0],
        hspace=0.45,
        wspace=0.24,
        left=0.068,
        right=0.985,
        top=0.86,
        bottom=0.11,
    )

    ax_q9 = fig.add_subplot(gs[0, 0])
    ax_g12 = fig.add_subplot(gs[0, 1])
    ax_leg = fig.add_subplot(gs[0, 2])
    ax_leg.axis("off")

    ax_l8 = fig.add_subplot(gs[1, 0])
    ax_m7 = fig.add_subplot(gs[1, 1])
    ax_g31 = fig.add_subplot(gs[1, 2])

    val_data = data["top1000_validation"]
    x = np.array([0, 1, 2])
    x_labels = ["K=100", "K=1000", "Full |V|"]

    curves = [
        ("human", "Human", "#2b2b2b", "o", "--"),
        ("gpt-6-astra", "GPT-6 Astra", "#c44e52", "s", "-"),
        ("claude-opus-5.5", "Claude Opus 5.5", "#e07b7e", "^", "-"),
        ("large_ow_mean", "Large Open-Wt. (mean)", "#4c72b0", "D", "-"),
    ]

    panels = [
        (ax_q9, "qwen-9b", "Qwen-9B", (0.905, 0.975), [0.91, 0.93, 0.95, 0.97]),
        (ax_g12, "gemma-12b", "Gemma-12B", (0.70, 0.865), [0.70, 0.75, 0.80, 0.85]),
        (ax_l8, "llama-3.1-8b", "Llama-3.1-8B-Instruct", (0.91, 0.975), [0.91, 0.93, 0.95, 0.97]),
        (ax_m7, "mistral-7b", "Mistral-7B-Instruct-v0.3", (0.82, 0.945), [0.83, 0.86, 0.89, 0.92]),
        (ax_g31, "gemma-31b", "Gemma-4-31B-it (31B bf16)", (0.65, 0.78), [0.66, 0.70, 0.74, 0.78]),
    ]

    for ax, sc_key, title, ylim, yticks in panels:
        sc_d = val_data[sc_key]
        for key, label, col, marker, ls in curves:
            row = sc_d[key]
            ys = [row["wcs_100"], row["wcs_1000"], row["wcs_full"]]
            ax.plot(x, ys, color=col, marker=marker, linestyle=ls, markersize=4.5, lw=1.35, label=label)
        ax.set_title(title, fontsize=8.4, fontweight="bold", pad=4)
        ax.set_ylim(*ylim)
        ax.set_yticks(yticks)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=7.2)
        ax.tick_params(axis="y", labelsize=7.2)
        ax.grid(True, linestyle=":", color="#b0b0b0", alpha=0.65)

    ax_q9.set_ylabel("WCS at p=0.99", fontsize=8.0)
    ax_l8.set_ylabel("WCS at p=0.99", fontsize=8.0)

    handles, labels = ax_q9.get_legend_handles_labels()
    ax_leg.legend(
        handles,
        labels,
        loc="center",
        fontsize=8.0,
        frameon=True,
        facecolor="#fafafa",
        edgecolor="#cccccc",
        title="Source / Tier",
        title_fontsize=8.2,
    )

    fig.suptitle(
        "Candidate-support expansion (K = 100 → K = 1000 → Full |V|) at p = 0.99 on HC3 ELI5",
        fontsize=9.2,
        fontweight="bold",
        y=0.965,
    )
    save_both(fig, "fig6_support_expansion", tight=False)


def draw_fig7_same_family(data: dict):
    # Clean single-column bar chart (figsize=(3.60, 2.55)) with non-overlapping labels above CI whiskers
    sfa = data["same_family_advantage"]
    groups = [
        ("Qwen/Gemma\nELI5", sfa["eli5_qwen_gemma"]["realistic_0.99"], sfa["eli5_qwen_gemma"]["ortho_clean_0.99"]),
        ("All 4 families\nELI5", sfa["eli5_all_4_families"]["realistic_0.99"], sfa["eli5_all_4_families"]["ortho_clean_0.99"]),
        ("Qwen/Gemma\nWiki CS/AI", sfa["wiki_qwen_gemma"]["realistic_0.99"], sfa["wiki_qwen_gemma"]["ortho_clean_0.99"]),
        ("All 4 families\nWiki CS/AI", sfa["wiki_all_4_families"]["realistic_0.99"], sfa["wiki_all_4_families"]["ortho_clean_0.99"]),
    ]

    fig = plt.figure(figsize=(3.60, 2.55))
    ax = fig.add_axes([0.195, 0.175, 0.780, 0.775])

    x = np.arange(len(groups))
    width = 0.34

    base_means = [g[1]["delta"] for g in groups]
    base_err_low = [g[1]["delta"] - g[1]["ci_low"] for g in groups]
    base_err_high = [g[1]["ci_high"] - g[1]["delta"] for g in groups]
    base_ci_high = [g[1]["ci_high"] for g in groups]

    ortho_means = [g[2]["delta"] for g in groups]
    ortho_err_low = [g[2]["delta"] - g[2]["ci_low"] for g in groups]
    ortho_err_high = [g[2]["ci_high"] - g[2]["delta"] for g in groups]
    ortho_ci_high = [g[2]["ci_high"] for g in groups]

    b1 = ax.bar(
        x - width / 2,
        base_means,
        width,
        yerr=[base_err_low, base_err_high],
        capsize=2.5,
        color="#2e5984",
        edgecolor="none",
        error_kw=dict(lw=0.9, ecolor="#111111"),
        label="All words (baseline)",
        zorder=3,
    )
    b2 = ax.bar(
        x + width / 2,
        ortho_means,
        width,
        yerr=[ortho_err_low, ortho_err_high],
        capsize=2.5,
        color="#6baed6",
        edgecolor="none",
        error_kw=dict(lw=0.9, ecolor="#111111"),
        label="Orthographic-clean",
        zorder=3,
    )

    for i, bar in enumerate(b1):
        h = bar.get_height()
        top = base_ci_high[i]
        ax.text(
            bar.get_x() + bar.get_width() / 2.0 - 0.04,
            top + 0.0032,
            f"+{h:.4f}",
            ha="center",
            va="bottom",
            fontsize=5.8,
            fontweight="bold",
            color="#111111",
        )
    for i, bar in enumerate(b2):
        h = bar.get_height()
        top = ortho_ci_high[i]
        ax.text(
            bar.get_x() + bar.get_width() / 2.0 + 0.05,
            top + 0.0007,
            f"+{h:.4f}",
            ha="center",
            va="bottom",
            fontsize=5.8,
            color="#1f4e78",
        )

    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=6.8)
    ax.set_ylim(0, 0.052)
    ax.set_yticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05])
    ax.tick_params(axis="y", labelsize=7.2)
    ax.grid(axis="y", linestyle=":", color="#b0b0b0", alpha=0.65, zorder=0)
    ax.set_ylabel("Same-family minus cross-family\nmean WCS (self excluded)", fontsize=7.2)
    ax.legend(loc="upper right", fontsize=6.5, frameon=True, facecolor="white", edgecolor="#cccccc")

    save_both(fig, "fig7_same_family", tight=False)


def main():
    data = json.loads(Path("results/reports/revised_nov2_recomputed.json").read_text())
    draw_fig1_pipeline(data)
    draw_fig2_tier_hierarchy(data)
    draw_fig3_filter_families(data)
    draw_fig4_filter_sensitivity(data)
    draw_fig5_family_decomposed(data)
    draw_fig6_support_expansion(data)
    draw_fig7_same_family(data)
    print(f"Generated all 7 figures in {OUT_DIR}")


if __name__ == "__main__":
    main()
