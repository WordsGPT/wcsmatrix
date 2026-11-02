#!/usr/bin/env python3
"""
Generate all 7 figures for WCS_revised_2_Nov (in both vector PDF and high-DPI PNG)
using the recomputed data in results/reports/revised_nov2_recomputed.json,
replacing all recomputable TBD placeholders with exact measured values and 95% CIs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.8,
    }
)

COLORS = {
    "human": "#2b2b2b",
    "frontier_prop": "#c44e52",
    "large_ow": "#4c72b0",
    "small_ow": "#55a868",
}


def save_both(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def draw_fig1_pipeline(out_dir: Path):
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 3.4)
    ax.axis("off")

    ax.text(
        5.25,
        3.22,
        "Evaluation pipeline: every source response is scored by every scorer with the Word Coverage Score (WCS)",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#222222",
    )

    def header_box(x, y, w, h, title):
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor="#2f3e4e",
            edgecolor="#2f3e4e",
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")

    def sub_box(x, y, w, h, lines, bg, border, text_color="#222222", ls="-"):
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            facecolor=bg,
            edgecolor=border,
            linewidth=1.0,
            linestyle=ls,
        )
        ax.add_patch(rect)
        for idx, (txt, bold) in enumerate(lines):
            yy = y + h - (idx + 0.65) * (h / max(len(lines), 1))
            ax.text(
                x + w / 2,
                yy,
                txt,
                ha="center",
                va="center",
                color=text_color,
                fontsize=8.2,
                fontweight="bold" if bold else "normal",
            )

    # Column 1: Generators
    header_box(0.2, 2.55, 2.8, 0.36, "Generators (sources)")
    sub_box(0.2, 2.02, 2.8, 0.46, [("Human writing", False), ("(HC3, detokenized)", False)], "#f4f4f4", "#888888")
    sub_box(0.2, 1.50, 2.8, 0.46, [("Frontier Proprietary", False), ("3 models", False)], "#f9ebea", "#c44e52", "#7b241c")
    sub_box(0.2, 0.98, 2.8, 0.46, [("Large Open-Weight", False), ("7 models (427B-2.78T)", False)], "#ebf5fb", "#4c72b0", "#1b4f72")
    sub_box(0.2, 0.46, 2.8, 0.46, [("Small Open-Weight", False), ("5 models (Qwen 3.5, Gemma 4)", False)], "#e9f7ef", "#55a868", "#196f3d")
    sub_box(0.2, -0.06, 2.8, 0.46, [("Llama, Mistral, Gemma-31B", False), ("4-family & high-VRAM suite", False)], "#fef9e7", "#d4ac0d", "#7d6608", ls="--")

    # Arrow 1 -> 2
    ax.annotate("", xy=(4.05, 1.45), xytext=(3.05, 1.45), arrowprops=dict(arrowstyle="->", lw=1.4, color="#444444"))
    ax.text(3.55, 1.72, "50 prompts\nper domain", ha="center", va="center", fontsize=8.0, color="#333333")
    ax.text(3.55, 1.15, "responses\n150-400\nwords", ha="center", va="center", fontsize=8.0, color="#333333")

    # Column 2: Scorers
    header_box(4.1, 2.55, 2.9, 0.36, "Scorers (WCS)")
    sub_box(
        4.1,
        1.88,
        2.9,
        0.58,
        [("5 open-weight scorers", False), ("Qwen 3.5 (4B-27B), Gemma 4 (E4B, 12B)", False)],
        "#ebf5fb",
        "#4c72b0",
        "#1b4f72",
    )
    sub_box(
        4.1,
        1.30,
        2.9,
        0.50,
        [("Llama, Mistral, Gemma-31B", False), ("4-family & high-VRAM suite", False)],
        "#fef9e7",
        "#d4ac0d",
        "#7d6608",
        ls="--",
    )
    sub_box(
        4.1,
        0.10,
        2.9,
        1.10,
        [
            ("teacher-forced pass", False),
            ("per position: candidate-pool", False),
            ("filter (Figure 3)", False),
            ("word covered iff every", False),
            ("sub-token survives the filter", False),
            ("WCS = mean word coverage", True),
        ],
        "#f8f9f9",
        "#666666",
    )

    # Arrow 2 -> 3
    ax.annotate("", xy=(7.65, 1.45), xytext=(7.05, 1.45), arrowprops=dict(arrowstyle="->", lw=1.4, color="#444444"))

    # Column 3: Outputs
    header_box(7.7, 2.55, 2.6, 0.36, "Outputs")
    sub_box(
        7.7,
        1.78,
        2.6,
        0.66,
        [("scorer-by-source", False), ("WCS matrix", False), ("(Figure 2, Table 4)", False)],
        "#f8f9f9",
        "#666666",
    )
    sub_box(
        7.7,
        1.02,
        2.6,
        0.66,
        [("tier means and contrasts", False), ("(Table 3)", False)],
        "#f8f9f9",
        "#666666",
    )
    sub_box(
        7.7,
        0.26,
        2.6,
        0.66,
        [("filter-family and", False), ("candidate-support audits", False), ("(Figures 3-6)", False)],
        "#f8f9f9",
        "#666666",
    )

    save_both(fig, out_dir, "fig1_pipeline")


def draw_fig2_tier_hierarchy(data: dict, out_dir: Path):
    deltas = data["eli5_core_source_deltas_4sc"]["realistic_0.99"]
    tiers = data["eli5_core_4sc_tiers"]["realistic_0.99"]["tier_means"]

    rows = [
        ("Human (HC3 ELI5)", "human", "human"),
        ("GPT-6 Astra", "gpt-6-astra", "frontier_prop"),
        ("Claude Opus 5.5", "claude-opus-5.5", "frontier_prop"),
        ("Gemini 4 Argon", "gemini-4-argon", "frontier_prop"),
        ("DeepSeek-V4.1-Flash", "deepseek-v4.1-flash-1017", "large_ow"),
        ("GLM-5.3", "glm-5.3", "large_ow"),
        ("MiniMax-M3", "minimax-m3", "large_ow"),
        ("Kimi-K3", "kimi-k3", "large_ow"),
        ("MiMo-V2.6-Pro", "mimo-v2.6-pro", "large_ow"),
        ("DeepSeek-V4-Pro", "deepseek-v4-pro-0813", "large_ow"),
        ("Qwen3.8-2.4T-A95B", "qwen3.8-2.4t-a95b", "large_ow"),
        ("Qwen-4B", "qwen-4b", "small_ow"),
        ("Qwen-9B", "qwen-9b", "small_ow"),
        ("Qwen-27B (int8)", "qwen-27b-int8", "small_ow"),
        ("Gemma-12B", "gemma-12b", "small_ow"),
        ("Gemma-E4B", "gemma-e4b", "small_ow"),
    ]

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    y_positions = []
    curr_y = 0
    prev_tier = None
    tier_spans = {}
    for label, key, tier in rows:
        if prev_tier is not None and tier != prev_tier:
            curr_y += 0.7
        y_positions.append((curr_y, label, key, tier))
        tier_spans.setdefault(tier, []).append(curr_y)
        curr_y += 1.0
        prev_tier = tier

    max_y = y_positions[-1][0] + 0.8
    ax.set_ylim(max_y, -0.8)
    ax.set_xlim(0.80, 1.00)

    # Background bands
    band_colors = {
        "human": "#eeeeee",
        "frontier_prop": "#f9ebea",
        "large_ow": "#ebf5fb",
        "small_ow": "#e9f7ef",
    }
    for tier, ys in tier_spans.items():
        y0, y1 = min(ys) - 0.45, max(ys) + 0.45
        ax.axhspan(y0, y1, color=band_colors[tier], alpha=0.65, zorder=0)
        tm = tiers[tier]
        if tier == "human":
            ax.axvline(tm, color=COLORS["human"], linestyle="--", linewidth=1.2, zorder=2)
            ax.text(tm + 0.002, y0 + 0.45, f"Human writing ({tm:.4f})", color=COLORS["human"], fontsize=8.5, va="center")
        else:
            ax.plot([tm, tm], [y0, y1], color=COLORS[tier], linestyle="--", linewidth=1.3, zorder=2)
            ax.text(tm, y0 - 0.12, f"tier mean {tm:.4f}", color=COLORS[tier], fontsize=8.2, ha="center", va="bottom", fontweight="bold")

    for y, label, key, tier in y_positions:
        val = deltas[key]["source_mean_4sc"]
        ax.scatter([val], [y], color=COLORS[tier], s=36, zorder=3)
        if key != "human":
            offset = -0.017 if key == "gpt-6-astra" else 0.003
            ha = "right" if key == "gpt-6-astra" else "left"
            ax.text(val + (0.003 if ha == "left" else -0.003), y, f"{val:.4f}", va="center", ha=ha, fontsize=8.2, color="#222222")

    ax.set_yticks([r[0] for r in y_positions])
    ax.set_yticklabels([r[1] for r in y_positions], fontsize=9.0)
    ax.set_xticks(np.arange(0.80, 1.001, 0.02))
    ax.xaxis.grid(True, linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Mean WCS (4 bf16 scorers)", fontsize=10.0)
    ax.set_title("Lexical reachability by source tier and generator", fontsize=11.0, pad=12)

    save_both(fig, out_dir, "fig2_tier_hierarchy")


def draw_fig3_filter_families(out_dir: Path):
    fig, ax = plt.subplots(figsize=(10.5, 4.0))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    ax.text(
        5.25,
        4.02,
        "Candidate-pool filters evaluated in the audit (block diagram: each block is one stage of the filter)",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#222222",
    )

    def badge(x, y, letter):
        rect = patches.FancyBboxPatch(
            (x, y - 0.18), 0.38, 0.36, boxstyle="round,pad=0.02,rounding_size=0.05", facecolor="#2f3e4e", edgecolor="#2f3e4e"
        )
        ax.add_patch(rect)
        ax.text(x + 0.19, y, letter, ha="center", va="center", color="white", fontsize=9.5, fontweight="bold")

    def stage_box(x, y, w, h, lines, bg, border):
        rect = patches.FancyBboxPatch(
            (x, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", facecolor=bg, edgecolor=border, linewidth=1.1
        )
        ax.add_patch(rect)
        for idx, txt in enumerate(lines):
            yy = y + (len(lines) - 1 - 2 * idx) * 0.11
            ax.text(x + w / 2, yy, txt, ha="center", va="center", fontsize=8.2, color="#222222")

    def arrow(x0, x1, y):
        ax.annotate("", xy=(x1, y), xytext=(x0, y), arrowprops=dict(arrowstyle="->", lw=1.3, color="#444444"))

    # Row A
    yA = 3.30
    badge(0.2, yA, "A")
    ax.text(0.75, yA, "Realistic top-p\nover top-K\n(main configuration)", va="center", fontsize=8.5, color="#222222")
    stage_box(2.5, yA, 1.5, 0.56, ["top-K window", "K = 100"], "#ebf5fb", "#4c72b0")
    arrow(4.02, 4.30, yA)
    stage_box(4.32, yA, 1.6, 0.56, ["renormalize", "within window"], "#e9f7ef", "#55a868")
    arrow(5.94, 6.22, yA)
    stage_box(6.24, yA, 1.85, 0.56, ["nucleus until", "cumulative mass = p", "p = 0.90 / 0.95 / 0.99"], "#f5eef8", "#8e44ad")
    ax.text(8.25, yA, "validation:\nK = 1000 and exact\nfull vocabulary", va="center", fontsize=8.0, style="italic", color="#555555")

    # Row B
    yB = 2.40
    badge(0.2, yB, "B")
    ax.text(0.75, yB, "Strict unnormalized\ntop-K", va="center", fontsize=8.5, color="#222222")
    stage_box(2.5, yB, 1.5, 0.56, ["top-K window", "K = 100"], "#ebf5fb", "#4c72b0")
    arrow(4.02, 4.30, yB)
    stage_box(4.32, yB, 1.6, 0.56, ["no", "renormalization"], "#f2f3f4", "#7f8c8d")
    arrow(5.94, 6.22, yB)
    stage_box(6.24, yB, 1.85, 0.56, ["nucleus over", "raw mass", "p = 0.99"], "#f5eef8", "#8e44ad")

    # Row C
    yC = 1.50
    badge(0.2, yC, "C")
    ax.text(0.75, yC, "Exact full-vocabulary\ntop-p (strict_full)", va="center", fontsize=8.5, color="#222222")
    stage_box(2.5, yC, 1.5, 0.56, ["full", "vocabulary |V|"], "#ebf5fb", "#4c72b0")
    arrow(4.02, 4.30, yC)
    stage_box(4.32, yC, 1.85, 0.56, ["nucleus over full", "vocabulary", "p = 0.99"], "#f5eef8", "#8e44ad")

    # Row D
    yD = 0.55
    badge(0.2, yD, "D")
    ax.text(0.75, yD, "Shipped generation\ndefaults (scorer\ndefault filters)", va="center", fontsize=8.5, color="#222222")
    stage_box(2.5, yD, 1.35, 0.56, ["temperature", "scaling T"], "#fdebd0", "#d68910")
    arrow(3.87, 4.10, yD)
    stage_box(4.12, yD, 1.15, 0.56, ["top-k", "window"], "#ebf5fb", "#4c72b0")
    arrow(5.29, 5.52, yD)
    stage_box(5.54, yD, 1.15, 0.56, ["renormalize"], "#e9f7ef", "#55a868")
    arrow(6.71, 6.94, yD)
    stage_box(6.96, yD, 1.15, 0.56, ["nucleus p"], "#f5eef8", "#8e44ad")
    ax.text(
        8.25,
        yD,
        "Qwen 3.5: T = 0.6, k = 20, p = 0.95\nGemma 4: T = 1.0, k = 64, p = 0.95\nvariants: Qwen non-thinking\n(T = 0.7, k = 20, p = 0.80), T = 1.0",
        va="center",
        fontsize=7.6,
        style="italic",
        color="#555555",
    )

    save_both(fig, out_dir, "fig3_filter_families")


def draw_fig4_filter_sensitivity(data: dict, out_dir: Path):
    tiers = data["eli5_core_4sc_tiers"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 3.6), sharey=False, gridspec_kw={"width_ratios": [1.4, 1.0]})

    tier_keys = [
        ("human", "Human", "s", "-"),
        ("frontier_prop", "Frontier Prop.", "D", "--"),
        ("large_ow", "Large Open-Wt.", "^", "-."),
        ("small_ow", "Small Open-Wt.", "o", "-"),
    ]

    x1 = [0, 1, 2]
    conds1 = ["realistic_0.9", "realistic_0.95", "realistic_0.99"]
    x2 = [0, 1]
    conds2 = ["strict_full_0.99", "default_shipped"]

    for tkey, tlabel, marker, ls in tier_keys:
        y1 = [tiers[c]["tier_means"][tkey] for c in conds1]
        ax1.plot(x1, y1, marker=marker, linestyle=ls, color=COLORS[tkey], label=tlabel, linewidth=1.4, markersize=5)
        y2 = [tiers[c]["tier_means"][tkey] for c in conds2]
        ax2.plot(x2, y2, marker=marker, linestyle=ls, color=COLORS[tkey], label=tlabel, linewidth=1.4, markersize=5)

    for ax in (ax1, ax2):
        ax.set_ylim(0.55, 1.01)
        ax.set_yticks(np.arange(0.55, 1.001, 0.05))
        ax.yaxis.grid(True, linestyle=":", alpha=0.5)
        ax.tick_params(axis="both", labelsize=8.0)

    ax1.set_xticks(x1)
    ax1.set_xticklabels(["Realistic\np=0.90", "Realistic\np=0.95", "Realistic\np=0.99"], fontsize=8.0)
    ax1.set_ylabel("Mean WCS (4 bf16 scorers)", fontsize=8.8)

    ax2.set_xticks(x2)
    ax2.set_xticklabels(["Strict full\np=0.99", "Shipped\ndefault"], fontsize=8.0)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8.2, bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_both(fig, out_dir, "fig4_filter_sensitivity")


def draw_fig5_family_decomposed(data: dict, out_dir: Path):
    fam_data = data["eli5_family_decomposed"]
    conds = ["realistic_0.9", "realistic_0.95", "realistic_0.99", "strict_full_0.99", "default_shipped"]
    xlabels = ["realistic\np = 0.90", "realistic\np = 0.95", "realistic\np = 0.99", "strict_full\np = 0.99", "shipped\ndefault"]

    fig = plt.figure(figsize=(10.0, 3.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 0.85], hspace=0.42, wspace=0.22)

    ax_qwen = fig.add_subplot(gs[0, 0])
    ax_gemma = fig.add_subplot(gs[0, 1])
    ax_leg = fig.add_subplot(gs[0, 2])
    ax_leg.axis("off")

    tier_keys = [
        ("human", "Human", "s"),
        ("frontier_prop", "Frontier Proprietary", "D"),
        ("large_ow", "Large Open-Weight", "^"),
        ("small_ow", "Small Open-Weight", "o"),
    ]

    for ax, fam_key, title in [(ax_qwen, "qwen", "Qwen scorers (bf16)"), (ax_gemma, "gemma", "Gemma scorers (bf16)")]:
        xs = list(range(len(conds)))
        for tkey, tlabel, marker in tier_keys:
            ys = [fam_data[fam_key][c][tkey] for c in conds]
            ax.plot(xs, ys, marker=marker, color=COLORS[tkey], label=tlabel, linewidth=1.3, markersize=4.5)
        ax.set_title(title, fontsize=9.0)
        ax.set_ylim(0.55, 1.02)
        ax.set_xticks(xs)
        ax.set_xticklabels([])
        ax.yaxis.grid(True, linestyle=":", alpha=0.5)
        ax.tick_params(axis="both", labelsize=8.0)

    ax_qwen.set_ylabel("Mean WCS over family scorers", fontsize=8.5)
    handles, labels = ax_qwen.get_legend_handles_labels()
    ax_leg.legend(handles, labels, loc="center left", frameon=False, fontsize=8.5)

    # Bottom row: Llama, Mistral (measured in 4-family suite!), Gemma-4-31B-it (TBD on high-VRAM server)
    exp_cells = data["eli5_4family_table8_cells"]
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
        "small_ow": ["qwen-4b", "qwen-9b", "qwen-27b-int8", "gemma-12b", "gemma-e4b"],
    }

    has_gemma31b = "gemma-31b" in exp_cells
    for col_idx, (title, sc_list) in enumerate(
        [
            ("Llama scorers (1B, 8B bf16)", ["llama-3.2-1b", "llama-3.1-8b"]),
            ("Mistral scorer (7B bf16)", ["mistral-7b"]),
            (
                "Gemma-4-31B-it (31B bf16)" if has_gemma31b else "Gemma-4-31B-it (high-VRAM TBD)",
                ["gemma-31b"] if has_gemma31b else [],
            ),
        ]
    ):
        ax = fig.add_subplot(gs[1, col_idx])
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels(xlabels, fontsize=6.8)
        ax.set_ylim(0.55, 1.02)
        if sc_list:
            sample_cell = exp_cells[sc_list[0]]["human"]
            if "realistic_0.9" in sample_cell:
                avail_x = [0, 1, 2, 3, 4]
                avail_keys = ["realistic_0.9", "realistic_0.95", "realistic_0.99", "strict_full_0.99", "default_shipped"]
            else:
                avail_x = [1, 2, 3, 4]
                avail_keys = ["realistic_0.95", "realistic_0.99", "strict_full_0.99", "default_shipped"]
            for tkey, tlabel, marker in tier_keys:
                ys = [
                    float(np.mean([exp_cells[sc][s][ck] for sc in sc_list for s in tier_srcs[tkey]]))
                    for ck in avail_keys
                ]
                ax.plot(avail_x, ys, marker=marker, color=COLORS[tkey], linewidth=1.2, markersize=4.0)
            ax.yaxis.grid(True, linestyle=":", alpha=0.5)
            ax.tick_params(axis="y", labelsize=7.5)
        else:
            ax.set_yticks([])
            ax.text(2.0, 0.785, "TBD (>60 GB VRAM)", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#444444")

    fig.suptitle("Tier-level reachability across filter configurations, decomposed by scorer family (HC3 ELI5)", fontsize=10.2, y=0.99)
    save_both(fig, out_dir, "fig5_family_decomposed")


def draw_fig6_support_expansion(data: dict, out_dir: Path):
    val = data["top1000_validation"]
    fig = plt.figure(figsize=(9.6, 3.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 0.80], hspace=0.45, wspace=0.25)

    curves = [
        ("human", "Human", "o", "--", COLORS["human"]),
        ("gpt-6-astra", "GPT-6 Astra", "s", "-", COLORS["frontier_prop"]),
        ("claude-opus-5.5", "Claude Opus 5.5", "D", "-", "#e07b7e"),
        ("large_ow_mean", "Large Open-Wt. (mean)", "^", "-", COLORS["large_ow"]),
        ("small_ow_5_mean", "Small Open-Wt. (mean)", "v", "-", COLORS["small_ow"]),
    ]

    for idx, (sc_key, sc_title, ylim) in enumerate(
        [("qwen-9b", "Qwen-9B", (0.91, 0.995)), ("gemma-12b", "Gemma-12B", (0.70, 0.915))]
    ):
        ax = fig.add_subplot(gs[0, idx])
        xs = [0, 1, 2]
        for ckey, clabel, marker, ls, col in curves:
            row = val[sc_key][ckey]
            ys = [row["wcs_100"], row["wcs_1000"], row["wcs_full"]]
            ax.plot(xs, ys, marker=marker, linestyle=ls, color=col, label=clabel, linewidth=1.4, markersize=4.5)
        ax.set_title(sc_title, fontsize=9.0)
        ax.set_xticks(xs)
        ax.set_xticklabels(["K=100", "K=1000", "Full |V|"], fontsize=8.0)
        ax.set_ylim(*ylim)
        ax.yaxis.grid(True, linestyle=":", alpha=0.5)
        ax.tick_params(axis="y", labelsize=8.0)
        if idx == 0:
            ax.set_ylabel("WCS at p=0.99", fontsize=8.8)

    ax_leg = fig.add_subplot(gs[0, 2])
    ax_leg.axis("off")
    handles, labels = fig.axes[0].get_legend_handles_labels()
    ax_leg.legend(handles, labels, loc="center left", frameon=False, fontsize=8.2)

    bottom_panels = [
        ("llama-3.1-8b", "Llama-3.1-8B"),
        ("mistral-7b", "Mistral-7B"),
        (
            "gemma-31b",
            "Gemma-4-31B-it" if "gemma-31b" in val else "Gemma-4-31B-it (high-VRAM TBD)",
        ),
    ]
    for col_idx, (sc_key, title) in enumerate(bottom_panels):
        ax = fig.add_subplot(gs[1, col_idx])
        ax.set_title(title, fontsize=8.5)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["K=100", "K=1000", "Full |V|"], fontsize=7.8)
        if sc_key in val:
            xs = [0, 1, 2]
            all_vals = []
            for ckey, clabel, marker, ls, col in curves:
                row = val[sc_key][ckey]
                ys = [row["wcs_100"], row["wcs_1000"], row["wcs_full"]]
                all_vals.extend(ys)
                ax.plot(xs, ys, marker=marker, linestyle=ls, color=col, linewidth=1.2, markersize=4.0)
            ax.set_ylim(min(all_vals) - 0.015, max(all_vals) + 0.015)
            ax.yaxis.grid(True, linestyle=":", alpha=0.5)
            ax.tick_params(axis="y", labelsize=7.5)
        else:
            ax.set_yticks([])
            ax.text(1.0, 0.5, "TBD", ha="center", va="center", fontsize=9.0, fontweight="bold", color="#444444")

    save_both(fig, out_dir, "fig6_support_expansion")


def draw_fig7_same_family(data: dict, out_dir: Path):
    sfa = data["same_family_advantage"]
    groups = [
        ("Qwen/Gemma\nELI5", sfa["eli5_qwen_gemma"]["realistic_0.99"], sfa["eli5_qwen_gemma"]["ortho_clean_0.99"]),
        ("All 4 families\nELI5", sfa["eli5_all_4_families"]["realistic_0.99"], sfa["eli5_all_4_families"]["ortho_clean_0.99"]),
        ("Qwen/Gemma\nWiki CS/AI", sfa["wiki_qwen_gemma"]["realistic_0.99"], sfa["wiki_qwen_gemma"]["ortho_clean_0.99"]),
    ]
    if "wiki_all_4_families" in sfa:
        groups.append(
            ("All 4 families\nWiki CS/AI", sfa["wiki_all_4_families"]["realistic_0.99"], sfa["wiki_all_4_families"]["ortho_clean_0.99"])
        )

    fig, ax = plt.subplots(figsize=(5.8 if len(groups) == 4 else 5.4, 3.4))
    xs = np.arange(len(groups))
    width = 0.34

    vals_base = [g[1]["delta"] for g in groups]
    err_low_base = [g[1]["delta"] - g[1]["ci_low"] for g in groups]
    err_high_base = [g[1]["ci_high"] - g[1]["delta"] for g in groups]

    vals_clean = [g[2]["delta"] for g in groups]
    err_low_clean = [g[2]["delta"] - g[2]["ci_low"] for g in groups]
    err_high_clean = [g[2]["ci_high"] - g[2]["delta"] for g in groups]

    bars1 = ax.bar(
        xs - width / 2,
        vals_base,
        width,
        yerr=[err_low_base, err_high_base],
        capsize=3,
        color="#2e5984",
        label="All words (baseline)",
        error_kw=dict(lw=1.0, ecolor="#111111"),
    )
    bars2 = ax.bar(
        xs + width / 2,
        vals_clean,
        width,
        yerr=[err_low_clean, err_high_clean],
        capsize=3,
        color="#6baed6",
        label="Orthographic-clean",
        error_kw=dict(lw=1.0, ecolor="#111111"),
    )

    for b in bars1:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.0045, f"+{h:.4f}", ha="center", va="bottom", fontsize=7.2)
    for b in bars2:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.0045, f"+{h:.4f}", ha="center", va="bottom", fontsize=7.2, color="#1f4e78")

    ax.set_xticks(xs)
    ax.set_xticklabels([g[0] for g in groups], fontsize=8.2)
    ax.set_ylim(0.0, 0.056)
    ax.set_ylabel("Same-family minus cross-family\nmean WCS (self excluded)", fontsize=8.5)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", frameon=False, fontsize=7.8)
    fig.tight_layout()
    save_both(fig, out_dir, "fig7_same_family")


def main():
    json_path = Path("results/reports/revised_nov2_recomputed.json")
    data = json.loads(json_path.read_text())
    out_dir = Path("overleaf/figures_nov2")
    draw_fig1_pipeline(out_dir)
    draw_fig2_tier_hierarchy(data, out_dir)
    draw_fig3_filter_families(out_dir)
    draw_fig4_filter_sensitivity(data, out_dir)
    draw_fig5_family_decomposed(data, out_dir)
    draw_fig6_support_expansion(data, out_dir)
    draw_fig7_same_family(data, out_dir)
    print(f"Generated all 7 figures in {out_dir}")


if __name__ == "__main__":
    main()
