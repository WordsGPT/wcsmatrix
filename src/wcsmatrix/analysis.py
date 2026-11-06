from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .io import digest, evaluation_prompts, read_jsonl, unique_rows, write_json
from .local import source_texts
from .metrics import paired_bootstrap


def csv_file(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collection_report(config, directory):
    prompts = read_jsonl(config["prompts"])
    texts = source_texts(config)
    configured_sources = (
        {"human"}
        | {m["name"] for m in config.get("models", [])}
        | {m["name"] for m in config.get("frontier_models", [])}
    )
    if configured_sources - {"human"}:
        texts = [r for r in texts if r["source"] in configured_sources]
    expected = {p["id"] for p in prompts}
    grouped = defaultdict(list)
    for row in texts:
        grouped[row["source"]].append(row)
    report = []
    for source, rows in sorted(grouped.items()):
        lengths = [len(r["text"].split()) for r in rows]
        complete = [r for r in rows if r.get("complete", True)]
        costs = [r.get("usage", {}).get("cost") for r in rows if r.get("usage")]
        cost = sum(float(c) for c in costs if c is not None)
        report.append(
            {
                "source": source,
                "kind": rows[0]["kind"],
                "answers": len(rows),
                "complete_answers": len(complete),
                "expected_answers": len(expected),
                "mean_words": float(np.mean(lengths)),
                "min_words": min(lengths),
                "max_words": max(lengths),
                "outside_150_400": sum(n < 150 or n > 400 for n in lengths),
                "reported_cost_usd": cost,
                "missing_prompt_ids": sorted(expected - {r["prompt_id"] for r in rows}),
            }
        )
    write_json(directory / "collection.json", report)
    csv_file(
        directory / "collection.csv",
        [{k: v for k, v in r.items() if k != "missing_prompt_ids"} for r in report],
    )
    return report


def analyze(config, allow_partial=False):
    directory = Path("results/reports") / config["name"]
    directory.mkdir(parents=True, exist_ok=True)
    collection_report(config, directory)
    scores = unique_rows(
        read_jsonl(Path("results/scores") / (config["name"] + ".jsonl")),
        ("prompt_id", "source", "scorer", "mode", "p"),
    )
    if not scores:
        print(f"Collection report: {directory}; WCS not yet measured (run server audits)")
        return
    prompts = {p["id"] for p in evaluation_prompts(config)}
    sources = (
        ["human"] + [m["name"] for m in config["models"]] + [m["name"] for m in config["frontier_models"]]
    )
    scorers = [m["name"] for m in config["models"]]
    expected = {
        (s, g, mode, p)
        for s in scorers
        for g in sources
        for mode in config["scoring"]["modes"]
        for p in config["scoring"]["top_p"]
    }
    grouped = defaultdict(dict)
    metadata = {}
    for row in scores:
        if row["prompt_id"] not in prompts:
            raise ValueError("Score contains prompt outside manifest")
        if row["source"] not in sources or row["scorer"] not in scorers:
            continue
        key = (row["scorer"], row["source"], row["mode"], row["p"])
        if key not in expected:
            raise ValueError(f"Unexpected score cell {key}")
        grouped[key][row["prompt_id"]] = row["wcs"]
        metadata[key] = row
    missing = [
        {"scorer": s, "source": g, "mode": mode, "p": p, "n": len(grouped[(s, g, mode, p)])}
        for s, g, mode, p in sorted(expected)
        if set(grouped[(s, g, mode, p)]) != prompts
    ]
    write_json(
        directory / "completeness.json",
        {
            "expected_prompts": len(prompts),
            "incomplete_cells": missing,
            "evaluation_exclusions": config.get("evaluation_exclusions", {}),
        },
    )
    if missing and not allow_partial:
        raise ValueError(
            f"{len(missing)} incomplete matrix cells. Finish collection/scoring or use --allow-partial explicitly."
        )
    cells, contrasts = [], []
    seed = config["analysis"]["seed"]
    reps = config["analysis"]["bootstrap_replicates"]
    for (s, g, mode, p), values in sorted(grouped.items()):
        if not values:
            continue
        row = metadata[(s, g, mode, p)]
        relation = (
            "human"
            if g == "human"
            else (
                "self"
                if s == g
                else ("same_family" if row["family"] == row["scorer_family"] else "cross_family")
            )
        )
        cells.append(
            {
                "scorer": s,
                "source": g,
                "mode": mode,
                "p": p,
                "wcs": float(np.mean(list(values.values()))),
                "n": len(values),
                "relation": relation,
            }
        )
        if g != "human":
            human = grouped[(s, "human", mode, p)]
            ids = sorted(set(human) & set(values))
            if ids:
                contrasts.append(
                    {
                        "scorer": s,
                        "source": g,
                        "mode": mode,
                        "p": p,
                        **paired_bootstrap([human[i] for i in ids], [values[i] for i in ids], seed, reps),
                    }
                )
    csv_file(directory / "matrix.csv", cells)
    csv_file(directory / "human-minus-source.csv", contrasts)
    group_contrasts = []
    for contrast in config.get("source_group_contrasts", []):
        a_sources, b_sources = contrast["sources_a"], contrast["sources_b"]
        if not a_sources or not b_sources or set(a_sources + b_sources) - set(sources):
            raise ValueError("Source group contrasts must reference configured non-empty source groups")
        for scorer in scorers:
            for mode in config["scoring"]["modes"]:
                for p in config["scoring"]["top_p"]:
                    a_values = [grouped[(scorer, source, mode, p)] for source in a_sources]
                    b_values = [grouped[(scorer, source, mode, p)] for source in b_sources]
                    ids = sorted(set.intersection(*(set(v) for v in a_values + b_values)))
                    if not ids:
                        continue
                    a = [float(np.mean([v[i] for v in a_values])) for i in ids]
                    b = [float(np.mean([v[i] for v in b_values])) for i in ids]
                    group_contrasts.append(
                        {
                            "contrast": contrast["name"],
                            "scorer": scorer,
                            "mode": mode,
                            "p": p,
                            "sources_a": len(a_sources),
                            "sources_b": len(b_sources),
                            "mean_a": float(np.mean(a)),
                            "mean_b": float(np.mean(b)),
                            **paired_bootstrap(a, b, seed, reps),
                        }
                    )
    csv_file(directory / "source-group-contrasts.csv", group_contrasts)
    # Equal weight to directional cells, then paired resampling of prompt-level means.
    family_contrasts = []
    for mode in config["scoring"]["modes"]:
        for p in config["scoring"]["top_p"]:
            same, cross = [], []
            for cell in cells:
                if cell["mode"] != mode or cell["p"] != p:
                    continue
                meta = metadata[(cell["scorer"], cell["source"], mode, p)]
                if meta["kind"] != "local":
                    continue
                values = grouped[(cell["scorer"], cell["source"], mode, p)]
                if cell["relation"] == "same_family":
                    same.append(values)
                elif cell["relation"] == "cross_family":
                    cross.append(values)
            if same and cross:
                ids = sorted(set.intersection(*(set(v) for v in same + cross)))
                if ids:
                    family_contrasts.append(
                        {
                            "mode": mode,
                            "p": p,
                            "same_family_cells": len(same),
                            "cross_family_cells": len(cross),
                            **paired_bootstrap(
                                [np.mean([v[i] for v in same]) for i in ids],
                                [np.mean([v[i] for v in cross]) for i in ids],
                                seed,
                                reps,
                            ),
                        }
                    )
    csv_file(directory / "same-minus-cross-family.csv", family_contrasts)
    direction = []
    for a_index, a in enumerate(scorers):
        for b in scorers[a_index + 1 :]:
            for mode in config["scoring"]["modes"]:
                for p in config["scoring"]["top_p"]:
                    ab, ba = grouped[(a, b, mode, p)], grouped[(b, a, mode, p)]
                    ids = sorted(set(ab) & set(ba))
                    if ids:
                        direction.append(
                            {
                                "scorer_a": a,
                                "scorer_b": b,
                                "mode": mode,
                                "p": p,
                                **paired_bootstrap([ab[i] for i in ids], [ba[i] for i in ids], seed, reps),
                            }
                        )
    csv_file(directory / "directional-asymmetry.csv", direction)
    paper = json.loads(Path("docs/paper-targets.json").read_text())
    comparisons = []
    for cell in cells:
        if cell["source"] == "human" and cell["p"] == 0.99 and cell["mode"] in paper:
            target = paper[cell["mode"]].get(cell["scorer"])
            if target is not None:
                comparisons.append(
                    {**cell, "paper_rounded_wcs": target, "difference_from_paper": cell["wcs"] - target}
                )
    csv_file(directory / "paper-comparison.csv", comparisons)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for mode in config["scoring"]["modes"]:
        for p in config["scoring"]["top_p"]:
            matrix = np.full((len(sources), len(scorers)), np.nan)
            for row in cells:
                if row["mode"] == mode and row["p"] == p:
                    matrix[sources.index(row["source"]), scorers.index(row["scorer"])] = row["wcs"]
            fig, ax = plt.subplots(figsize=(11, max(5, len(sources) * 0.5)))
            im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(scorers)), scorers, rotation=40, ha="right")
            ax.set_yticks(range(len(sources)), sources)
            ax.set(
                xlabel="Scorer",
                ylabel="Text source",
                title=f"WCS {mode}, p={p}" + (" (partial)" if missing else ""),
            )
            for y, x in np.ndindex(matrix.shape):
                if np.isfinite(matrix[y, x]):
                    ax.text(
                        x,
                        y,
                        f"{matrix[y, x]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="black" if matrix[y, x] > 0.6 else "white",
                    )
            fig.colorbar(im, ax=ax, label="Mean per-prompt word coverage")
            fig.tight_layout()
            fig.savefig(directory / f"matrix-{mode}-{p}.png", dpi=180)
            plt.close(fig)
    write_json(
        directory / "analysis-manifest.json",
        {
            "config_sha256": digest(config),
            "score_rows": len(scores),
            "partial": bool(missing),
            "evaluation_exclusions": config.get("evaluation_exclusions", {}),
            "bootstrap_replicates": reps,
            "seed": seed,
            "intervals": "pointwise paired percentile bootstrap; no multiplicity correction",
        },
    )
    print(f"Analysis written to {directory}")
