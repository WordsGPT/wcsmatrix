import csv
import json
from pathlib import Path

import pytest

from wcsmatrix.analysis import analyze
from wcsmatrix.io import write_json, write_jsonl


def test_complete_matrix_prompt_pairing_and_paper_comparison(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = {
        "name": "test",
        "prompts": "prompts.jsonl",
        "human_texts": "human.jsonl",
        "local_texts": "local.jsonl",
        "frontier_texts": "frontier.jsonl",
        "models": [{"name": "a"}, {"name": "b"}],
        "frontier_models": [],
        "scoring": {"modes": ["realistic"], "top_p": [0.99]},
        "analysis": {"seed": 42, "bootstrap_replicates": 100},
        "source_group_contrasts": [
            {"name": "local_minus_human", "sources_a": ["a", "b"], "sources_b": ["human"]}
        ],
    }
    write_jsonl("prompts.jsonl", [{"id": "p1"}, {"id": "p2"}])
    write_json("docs/paper-targets.json", {"realistic": {"a": 0.5, "b": 0.5}})
    scores = []
    for s, fam in [("a", "x"), ("b", "y")]:
        for g, gfam in [("human", "human"), ("a", "x"), ("b", "y")]:
            for i, prompt in enumerate(["p2", "p1"]):
                scores.append(
                    {
                        "prompt_id": prompt,
                        "source": g,
                        "scorer": s,
                        "family": gfam,
                        "scorer_family": fam,
                        "kind": "human" if g == "human" else "local",
                        "mode": "realistic",
                        "p": 0.99,
                        "wcs": (0.3 if g == "human" else 0.7) + i * 0.1,
                    }
                )
    write_jsonl("results/scores/test.jsonl", scores)
    analyze(cfg)
    output = Path("results/reports/test")
    contrasts = list(csv.DictReader((output / "human-minus-source.csv").open()))
    assert len(contrasts) == 4
    assert all(float(r["delta"]) == pytest.approx(-0.4) for r in contrasts)
    assert all(float(r["ci_low"]) == pytest.approx(-0.4) for r in contrasts)
    groups = list(csv.DictReader((output / "source-group-contrasts.csv").open()))
    assert len(groups) == 2
    assert all(float(r["delta"]) == pytest.approx(0.4) for r in groups)
    assert all(float(r["ci_low"]) == pytest.approx(0.4) for r in groups)
    assert all(int(r["n"]) == 2 and int(r["sources_a"]) == 2 for r in groups)
    assert len(list(output.glob("matrix-*.png"))) == 1
    assert json.loads((output / "analysis-manifest.json").read_text())["partial"] is False
    write_jsonl("results/scores/test.jsonl", scores[:-1])
    with pytest.raises(ValueError, match="incomplete matrix"):
        analyze(cfg)
    analyze(cfg, allow_partial=True)
    assert json.loads((output / "analysis-manifest.json").read_text())["partial"] is True
    extra_source_row = {
        **scores[0],
        "source": "unconfigured-source",
        "scorer": "a",
    }
    write_jsonl("results/scores/test.jsonl", scores + [extra_source_row])
    analyze(cfg)
    assert json.loads((output / "analysis-manifest.json").read_text())["partial"] is False
