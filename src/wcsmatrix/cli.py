from __future__ import annotations

import argparse

from .io import load_config


def main():
    parser = argparse.ArgumentParser(description="Reproduce and extend the WCS scorer/source matrix")
    parser.add_argument("--config", default="configs/reproduction.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="Download pinned HC3 data and freeze question selection")
    p = sub.add_parser("frontier", help="Collect frontier-source answers via OpenRouter")
    p.add_argument("--env-file")
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--max-cost-usd", type=float, default=20)
    p = sub.add_parser("generate", help="Generate one local model at a time")
    p.add_argument("--model")
    p.add_argument("--limit", type=int)
    p = sub.add_parser("audit", help="Teacher-force all source texts with one scorer at a time")
    p.add_argument("--model")
    p.add_argument("--source", action="append", help="Restrict audit to specific source name(s)")
    p.add_argument("--limit", type=int)
    sub.add_parser("rescore", help="Recompute WCS from compressed token audits, without GPU")
    p = sub.add_parser("analyze", help="Paired bootstrap, heatmaps, paper comparisons")
    p.add_argument("--allow-partial", action="store_true")
    sub.add_parser("preflight", help="Check model access, GPU memory, and installed software")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "prepare":
        from .data import prepare

        prepare(config)
    elif args.command == "frontier":
        from .frontier import collect

        collect(config, args.env_file, args.limit, args.workers, args.max_cost_usd)
    elif args.command == "generate":
        from .local import generate

        generate(config, args.model, args.limit)
    elif args.command == "audit":
        from .local import audit

        audit(config, args.model, args.limit, args.source)
    elif args.command == "rescore":
        from .local import rescore

        rescore(config)
    elif args.command == "analyze":
        from .analysis import analyze

        analyze(config, args.allow_partial)
    elif args.command == "preflight":
        from .local import preflight

        preflight(config)


if __name__ == "__main__":
    main()
