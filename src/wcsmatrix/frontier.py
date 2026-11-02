from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from dotenv import load_dotenv

from .io import append_jsonl, digest, now, read_jsonl, unique_rows, write_json, write_jsonl

CATALOG = "https://openrouter.ai/api/v1/models"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"


def collect(config, env_file=None, limit=None, workers=3, max_cost=20.0):
    if not config.get("frontier_collection_enabled", True):
        raise ValueError("This is an analysis-only combined config; collect using configs/open-frontier.yaml")
    if env_file:
        load_dotenv(env_file, override=False)
    else:
        load_dotenv()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("Set OPENROUTER_API_KEY or use --env-file; credentials are never copied")
    prompts = read_jsonl(config["prompts"])
    if not prompts:
        raise ValueError("No prompts; run prepare first")
    if limit is not None:
        prompts = prompts[:limit]
    models = config["frontier_models"]
    target = Path(config["frontier_texts"])
    existing = unique_rows(read_jsonl(target), ("prompt_id", "source"))
    done = {(r["prompt_id"], r["source"]): r for r in existing}
    with httpx.Client(timeout=60) as client:
        response = client.get(CATALOG)
        response.raise_for_status()
        catalog = response.json()
    lookup = {m["id"]: m for m in catalog["data"]}
    jobs, bound = [], 0.0
    for model in models:
        if model["id"] not in lookup:
            raise ValueError(f"Model unavailable: {model['id']}")
        info = lookup[model["id"]]
        params = model.get("parameters", {})
        for param in params:
            if param not in info["supported_parameters"]:
                raise ValueError(f"{model['id']} does not support {param}")
        for prompt in prompts:
            payload = {
                "model": model["id"],
                "messages": [{"role": "user", "content": prompt["prompt"]}],
                "max_tokens": config["frontier_max_tokens"],
                "provider": {"require_parameters": True},
                **params,
            }
            identity = digest({"payload": payload, "source": model["name"]})
            if model.get("metadata"):
                identity = digest(
                    {"payload": payload, "source": model["name"], "metadata": model["metadata"]}
                )
            old = done.get((prompt["id"], model["name"]))
            if old:
                if old["request_sha256"] != identity:
                    raise ValueError(
                        "Existing frontier output has different settings; choose a new output path"
                    )
                if old.get("complete", True):
                    continue
            price = info["pricing"]
            # UTF-8 bytes upper-bound ordinary prompt token count, plus chat overhead.
            bound += (len(prompt["prompt"].encode()) + 128) * float(price["prompt"]) + config[
                "frontier_max_tokens"
            ] * float(price["completion"])
            jobs.append((model, prompt, payload, identity))
    print(f"{len(jobs)} new API requests; conservative token-price bound ${bound:.2f}", flush=True)
    if bound > max_cost:
        raise ValueError(
            f"Estimated bound exceeds --max-cost-usd {max_cost}; use a smaller --limit or explicit larger cap"
        )
    write_json(
        config.get("frontier_catalog_path", "results/manifests/frontier-catalog.json"),
        {"retrieved_at": now(), "models": [lookup[m["id"]] for m in models]},
    )
    errors = []

    def run(job):
        model, prompt, payload, identity = job
        started = time.monotonic()
        for attempt in range(4):
            try:
                with httpx.Client(timeout=300) as client:
                    response = client.post(
                        ENDPOINT,
                        headers={"Authorization": "Bearer " + key, "X-Title": "WCS Matrix Research"},
                        json=payload,
                    )
                if response.status_code in {429, 500, 502, 503, 504} and attempt < 3:
                    time.sleep(2 ** (attempt + 1))
                    continue
                # Never include body or request headers in error messages.
                if response.status_code != 200:
                    raise RuntimeError(f"API HTTP {response.status_code}")
                raw = response.json()
                if raw.get("error"):
                    raise RuntimeError("API returned an error object")
                choice = raw["choices"][0]
                text = choice["message"].get("content")
                if not isinstance(text, str) or not text.strip():
                    raise RuntimeError("Empty visible answer")
                row = {
                    "prompt_id": prompt["id"],
                    "source": model["name"],
                    "family": model["family"],
                    "kind": "frontier",
                    "model_requested": model["id"],
                    "model_returned": raw.get("model"),
                    "provider": raw.get("provider"),
                    "text": text,
                    "text_sha256": digest(text),
                    "prompt_sha256": digest(prompt["prompt"]),
                    "request_sha256": identity,
                    "request": payload,
                    "response_id": raw.get("id"),
                    "created_at": now(),
                    "finish_reason": choice.get("finish_reason"),
                    "usage": raw.get("usage"),
                    "elapsed_seconds": time.monotonic() - started,
                    "complete": choice.get("finish_reason") == "stop",
                    "words": len(text.split()),
                }
                if model.get("metadata"):
                    row["source_metadata"] = model["metadata"]
                # Raw response may contain reasoning; kept local, not versioned.
                write_json(f"results/raw/frontier/{model['name']}/{prompt['id']}.json", raw)
                return row
            except (httpx.TimeoutException, httpx.NetworkError):
                # A timeout can have been billed; do not blindly retry an ambiguous completion.
                raise RuntimeError(
                    "Network/timeout error; request may have been billed; rerun to resume"
                ) from None
        raise RuntimeError("Retry exhaustion")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, job): job for job in jobs}
        for future in as_completed(futures):
            model, prompt, _, _ = futures[future]
            try:
                row = future.result()
                key_pair = (row["prompt_id"], row["source"])
                if key_pair in done:
                    done[key_pair] = row
                    write_jsonl(target, list(done.values()))
                else:
                    done[key_pair] = row
                    append_jsonl(target, row)
                print(
                    f"{model['name']} {prompt['id']}: {row['words']} words, {row['finish_reason']}",
                    flush=True,
                )
            except Exception as exc:
                error = {"model": model["id"], "prompt_id": prompt["id"], "error": str(exc), "at": now()}
                errors.append(error)
                append_jsonl("results/manifests/frontier-errors.jsonl", error)
                print(f"Failed {model['name']} {prompt['id']}: {exc}", flush=True)
    if errors:
        raise RuntimeError(f"{len(errors)} API requests failed; successful outputs retained; rerun to resume")
