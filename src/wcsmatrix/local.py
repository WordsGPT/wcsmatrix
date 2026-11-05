from __future__ import annotations

import gc
import importlib.metadata
import os
import platform
from pathlib import Path

import numpy as np

from .io import (
    append_jsonl,
    digest,
    evaluation_prompts,
    now,
    read_jsonl,
    unique_rows,
    write_json,
    write_jsonl,
)
from .metrics import coverage, word_groups


def _hf_token():
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        val = os.environ.get(var, "").strip()
        if val:
            return val
    for path in (Path(".hf_token"), Path.home() / ".cache" / "huggingface" / "token"):
        if path.exists():
            val = path.read_text(encoding="utf-8", errors="ignore").strip()
            if val:
                return val
    return None


def selected_models(config, name):
    models = [m for m in config["models"] if name is None or m["name"] == name]
    if not models:
        raise ValueError(f"Unknown model {name}")
    return models


def load_model(spec, hardware):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoTokenizer, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; install the gpu extra on the server")
    num_gpus = torch.cuda.device_count()
    gpu_info = [torch.cuda.mem_get_info(i) for i in range(num_gpus)]
    free = sum(f for f, _ in gpu_info)
    limit = min(float(hardware["vram_limit_gib"]), free / 2**30)
    estimate = spec["estimated_parameters_b"] * (1.02 if spec["precision"] == "int8" else 1.98) / 1.073741824
    reserve = min(float(hardware["reserve_gib"]), max(1.0, limit - estimate - 0.2))
    weight_budget = limit - reserve
    if weight_budget < estimate:
        raise RuntimeError(
            f"{spec['name']} estimated weights {estimate:.1f} GiB exceed {weight_budget:.1f} GiB budget"
        )
    free0, total0 = gpu_info[0]
    single_limit = min(float(hardware["vram_limit_gib"]), free0 / 2**30)
    single_reserve = min(float(hardware["reserve_gib"]), max(1.0, single_limit - estimate - 0.2))
    single_budget = single_limit - single_reserve
    if num_gpus == 1 or single_budget >= estimate:
        torch.cuda.set_per_process_memory_fraction(min(single_limit * 2**30 / total0, 1.0), 0)
        gpu_max_memory = {0: f"{max(int(single_budget), int(np.ceil(estimate) + 1))}GiB"}
    else:
        per_gpu_reserve = max(2.0, reserve / num_gpus)
        balanced_cap = int(np.ceil(estimate / num_gpus) + 5)
        gpu_max_memory = {}
        for i, (free_i, total_i) in enumerate(gpu_info):
            limit_i = min(free_i / 2**30, float(hardware["vram_limit_gib"]) / num_gpus)
            torch.cuda.set_per_process_memory_fraction(min(limit_i * 2**30 / total_i, 1.0), i)
            budget_i = max(1, min(int(limit_i - per_gpu_reserve), balanced_cap))
            gpu_max_memory[i] = f"{budget_i}GiB"
    tok = _hf_token()
    try:
        tokenizer = AutoTokenizer.from_pretrained(spec["id"], revision=spec["revision"], use_fast=True, token=tok)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            spec["id"], revision=spec["revision"], use_fast=True, local_files_only=True, token=tok
        )
    if not tokenizer.is_fast:
        raise ValueError("Fast tokenizer with offsets is required")
    cls = AutoModelForMultimodalLM if spec.get("loader") == "multimodal" else AutoModelForCausalLM
    attn = "eager" if spec.get("family") == "gemma" else "sdpa"
    kwargs = dict(
        revision=spec["revision"],
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory={**gpu_max_memory, "cpu": f"{hardware['cpu_memory_gib']}GiB"},
        attn_implementation=attn,
        trust_remote_code=False,
        token=tok,
    )
    if spec["precision"] == "int8":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif spec["precision"] != "bf16":
        raise ValueError("Only paper precisions bf16 and int8 are supported")
    try:
        model = cls.from_pretrained(spec["id"], **kwargs).eval()
    except Exception:
        model = cls.from_pretrained(spec["id"], local_files_only=True, **kwargs).eval()
    if any(str(device) in {"cpu", "disk"} for device in getattr(model, "hf_device_map", {}).values()):
        raise RuntimeError("Model spilled to CPU/disk; increase budget or select smaller model")
    return model, tokenizer


def unload(model, tokenizer):
    del model, tokenizer
    gc.collect()
    import torch

    torch.cuda.empty_cache()


def prompt_prefix(tokenizer, prompt, spec, context_mode="chat"):
    if context_mode == "chat":
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            **spec.get("chat_kwargs", {}),
        )
    if context_mode == "plain":
        return (tokenizer.bos_token or "") + prompt + "\n\n"
    if context_mode == "text_only":
        if tokenizer.bos_token is None:
            raise ValueError("text_only requires a BOS token to score the first word")
        return tokenizer.bos_token
    raise ValueError(f"Unknown context mode {context_mode}")


def generate(config, name=None, limit=None):
    import torch

    prompts = read_jsonl(config["prompts"])
    if not prompts:
        raise ValueError("Run prepare first")
    if limit is not None:
        prompts = prompts[:limit]
    target = config["local_texts"]
    existing = unique_rows(read_jsonl(target), ("prompt_id", "source"))
    done = {(r["prompt_id"], r["source"]): r for r in existing}
    for spec in selected_models(config, name):
        pending = []
        for prompt in prompts:
            identity = digest({"model": spec, "prompt": prompt["prompt"], "generation": config["generation"]})
            old = done.get((prompt["id"], spec["name"]))
            if old:
                if old["request_sha256"] != identity:
                    raise ValueError("Changed generation configuration; use a new local_texts path")
            else:
                pending.append((prompt, identity))
        if not pending:
            continue
        model, tokenizer = load_model(spec, config["hardware"])
        try:
            for prompt, identity in pending:
                sample_seed = config["dataset"]["seed"] + int(digest(prompt["id"])[:8], 16)
                torch.manual_seed(sample_seed)
                prefix = prompt_prefix(tokenizer, prompt["prompt"], spec)
                input_device = model.get_input_embeddings().weight.device
                encoded = tokenizer(prefix, add_special_tokens=False, return_tensors="pt").to(input_device)
                if (
                    encoded.input_ids.shape[1] + config["generation"]["max_new_tokens"]
                    > config["scoring"]["max_sequence_tokens"]
                ):
                    raise ValueError("Generation would exceed configured context cap")
                options = dict(config["generation"])
                if options.get("do_sample", False):
                    options.setdefault("temperature", 1.0)
                    options.setdefault("top_p", 0.95)
                else:
                    options.update(temperature=None, top_p=None, top_k=None)
                with torch.inference_mode():
                    output = model.generate(**encoded, **options)
                answer_ids = output[0, encoded.input_ids.shape[1] :].tolist()
                text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
                if not text:
                    raise ValueError("Local model generated an empty answer")
                eos = model.generation_config.eos_token_id
                eos_ids = set(eos if isinstance(eos, list) else [eos])
                row = {
                    "prompt_id": prompt["id"],
                    "source": spec["name"],
                    "family": spec["family"],
                    "kind": "local",
                    "text": text,
                    "text_sha256": digest(text),
                    "prompt_sha256": digest(prompt["prompt"]),
                    "request_sha256": identity,
                    "model_spec": spec,
                    "generation": config["generation"],
                    "sample_seed": sample_seed,
                    "generated_tokens": len(answer_ids),
                    "complete": answer_ids[-1] in eos_ids,
                    "created_at": now(),
                }
                append_jsonl(target, row)
                print(f"Generated {spec['name']} {prompt['id']}", flush=True)
        finally:
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()


def source_texts(config):
    rows = []
    for field in ("human_texts", "local_texts", "frontier_texts"):
        rows.extend(read_jsonl(config[field]))
    for path in config.get("additional_texts", []):
        rows.extend(read_jsonl(path))
    unique_rows(rows, ("prompt_id", "source"))
    return rows


def encode_text(tokenizer, prefix, text):
    encoded = tokenizer(prefix + text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    selected = []
    relative = []
    boundary = len(prefix)
    for index, (a, b) in enumerate(offsets):
        if a < boundary < b:
            if not (prefix + text)[boundary:b].isspace():
                raise ValueError(
                    "Tokenizer merged across prompt/answer boundary; choose a prefix ending in a special token"
                )
            continue
        if a >= boundary and b > a:
            if index == 0:
                raise ValueError("First text token has no preceding context")
            selected.append(index)
            relative.append((a - boundary, b - boundary))
    if not selected:
        raise ValueError("No answer tokens")
    return encoded["input_ids"], np.array(selected), np.array(relative, dtype=np.int32)


def token_audit(model, tokenizer, prompt, text, spec, scoring):
    import torch

    prefix = prompt_prefix(tokenizer, prompt, spec, scoring["context_mode"])
    ids, selected, offsets = encode_text(tokenizer, prefix, text)
    if len(ids) > scoring["max_sequence_tokens"]:
        raise ValueError(f"Text needs {len(ids)} tokens; exceeds context cap. No silent truncation.")
    if scoring.get("temperature", 1.0) <= 0:
        raise ValueError("Scoring temperature must be positive")
    selected_set = set(selected.tolist())
    device = model.get_input_embeddings().weight.device
    top_values, top_ids, target_indices, full_mass_before = [], [], [], []
    past = None
    chunk = scoring["chunk_size"]
    with torch.inference_mode():
        for start in range(0, len(ids) - 1, chunk):
            end = min(start + chunk, len(ids) - 1)
            inputs = torch.tensor([ids[start:end]], dtype=torch.long, device=device)
            output = model(
                input_ids=inputs,
                past_key_values=past,
                use_cache=True,
                attention_mask=torch.ones((1, end), dtype=torch.long, device=device),
            )
            past = output.past_key_values
            if past is None:
                raise RuntimeError("Model did not return KV cache; chunked scoring would lose context")
            positions = [i for i in range(start + 1, end + 1) if i in selected_set]
            if positions:
                out_device = output.logits.device
                local_positions = torch.tensor([i - start - 1 for i in positions], device=out_device)
                logits = output.logits[0, local_positions].float() / scoring.get("temperature", 1.0)
                probs = torch.softmax(logits, dim=-1)
                k = min(scoring["top_k"], probs.shape[-1])
                values, indices = torch.topk(probs, k=k, dim=-1, sorted=True)
                observed = torch.tensor([ids[i] for i in positions], device=out_device)
                matches = indices == observed[:, None]
                where = torch.where(matches.any(-1), matches.int().argmax(-1), -1)
                actual = probs.gather(1, observed[:, None])
                before = torch.where(probs > actual, probs, 0).sum(-1)
                top_values.append(values.cpu().numpy())
                top_ids.append(indices.cpu().numpy().astype(np.int32))
                target_indices.append(where.cpu().numpy().astype(np.int16))
                full_mass_before.append(before.cpu().numpy())
                del logits, probs, values, indices, matches, where, actual, before
            del output, inputs
    del past
    return dict(
        top_probs=np.concatenate(top_values),
        top_ids=np.concatenate(top_ids),
        target_top_index=np.concatenate(target_indices),
        mass_before=np.concatenate(full_mass_before),
        offsets=offsets,
        target_token_ids=np.array([ids[i] for i in selected], dtype=np.int32),
        context_tokens=np.array(len(ids)),
        prefix=np.array(prefix),
    )


def audit(config, name=None, limit=None, sources=None):
    import torch

    prompts = {r["id"]: r for r in evaluation_prompts(config)}
    texts = source_texts(config)
    known_ids = {r["id"] for r in read_jsonl(config["prompts"])}
    if {r["prompt_id"] for r in texts} - known_ids:
        raise ValueError("Source texts contain prompts outside the frozen manifest")
    texts = [r for r in texts if r["prompt_id"] in prompts]
    if sources:
        allowed_sources = {sources} if isinstance(sources, str) else set(sources)
        texts = [r for r in texts if r["source"] in allowed_sources]
    if not texts:
        raise ValueError("No source texts; run prepare/generate/frontier first")
    if limit is not None:
        allowed = set(list(prompts)[:limit])
        texts = [r for r in texts if r["prompt_id"] in allowed]
    for spec in selected_models(config, name):
        model, tokenizer = None, None
        for row in texts:
            if row.get("complete") is False:
                print(f"Excluded incomplete answer {row['source']} {row['prompt_id']}", flush=True)
                continue
            prompt = prompts[row["prompt_id"]]
            if row["prompt_sha256"] != digest(prompt["prompt"]):
                raise ValueError("Source text does not match this prompt manifest")
            scorer_settings = {
                k: v for k, v in config["scoring"].items() if k not in {"modes", "top_p", "word_policy"}
            }
            identity = digest(
                {"model": spec, "scoring": scorer_settings, "text": row["text"], "prompt": prompt["prompt"]}
            )
            root = Path("results/audits") / config["name"] / spec["name"] / row["source"]
            path = root / (row["prompt_id"] + ".npz")
            meta_path = path.with_suffix(".json")
            if path.exists() and meta_path.exists():
                import json

                if json.loads(meta_path.read_text())["audit_sha256"] != identity:
                    raise ValueError("Audit identity changed; use a new experiment name")
                continue
            if model is None:
                model, tokenizer = load_model(spec, config["hardware"])
            arrays = token_audit(model, tokenizer, prompt["prompt"], row["text"], spec, config["scoring"])
            root.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, **arrays)
            tmp.replace(path)
            write_json(
                meta_path,
                {
                    "audit_sha256": identity,
                    "prompt_id": row["prompt_id"],
                    "source": row["source"],
                    "family": row["family"],
                    "kind": row["kind"],
                    "scorer": spec["name"],
                    "scorer_family": spec["family"],
                    "model_spec": spec,
                    "text_sha256": digest(row["text"]),
                    "text": row["text"],
                    "scoring": scorer_settings,
                    "created_at": now(),
                    "context_tokens": int(arrays["context_tokens"]),
                    "top100_miss_fraction": float(np.mean(arrays["target_top_index"] < 0)),
                },
            )
            print(f"Audited {spec['name']} <- {row['source']} {row['prompt_id']}", flush=True)
        if model is not None:
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
    rescore(config)


def rescore(config):
    import json

    rows = []
    specs = {m["name"]: m for m in config["models"]}
    prompts = {p["id"]: p for p in evaluation_prompts(config)}
    texts = {(r["prompt_id"], r["source"]): r for r in source_texts(config)}
    scoring_settings = {
        k: v for k, v in config["scoring"].items() if k not in {"modes", "top_p", "word_policy"}
    }
    for path in sorted((Path("results/audits") / config["name"]).glob("*/*/*.npz")):
        meta = json.loads(path.with_suffix(".json").read_text())
        source = texts.get((meta["prompt_id"], meta["source"]))
        if source is None or meta["scorer"] not in specs or meta["prompt_id"] not in prompts:
            raise ValueError("Audit no longer belongs to configured sources/scorers/prompts")
        expected = digest(
            {
                "model": specs[meta["scorer"]],
                "scoring": scoring_settings,
                "text": source["text"],
                "prompt": prompts[meta["prompt_id"]]["prompt"],
            }
        )
        if meta["audit_sha256"] != expected:
            raise ValueError("Inference identity changed; use a new experiment name and rerun audits")
        with np.load(path, allow_pickle=False) as arrays:
            top_probs = arrays["top_probs"]
            target_top_index = arrays["target_top_index"]
            mass_before = arrays["mass_before"]
            offsets = arrays["offsets"].tolist()
        groups = word_groups(meta["text"], offsets, config["scoring"]["word_policy"])
        for mode in config["scoring"]["modes"]:
            for p in config["scoring"]["top_p"]:
                value = coverage(
                    top_probs,
                    target_top_index,
                    groups,
                    p,
                    mode,
                    mass_before,
                )
                rows.append(
                    {
                        k: meta[k]
                        for k in ("prompt_id", "source", "family", "kind", "scorer", "scorer_family")
                    }
                    | {
                        "mode": mode,
                        "p": p,
                        "word_policy": config["scoring"]["word_policy"],
                        "audit_sha256": meta["audit_sha256"],
                        **value,
                    }
                )
    score_path = Path("results/scores") / (config["name"] + ".jsonl")
    allowed_modes = set(config["scoring"]["modes"])
    allowed_ps = {float(x) for x in config["scoring"]["top_p"]}
    allowed_policy = config["scoring"]["word_policy"]
    existing = {
        (r["prompt_id"], r["source"], r["scorer"], r["mode"], float(r["p"])): r
        for r in read_jsonl(score_path)
        if (r["prompt_id"], r["source"]) in texts
        and r["scorer"] in specs
        and r["prompt_id"] in prompts
        and r.get("mode") in allowed_modes
        and float(r.get("p", -1)) in allowed_ps
        and r.get("word_policy") == allowed_policy
    }
    for r in rows:
        existing[(r["prompt_id"], r["source"], r["scorer"], r["mode"], float(r["p"]))] = r
    merged = sorted(
        existing.values(),
        key=lambda r: (r["scorer"], r["source"], r["prompt_id"], r["mode"], r["p"]),
    )
    write_jsonl(score_path, merged)
    print(f"Wrote {len(merged)} per-prompt scores")


def preflight(config):
    import httpx
    import torch
    from transformers import AutoConfig

    report = {
        "at": now(),
        "python": platform.python_version(),
        "config": config,
        "packages": {
            p: importlib.metadata.version(p) for p in ["torch", "transformers", "accelerate", "bitsandbytes"]
        },
        "cuda_available": torch.cuda.is_available(),
        "models": [],
    }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    num_gpus = torch.cuda.device_count()
    gpu_info = [torch.cuda.mem_get_info(i) for i in range(num_gpus)]
    free = sum(f for f, _ in gpu_info)
    total = sum(t for _, t in gpu_info)
    report["gpu"] = {
        "name": " + ".join(torch.cuda.get_device_name(i) for i in range(num_gpus)),
        "count": num_gpus,
        "total_gib": total / 2**30,
        "free_gib": free / 2**30,
        "configured_limit_gib": config["hardware"]["vram_limit_gib"],
    }
    tok = _hf_token()
    headers = {"Authorization": f"Bearer {tok}"} if tok else None
    errors = []
    for spec in config["models"]:
        try:
            try:
                cfg = AutoConfig.from_pretrained(
                    spec["id"], revision=spec["revision"], trust_remote_code=False, token=tok
                )
            except Exception:
                cfg = AutoConfig.from_pretrained(
                    spec["id"], revision=spec["revision"], trust_remote_code=False, local_files_only=True, token=tok
                )
            info = {"name": spec["name"], "architecture": cfg.architectures, "model_type": cfg.model_type}
            # Config access verifies pinned revision and gated-repository authorization.
            response = httpx.get(
                f"https://huggingface.co/api/models/{spec['id']}/revision/{spec['revision']}",
                headers=headers,
                timeout=30,
            )
            info["metadata_http_status"] = response.status_code
            budget = (
                min(config["hardware"]["vram_limit_gib"], free / 2**30) - config["hardware"]["reserve_gib"]
            )
            estimate = (
                spec["estimated_parameters_b"] * (1.15 if spec["precision"] == "int8" else 2.05) / 1.073741824
            )
            info["estimated_weights_gib"] = estimate
            if estimate > budget:
                raise ValueError(f"Estimated weights {estimate:.1f} GiB exceed {budget:.1f} GiB available")
            report["models"].append(info)
        except Exception as exc:
            errors.append({"name": spec["name"], "type": type(exc).__name__})
    report["errors"] = errors
    write_json("results/manifests/preflight.json", report)
    print(report["gpu"])
    if errors:
        raise RuntimeError(
            f"Model preflight failures: {errors}; inspect revisions, access, and memory budget"
        )
    print("Preflight passed; actual generation/scoring smoke test still required")
