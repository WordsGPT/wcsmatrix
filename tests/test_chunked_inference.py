"""Optional torch test: compare chunked teacher forcing against a full-context oracle."""

from types import SimpleNamespace

import numpy as np
import pytest

from wcsmatrix.local import token_audit

torch = pytest.importorskip("torch")


class Tokenizer:
    bos_token = "^"

    def __call__(self, text, **kwargs):
        return {
            "input_ids": [ord(c) % 32 for c in text],
            "offset_mapping": [(i, i + 1) for i in range(len(text))],
        }


class CumulativeModel:
    """Next-token logits depend on the entire past, exposing lost-cache and shift errors."""

    def __init__(self):
        self.embedding = torch.nn.Embedding(32, 2)

    def get_input_embeddings(self):
        return self.embedding

    def __call__(self, input_ids, past_key_values=None, **kwargs):
        cumulative = input_ids.cumsum(-1) + (past_key_values if past_key_values is not None else 0)
        logits = -torch.abs(torch.arange(32)[None, None, :] - (cumulative % 32)[..., None]).float() / 3
        return SimpleNamespace(logits=logits, past_key_values=cumulative[:, -1:])


def test_chunked_audit_matches_full_context_and_one_token_chunks():
    model, tokenizer = CumulativeModel(), Tokenizer()
    scoring = {
        "context_mode": "plain",
        "max_sequence_tokens": 100,
        "top_k": 10,
        "temperature": 1.0,
        "chunk_size": 3,
    }
    arrays = token_audit(model, tokenizer, "Q", "some text", {}, scoring)
    full = token_audit(model, tokenizer, "Q", "some text", {}, dict(scoring, chunk_size=100))
    single = token_audit(model, tokenizer, "Q", "some text", {}, dict(scoring, chunk_size=1))
    for field in ["top_probs", "top_ids", "target_top_index", "mass_before", "offsets", "target_token_ids"]:
        np.testing.assert_allclose(arrays[field], full[field])
        np.testing.assert_allclose(arrays[field], single[field])
    ids = torch.tensor([[ord(c) % 32 for c in "^Q\n\nsome text"]])
    logits = model(ids[:, :-1]).logits[0, 3:]
    probs = logits.softmax(-1)
    expected = probs.topk(10, sorted=True).values.numpy()
    np.testing.assert_allclose(arrays["top_probs"], expected, rtol=1e-6)


@pytest.mark.parametrize("model_type", ["qwen3_5_text", "gemma4_text", "gemma4_unified_text"])
def test_real_transformers_architectures_match_unchunked(model_type):
    transformers = pytest.importorskip("transformers")
    if model_type not in transformers.CONFIG_MAPPING:
        pytest.skip("Requires the locked Transformers version")
    common = dict(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    if model_type == "qwen3_5_text":
        common.update(
            layer_types=["linear_attention", "full_attention"],
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_conv_kernel_dim=4,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 1.0,
                "mrope_section": [1, 1, 2],
                "mrope_interleaved": True,
            },
        )
    else:
        common.update(
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=16,
            num_kv_shared_layers=0,
            per_layer_config={},
            vocab_size_per_layer_input=32,
            hidden_size_per_layer_input=4 if model_type == "gemma4_text" else 0,
            rope_parameters={
                "full_attention": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": 1.0,
                },
                "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            },
        )
    cfg = transformers.CONFIG_MAPPING[model_type](**common)
    torch.manual_seed(7)
    model = transformers.AutoModelForCausalLM.from_config(cfg, attn_implementation="sdpa").eval()
    scoring = {
        "context_mode": "plain",
        "max_sequence_tokens": 100,
        "top_k": 10,
        "temperature": 1.0,
        "chunk_size": 3,
    }
    chunked = token_audit(model, Tokenizer(), "Question", "some more text here", {}, scoring)
    full = token_audit(
        model, Tokenizer(), "Question", "some more text here", {}, dict(scoring, chunk_size=100)
    )
    np.testing.assert_allclose(chunked["top_probs"], full["top_probs"], atol=2e-6, rtol=2e-5)
    np.testing.assert_array_equal(chunked["top_ids"], full["top_ids"])
    np.testing.assert_array_equal(chunked["target_top_index"], full["target_top_index"])
    np.testing.assert_allclose(chunked["mass_before"], full["mass_before"], atol=2e-6, rtol=2e-5)


def test_multi_device_sharded_audit_matches_single_device():
    accelerate = pytest.importorskip("accelerate")
    transformers = pytest.importorskip("transformers")
    if "gemma4_text" not in transformers.CONFIG_MAPPING or not torch.cuda.is_available():
        pytest.skip("Requires CUDA and locked Transformers version")
    cfg = transformers.CONFIG_MAPPING["gemma4_text"](
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=16,
        num_kv_shared_layers=0,
        per_layer_config={},
        vocab_size_per_layer_input=32,
        hidden_size_per_layer_input=0,
        rope_parameters={
            "full_attention": {"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 1.0},
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        },
    )
    torch.manual_seed(7)
    single_gpu = transformers.AutoModelForCausalLM.from_config(cfg, attn_implementation="eager").eval().to("cuda:0")
    torch.manual_seed(7)
    sharded = transformers.AutoModelForCausalLM.from_config(cfg, attn_implementation="eager").eval()
    sharded.tie_weights()
    sharded = accelerate.dispatch_model(
        sharded,
        device_map={
            "model.embed_tokens": 0,
            "lm_head": 0,
            "model.layers.0": 0,
            "model.layers.1": "cpu",
            "model.norm": "cpu",
            "model.rotary_emb": "cpu",
        },
    )
    scoring = {
        "context_mode": "plain",
        "max_sequence_tokens": 100,
        "top_k": 10,
        "temperature": 1.0,
        "chunk_size": 3,
    }
    a1 = token_audit(single_gpu, Tokenizer(), "Question", "some more text here", {}, scoring)
    a2 = token_audit(sharded, Tokenizer(), "Question", "some more text here", {}, scoring)
    for k in ["top_probs", "top_ids", "target_top_index", "mass_before"]:
        np.testing.assert_allclose(a1[k], a2[k], atol=1e-6)

