import json

from serve_optimize.model_store import TINY_MODEL_IDS
from serve_optimize.modeling import infer_model_capability_metadata, infer_model_spec


def test_infer_known_model() -> None:
    spec = infer_model_spec("mistral-7b")
    assert spec.parameter_count_b == 7.3
    assert spec.family == "mistral"


def test_infer_unknown_parameter_count() -> None:
    spec = infer_model_spec("example/Llama-13B-test")
    assert spec.parameter_count_b == 13.0
    assert spec.family == "llama"


def test_tiny_model_defaults() -> None:
    assert TINY_MODEL_IDS == [
        "hf-internal-testing/tiny-random-gpt2",
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
    ]


def test_model_capability_metadata_reads_context_length(tmp_path) -> None:
    (tmp_path / "config.json").write_text(
        json.dumps({"max_position_embeddings": 40960, "torch_dtype": "bfloat16"}),
        encoding="utf-8",
    )

    metadata = infer_model_capability_metadata(str(tmp_path))

    assert metadata.max_context_tokens == 40960
    assert infer_model_spec("Qwen/Qwen3-0.6B").max_context_tokens == 40960
