from serve_optimize.model_store import TINY_MODEL_IDS
from serve_optimize.modeling import infer_model_spec


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
