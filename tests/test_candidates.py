from serve_optimize.candidates import generate_candidates
from serve_optimize.modeling import infer_model_spec
from serve_optimize.schemas import Goal, HardwareSnapshot


def test_generate_candidates_without_gpu_uses_smoke_backends() -> None:
    hardware = HardwareSnapshot.empty("host", "platform", "3.12", "test")
    model = infer_model_spec("tinyllama-1.1b")
    candidates = generate_candidates(hardware, model, goal=Goal.BALANCED, limit=5)
    assert candidates
    assert {candidate.backend for candidate in candidates} <= {"dry-run", "transformers"}
