"""TensorRT-LLM adapter placeholder.

TensorRT-LLM requires an engine build step, so this adapter starts as a launch
plan placeholder instead of pretending a generic command is enough.
"""

from __future__ import annotations

from serve_optimize.backends.base import LaunchPlan
from serve_optimize.schemas import ServingConfig


class TrtLlmAdapter:
    name = "trt-llm"

    def is_available(self) -> bool:
        return False

    def build_launch_plan(self, config: ServingConfig) -> LaunchPlan:
        return LaunchPlan(
            command=[],
            environment={},
            notes=[
                "TensorRT-LLM support requires model conversion, engine build, and server launch steps.",
                f"Requested config id: {config.id}",
            ],
        )

