"""The train slot in Khaos's words. An engine's train module offers training_api(repo_dir), adapter_kwargs(cfg),
preprocess_pairs(...), open_module(...) and adapter_file_in(dir); this is the refusal they share."""

from __future__ import annotations


class TrainingUnavailable(RuntimeError):
    """The engine's training side cannot be reached here, with the reason."""


__all__ = ["TrainingUnavailable"]
