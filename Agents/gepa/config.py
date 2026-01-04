from dataclasses import dataclass


@dataclass
class GepaConfig:
    """GEPA 运行超参配置。"""

    budget: int = 50
    window_budget: int | None = None
    mini_batch_size: int = 2
    num_initial: int = 6
    exploit_prob: float = 0.95
    merge_prob: float = 0.9
    seed_prompt: str | None = None
    max_workers: int = 8
    use_json_mode: bool = True
    target_temperature: float = 0.0
    reflection_temperature: float = 0.2
    max_tokens: int = 4096
    feedback_ratio: float = 0.5



