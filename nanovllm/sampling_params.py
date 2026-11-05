from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False # True 时即使生成了结束符也继续，压测用

    """
    __post_init__ 是 dataclass 在自动生成的 __init__ 最后调用的一次性“收尾钩子”，
    这段代码用它来做参数校验
    """
    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
