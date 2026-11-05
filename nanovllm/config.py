import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str # 模型本地目录
    max_num_batched_tokens: int = 16384 # [Prefill token 预算][调度器、预热] 一轮 Prefill 最多算多少**新** token，防止一轮太久、OOM
    max_num_seqs: int = 512 # [每轮序列数限制][调度器、图捕获] 一轮最多多少条请求
    max_model_len: int = 4096 # [预热、图捕获] 单条序列最大长度
    gpu_memory_utilization: float = 0.9 # [自动估算缓存容量][分配 KV 池] 显存预算比例
    tensor_parallel_size: int = 1 # [单卡推理][并行] 用几张 GPU
    enforce_eager: bool = False # [执行器] True 时禁用 CUDA Graph，方便调试
    hf_config: AutoConfig | None = None
    eos: int = -1
    """
    [Paged Attention] “逻辑位置 → 物理位置”的映射，这是后面所有 Paged Attention 的代码的基础

    这里的 block，就是”一块分页“的意思，这里定义一块可以容纳 256 个 token 对应的 KV
    """
    kvcache_block_size: int = 256 # [缓存粒度控制] 块大小配置，即每个 KV 块多少 token
    num_kvcache_blocks: int = -1 # KV 块数，-1 表示启动时**自动算**

    def __post_init__(self):
        assert os.path.isdir(self.model)
        # IMPORTANT: FlashAttention 分页模式的硬性要求：页大小必须是 256 的倍数
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        # 配置的长度不能超过模型位置编码支持的长度
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
