from functools import lru_cache
import torch
from torch import nn

"""本方法负责真正旋转一个 Q 或 K 张量"""
def apply_rotary_emb(
    x: torch.Tensor,    # [N, 头数, head_dim]
    cos: torch.Tensor,  # [N, 1, head_dim/2]，中间的 1 用来广播到所有头
    sin: torch.Tensor,
) -> torch.Tensor:
    # 向量切成前后两半：x1[i] 和 x2[i] 组成一个旋转平面。三角函数精度敏感，转 fp32 算
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    # 将旋转后的两半重新拼回原来的特征排列
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


"""
这个是 RoPE：把 token 的绝对位置信息融入 Q 和 K

原理：把向量每两维看成一个二维平面，按“位置 × 频率”的角度旋转：
"""
class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,                 # 一个 head 的完整特征宽度
        rotary_dim: int,                # 其中有多少维参与旋转
        max_position_embeddings: int,   # 建立位置表的长度，覆盖位置 0 到 M-1
        base: float,                    # 控制不同特征对的旋转频率，在模型配置里叫 rope_theta。它不是直接使用的旋转角度，实际角度还要经过下面的频率计算。
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size # 限制：一个 head 的所有维度都参与旋转
        """
        每个旋转平面一个频率，公式：1 / base^(2i/d)。
        （共 d/2 个）

        相当于：
        indices = torch.arange(0, rotary_dim, 2, dtype=torch.float)
        exponents = indices / rotary_dim
        inv_freq = 1.0 / (base ** exponents)

        dtype=torch.float 在这里表示 FP32。即使模型主干采用 BF16，这个频率向量仍明确使用 FP32 来创建。
        """
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        """
        建立所有位置的编号
        (这个 t 是建立整张位置表时使用的全部位置；前向传进来的 positions，则是本轮实际需要哪些位置。)
        """
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        """
        外积：freqs[位置, 频率] = 位置 × 频率，形状 [P, d/2]

        (freqs 其实是角度表)
        """
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        """对角度表中的每个元素，分别计算余弦和正弦"""
        cos = freqs.cos()
        sin = freqs.sin()
        """
        拼成 [P, 1, d]。中间的 1 让查表结果能广播到所有头
        ————————————

        cos：[M, d/2]
        sin：[M, d/2]

        拼接后：[M, d]

        每一行的排列是：[全部 cos 值 | 全部 sin 值]

        ————————————
        .unsqueeze_(1) 在第 1 维增加一个长度为 1 的维度：
        [M, d] → [M, 1, d]
        (这么做是为了让查表结果能广播到所有头)
        """
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        """
        buffer：随模型移动设备，但不是可训练参数。

        persistent=False 表示它不进入这个模块的 state_dict，因为可以根据配置重新生成。它在模块运行期间仍然保存在内存中
        """
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor, # [N]，本轮 token 的逻辑位置，positions 是逻辑位置（在序列里第几个 token），与 KV 存在哪个物理块无关。
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """根据本轮的位置取出对应行"""
        cos_sin = self.cos_sin_cache[positions]     # 用位置查表 → [N, 1, d]
        cos, sin = cos_sin.chunk(2, dim=-1)         # 各 [N, 1, d/2]
        """对 Q 旋转"""
        query = apply_rotary_emb(query, cos, sin)
        """对 K 旋转"""
        key = apply_rotary_emb(key, cos, sin)
        return query, key                           # V 不旋转！只有 Q·K 决定相似度


# 相同参数返回同一个对象。所有层共享一张表。
@lru_cache(1) # 缓存最近一组参数对应的返回对象
def get_rope(
    head_size: int,     # 指一个 head 有多少个特征，不是有多少个 head
    rotary_dim: int,    # 其中有多少维参与旋转
    max_position: int,  # 建立位置表的长度，覆盖位置 0 到 M-1
    base: float,        # 控制不同特征对的旋转频率
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
