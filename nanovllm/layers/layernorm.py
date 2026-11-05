import torch
from torch import nn

"""
这段代码只在模型不是 float32 时安全。
张量已经是 float32 时，`x.float()` 和 `x.to(float32)` 都直接返回原张量，不做拷贝。
于是 `residual` 和 `x` 是同一块内存，后面的 `x.mul_` 会把残差也改掉，结果悄悄出错。
Qwen3-0.6B 是 bf16，所以不受影响。如果你以后要跑 float32 权重，请在第 12 章看修复方法。
"""

"""
这个的作用是实现归一化：
    output = x / sqrt(mean(x²) + eps) × weight

它只看平方均值，不会先减去均值，所以叫 RMSNorm，而不是普通 LayerNorm
"""
class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int, # 表示要归一化的最后一维有多少个特征
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps # 一个很小的正数，用来保证公式的分母也不会为 0，用来保证数值稳定
        self.weight = nn.Parameter(torch.ones(hidden_size)) # 把传入的参数用来定义这个 Module 的权重（这里变量名正好是权重的意思，不过任何名字都是在定义这个 Module 的权重）

    """
    输入：x
    输出：RMSNorm(x)
    """
    @torch.compile # 这会让 PyTorch 在调用时捕获和编译这段计算，尝试生成更高效的执行方式。默认编译后端是 TorchInductor，GPU 代码生成中会使用 Triton 等技术
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # 记住输入的数据类型
        orig_dtype = x.dtype
        # IMPORTANT: 临时使用 FP32 计算，最后还要转回原来的类型。使用 FP32，是因为后面包含平方和求均值。FP16 的数值范围较小，平方容易溢出。例如 300²=90000，超过 FP16 能表示的最大有限值。
        x = x.float()
        """输出 = 输入 ÷ sqrt(输入平方的平均值 + eps) × 权重"""
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        # 先将已经归一化的 FP32 结果转回输入类型，再乘上逐维权重
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    """
    带残差的版本：目标是同时产生两份结果 x, residual

    先算：
    summed = x + residual

    再返回：
    RMSNorm(summed) → 送入接下来的 Attention 或 MLP
    summed          → 保留为后续残差通路
    """
    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        # 先将两边转换成 FP32，再相加，让加法及后续归一化使用 FP32 的结果
        x = x.float().add_(residual.float())
        # 这一步保存尚未归一化的相加结果，并恢复成模型使用的类型。
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        为什么要同时有 rms_forward 和 add_rms_forward ？
            - 因为 Transformer 需要“归一化后送去计算”，同时需要“通过残差通路保留原状态”。本项目把残差加法与归一化合并执行，所以提供了两个入口。
        """
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
