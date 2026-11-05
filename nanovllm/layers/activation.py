import torch
from torch import nn
import torch.nn.functional as F


"""
MLP 的激活函数: 实现 Qwen3 MLP 的核心运算 `SiLU(gate) × up`

HF 实现中，MLP 有 `gate_proj` 和 `up_proj` 两个投影。为了省一次矩阵乘法，我们把它们合并成一个投影，输出在最后一维拼在一起，所以这里先切开。
"""
class SiluAndMul(nn.Module):

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #
        """
        实际上是 gate, up = gate_up.chunk(2, dim=-1)，外面传进来的就是 gate_up，所以这里要拆开

        （语法上：这段代码的意思是：沿着最后一维（dim=-1），尽量把 x 切成 2 块，返回一个 tuple）
        """
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
