import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

"""
要把 Q、K、V 三个线性层合并成**一次**矩阵乘法。
三个独立的 `nn.Linear` 要做 3 次矩阵乘法。合并后只做 1 次，更快，再 `split` 成三份。MLP 的 gate/up 同理。

q_proj.weight  [2048, 1024]   ─┐
k_proj.weight  [1024, 1024]    ├─→ qkv_proj.weight [4096, 1024]
v_proj.weight  [1024, 1024]   ─┘   第 0 维：Q 占 0~2047，K 占 2048~3071，V 占 3072~4095
"""

def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module): # [张量并行] 的核心逻辑，模型层明确使用这些并行层

    def __init__(
        self,
        input_size: int,            # 每个输入向量有多少个特征
        output_size: int,           # 每个输出向量有多少个特征
        bias: bool = False,         # 是否创建并使用偏置向量
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        """
        这份参数以后需要加载数值时，使用保存在自己身上的这套加载规则
        weight_loader 是作者定义的属性协议，PyTorch 不会因为这个属性存在，就自动加载权重。 外部加载代码仍要显式调用它。
        """
        self.weight.weight_loader = self.weight_loader
        # 如果调用方要求使用偏置，就进入这个分支
        if bias:
            # 创建长度为 output_size 的偏置
            self.bias = nn.Parameter(torch.empty(output_size))
            """给偏置参数挂上同一套加载规则"""
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


"""普通复制线性层"""
class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


"""列并行线性层"""
class ColumnParallelLinear(LinearBase): # [张量并行] 权重如何切分：输出维度除以 TP 数；加载时沿权重第 0 维取当前 rank 的分片；前向如何执行：只计算当前分片的输出，保持输出分片状态

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


"""
专用合并并行层

Gate/Up 合并由这里实现
"""
class MergedColumnParallelLinear(ColumnParallelLinear): # [合并投影] gate/up 合并

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int], # 各个子投影的输出宽度，比如 [gate, up]
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias) # 用 sum(output_sizes) 创建一个合并投影，加载时按照 loaded_shard_id，把 gate 和 up 的本地分片分别放到正确区间

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int            # 表示当前加载的是第几个子投影：0 → gate，1 → up。这里的 shard 指合并参数中的一段，不是 GPU 编号
    ):
        param_data = param.data
        # [单卡 TODO:] 计算当前段前面，一共已经占用了多少输出行。
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        # 取当前段的行数
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


"""Q/K/V 合并在这里实现"""
class QKVParallelLinear(ColumnParallelLinear): # [合并投影] Q/K/V 合并

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str): # [合并投影] 根据 "q"、"k"、"v" 计算目标区间，再把对应原始权重的当前 rank 分片写进去。
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


"""行并行线性层"""
class RowParallelLinear(LinearBase): # [张量并行] 权重如何切分：输入维度除以 TP 数；加载时沿权重第 1 维切分；前向如何执行：计算局部结果，再执行 dist.all_reduce(y) 求和

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    # 直接接收已经切分的输入，输入分片由前面的并行层产生
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            # 执行通信《all_reduce》 # [NCCL 通信] 行并行结果求和
            dist.all_reduce(y)
        return y
