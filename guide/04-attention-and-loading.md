# 第 4 章：补上注意力，把真模型接回去

**上一章留下的问题**：`Qwen3DecoderLayer` 里还缺 `Qwen3Attention`。

**本章要回答**：

```text
Qwen3Attention 要做什么？
├─ 把隐藏向量变成 Q、K、V                → 需要 QKVParallelLinear
├─ 告诉模型每个 token 的位置             → 需要 RoPE
├─ 计算注意力                           → 需要 Attention
└─ 接回 ModelRunner                     → 需要 load_model
```

## 本章任务卡

|                        |                                                                                                                                                                                                                                                        |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **要解决什么**   | 模型还缺注意力；权重还没放进去；`ModelRunner` 还是假的                                                                                                                                                                                               |
| **要实现什么**   | `Qwen3Attention`、`QKVParallelLinear`、`RotaryEmbedding`、最简版 `Attention`、`load_model`、真正的 `ModelRunner`                                                                                                                           |
| **要调用什么**   | `split`、`view`、`flatten`、`einsum`、`register_buffer`、`lru_cache`、`F.scaled_dot_product_attention`、`safe_open`、`get_parameter`、`dist.init_process_group`、`set_default_device`、`set_default_dtype`、`inference_mode` |
| **为什么这样做** | 先用最简单的注意力把模型跑对，并和 HF 对比。确认模型没问题之后，后面出错就只可能是引擎的问题                                                                                                                                                           |

## 4.1 注意力要解决什么问题

一个词的含义依赖上下文。“苹果很好吃”和“苹果发布了新手机”里的“苹果”不一样。注意力让每个 token 去看前面的 token，并按相关性吸收它们的信息。

每个 token 用线性投影得到三个向量：

- **Q（Query）**：我想找什么；
- **K（Key）**：我能提供什么线索；
- **V（Value）**：我实际的内容。

计算第 `i` 个 token 的输出时：

```text
output[i] = softmax(Q[i] · K[0..i] / sqrt(d)) · V[0..i]
```

“只看 `0..i`”叫**因果（causal）**约束：不能看未来。

## 4.2 Qwen3Attention：把流程写下来

对照权重树，这一层有 `q_proj`、`k_proj`、`v_proj`、`o_proj`、`q_norm`、`k_norm`。流程是：

```text
hidden_states
  → 投影，切出 Q、K、V
  → reshape 成 [N, 头数, head_dim]
  → Q、K 过 RMSNorm
  → Q、K 加上位置信息（RoPE）
  → Attention
  → 展平 + o_proj 投影回 hidden_size
```

**多头**：把向量分成多组，每组独立做注意力，可以关注不同方面。

配置里的三个数决定了形状：`num_attention_heads=16`、`num_key_value_heads=8`、`head_dim=128`。

```text
hidden_states          [N, 1024]
qkv = qkv_proj(...)    [N, 4096]               (16 + 8 + 8) × 128
q                      [N, 2048] → view → [N, 16, 128]
k, v                   [N, 1024] → view → [N,  8, 128]
o = attn(q, k, v)      [N, 16, 128]
o.flatten(1, -1)       [N, 2048]
o_proj                 [N, 1024]
```

两个疑问：

- **K/V 头数为什么只有 Q 的一半？** 这叫 GQA（Grouped-Query Attention）：多个 Q 头共用一个 K/V 头。第 6 章你会看到 K/V 要缓存在显存里，头数减半，缓存就省一半。
- **为什么 `16 × 128 = 2048 ≠ 1024`？** Qwen3 让注意力工作在更大的维度，再用 `o_proj` 投影回 1024。

追加到 `qwen3.py`：

<!-- append: nanovllm/models/qwen3.py -->

```python
class Qwen3Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        # 单卡时 tp_size=1，这几行没有效果。多卡时每张卡只负责一部分头（第 11 章）
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        # head_dim 配置里可能没写，这时才由 hidden_size 推算
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5     # 注意力公式里的 1/sqrt(d)
        self.qkv_bias = qkv_bias

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
        )
        # transformers 5 把 rope_theta 收进了 rope_scaling 字典，两种配置都要兼容
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )
        # Qwen3 的设计：没有偏置时，Q、K 各加一个归一化。名字要和权重文件一致
        if not self.qkv_bias:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        # split 不复制内存：q、k、v 是同一块大张量的不同切片
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # view 用 -1：N 是多少都可以，不用单独传
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        if not self.qkv_bias:
            q = self.q_norm(q)
            k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output
```

### 逐行看 Qwen3Attention

**`__init__`：**

| 代码                                                           | 值（0.6B） | 为什么                                                                 |
| -------------------------------------------------------------- | ---------- | ---------------------------------------------------------------------- |
| `tp_size = dist.get_world_size()`                            | 1          | 多卡时每张卡只负责一部分头。这也是必须先初始化进程组的原因             |
| `self.num_heads = total // tp_size`                          | 16         | 本卡负责的 Q 头数                                                      |
| `self.head_dim = head_dim or hidden_size // total_num_heads` | 128        | 配置写了就用配置的。`1024 // 16 = 64` 并不是 0.6B 的真实值           |
| `self.q_size`、`self.kv_size`                              | 2048、1024 | `forward` 里 `split` 要用                                          |
| `self.scaling = head_dim ** -0.5`                            | ≈0.088    | 公式里的`1/sqrt(d)`。维度越高点积越大，不缩放的话 softmax 会过于尖锐 |
| `if not self.qkv_bias`                                       | 真         | Qwen3 没有偏置时才有`q_norm`、`k_norm`，这个写法和 HF 的判断一致   |

**`forward`，以 N=5 为例：**

```text
hidden_states          [5, 1024]
qkv                    [5, 4096]
split                  q [5, 2048]   k [5, 1024]   v [5, 1024]
view                   q [5, 16, 128]  k [5, 8, 128]  v [5, 8, 128]
q_norm / k_norm        形状不变，每个头向量归一化
rotary_emb             形状不变，Q、K 被旋转
attn                   [5, 16, 128]     每个 Q 头一份结果
flatten(1, -1)         [5, 2048]        把第 1 维到最后一维压平
o_proj                 [5, 1024]        回到隐藏维度，下一层才能接上
```

**顺序不能换**：先 Norm 后 RoPE，这是模型训练时的顺序。权重是按这个顺序训练出来的，换了就算错。

用到的新东西：`QKVParallelLinear`、`get_rope`、`Attention`。

## 4.3 合并的 QKV 投影

道理和上一章的 `gate_up_proj` 一样：三次矩阵乘法合并成一次。

```text
q_proj.weight  [2048, 1024]   ─┐
k_proj.weight  [1024, 1024]    ├─→ qkv_proj.weight [4096, 1024]
v_proj.weight  [1024, 1024]   ─┘   第 0 维：Q 占 0~2047，K 占 2048~3071，V 占 3072~4095
```

追加到 `nanovllm/layers/linear.py`：

<!-- append: nanovllm/layers/linear.py -->

```python
class QKVParallelLinear(LinearBase):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = total_num_heads
        self.num_kv_heads = total_num_kv_heads
        # Qwen3-0.6B：(16 + 2×8) × 128 = 4096
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
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
        param.data.narrow(0, shard_offset, shard_size).copy_(loaded_weight)
```

**用 Qwen3-0.6B 核对一遍参数：**

| 变量                                        | 计算               | 结果                           |
| ------------------------------------------- | ------------------ | ------------------------------ |
| Q 的输出宽度                                | 16 × 128          | 2048，对应`q_proj` 的第 0 维 |
| K、V 的输出宽度                             | 8 × 128           | 1024                           |
| `output_size`                             | (16 + 2×8) × 128 | 4096                           |
| `loaded_shard_id="k"` 的 `shard_offset` | 16 × 128          | 2048，K 从这里开始             |
| `loaded_shard_id="v"` 的 `shard_offset` | 2048 + 1024        | 3072，V 从这里开始             |

## 4.4 位置信息：RoPE

**问题**：注意力公式只是 Q·K，和 token 的先后顺序无关。那“狗咬人”和“人咬狗”岂不是一样？

**解决办法**：把位置编码进 Q 和 K。把向量每两维看成一个二维平面，按“位置 × 频率”的角度旋转：

```text
[x1']   [cos θ  -sin θ] [x1]
[x2'] = [sin θ   cos θ] [x2]        θ = position × inv_freq
```

这样两个 token 的 Q·K 点积，只和它们的**位置差**有关。

**为什么预计算**：每层、每轮都要用。提前算好一张 `cos/sin` 表，运行时只需查表。

**为什么 `get_rope` 带 `lru_cache`**：每一层的参数完全相同，28 层共享一张表就够了，否则这张表要存 28 份。

文件：`nanovllm/layers/rotary_embedding.py`

<!-- file: nanovllm/layers/rotary_embedding.py -->

```python
from functools import lru_cache
import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,    # [N, 头数, head_dim]
    cos: torch.Tensor,  # [N, 1, head_dim/2]，中间的 1 用来广播到所有头
    sin: torch.Tensor,
) -> torch.Tensor:
    # 向量切成前后两半：x1[i] 和 x2[i] 组成一个旋转平面。三角函数精度敏感，转 fp32 算
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        # 每个旋转平面一个频率：1 / base^(2i/d)。共 d/2 个
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)  # 所有可能的位置
        # 外积：freqs[位置, 频率] = 位置 × 频率，形状 [P, d/2]
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        # 拼成 [P, 1, d]。中间的 1 让查表结果能广播到所有头
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        # buffer：随模型移动设备，但不是可训练参数。
        # persistent=False：不写入 state_dict。它能算出来，权重文件里也没有
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,  # [N]，每个 token 的绝对位置
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]     # 用位置查表 → [N, 1, d]
        cos, sin = cos_sin.chunk(2, dim=-1)         # 各 [N, 1, d/2]
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key                           # V 不旋转！只有 Q·K 决定相似度


# 相同参数返回同一个对象。所有层共享一张表。
@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
):
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
```

### 逐行看 RoPE

**建表（`__init__`）：**

| 代码                                                         | 形状（0.6B）        | 为什么                                                             |
| ------------------------------------------------------------ | ------------------- | ------------------------------------------------------------------ |
| `torch.arange(0, rotary_dim, 2, dtype=torch.float)`        | `[64]`            | 每两维一个旋转平面，共 64 个                                       |
| `base ** (... / rotary_dim)` 再取倒数                      | `[64]`            | 不同平面转速不同：有的快，能区分相邻位置；有的慢，能区分很远的位置 |
| `torch.arange(max_position_embeddings, dtype=torch.float)` | `[32768]`         | 所有可能的位置                                                     |
| `einsum("i,j -> ij", t, inv_freq)`                         | `[32768, 64]`     | 外积，每个元素是“位置 × 频率”，也就是旋转角度                   |
| `torch.cat((cos, sin), dim=-1)`                            | `[32768, 128]`    | 前一半 cos，后一半 sin                                             |
| `.unsqueeze_(1)`                                           | `[32768, 1, 128]` | 留一个头维度，查表后能广播                                         |

**为什么这里特意写 `dtype=torch.float`？** 建模型时默认类型被设成了 bf16，而 bf16 的有效位数很少，连 257 这样的整数都表示不了。位置和角度必须用 fp32，否则位置信息直接就错了。

**运行（`forward`）：** `positions` 有 N 个元素，`cache[positions]` 得到 `[N, 1, 128]`，切开后是 `cos`、`sin` 各 `[N, 1, 64]`。`Q` 是 `[N, 16, 128]`，切成两半后是 `[N, 16, 64]`，中间的 1 广播到 16。K 的 8 个头也一样，所以 Q、K 能共用一张表。

**`lru_cache(1)`**：函数的参数相同时直接返回上次的对象。28 层都调用 `get_rope`，只建一次表。

## 4.5 Attention：先用最简单的

**现在先不追求性能。** 用 PyTorch 自带的 `scaled_dot_product_attention`，只支持一条序列。下一章遇到问题时再换。

GQA 的实现：Q 头 0、1 对应 KV 头 0，Q 头 2、3 对应 KV 头 1……用 `repeat_interleave` 把 K/V 复制对齐即可。

文件：`nanovllm/layers/attention.py`（下一章会替换）

<!-- file: nanovllm/layers/attention.py -->

```python
import torch
from torch import nn
import torch.nn.functional as F


class Attention(nn.Module):

    def __init__(self, num_heads, head_dim, scale, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        repeat = self.num_heads // self.num_kv_heads
        k = k.repeat_interleave(repeat, dim=1)   # [N, 8, d] → [N, 16, d]
        v = v.repeat_interleave(repeat, dim=1)
        # SDPA 约定序列维在倒数第二维：[..., 序列, 维度]。把“头”换到前面当 batch
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        # is_causal=True：下三角掩码，不看未来。缩放默认就是 1/sqrt(d)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return o.transpose(0, 1)                 # [N, 16, d]
```

**逐行看这版 Attention：**

- `repeat = num_heads // num_kv_heads`：16 ÷ 8 = 2，每个 K/V 头给 2 个 Q 头用。
- `repeat_interleave(repeat, dim=1)`：`[K0, K1, ...]` 变成 `[K0, K0, K1, K1, ...]`，头数对齐到 16。
- `transpose(0, 1)`：SDPA 把 `[头, N, d]` 的最后两维看作“序列、维度”，第 0 维看作批。这样每个头就是独立的一批。
- `is_causal=True`：内部自动生成下三角掩码，每个 token 只能看自己和之前的 token。
- 最后一个 `transpose` 换回 `[N, 头, d]`，上层不用知道这个细节。

这一版只能处理**一条**序列：所有 token 都被当作同一序列的连续位置。下一章解决这个问题。

**模型结构到这里就完整了。** 还差把训练好的权重放进去。

## 4.6 把权重放进模型

遍历所有 `.safetensors`，把每个权重放到模型的正确位置。分两种情况：

- 名字里含 `q_proj` 这类片段：这是**被我们合并的权重**。替换名字，并告诉 `weight_loader` 这是第几片。这就是上一章 `packed_modules_mapping` 的作用。
- 其他：直接按名字找到参数。

```text
model.layers.0.self_attn.q_proj.weight
                    ↓ 把 "q_proj" 替换成 "qkv_proj"
model.layers.0.self_attn.qkv_proj.weight  →  weight_loader(param, tensor, "q")
```

文件：`nanovllm/utils/loader.py`

<!-- file: nanovllm/utils/loader.py -->

```python
import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    # copy_ 会自动跨设备复制，loaded_weight 在 CPU 也没问题
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        # safe_open 按需读取，不会一次把整个文件读进内存
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        # 合并层一定有 weight_loader。没有就让它报错，不能静默跳过
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                # for-else：循环没有 break，才走这里，也就是“不是合并权重”
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
```

### 逐行看 load_model

以两个权重名为例：

```text
model.layers.0.mlp.down_proj.weight
  不含任何 packed 片段 → 走 else → get_parameter 按原名找到
  RowParallelLinear 有 weight_loader → weight_loader(param, tensor)

model.layers.0.mlp.gate_proj.weight
  含 "gate_proj" → v="gate_up_proj", shard_id=0
  param_name = model.layers.0.mlp.gate_up_proj.weight
  weight_loader(param, tensor, 0) → 写入第 0～3071 行
```

| 代码                                                       | 为什么                                                                               |
| ---------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| `glob("*.safetensors")`                                  | 大模型的权重会分成多个文件，全部遍历一遍。放置位置由名字决定，所以遍历顺序不影响结果 |
| `safe_open(file, "pt", "cpu")`                           | 先读到 CPU。按需读取，内存不会一下子涨满                                             |
| `f.get_tensor(weight_name)`                              | 这时才真正读取这一个权重                                                             |
| `for ... else`                                           | `else` 只在循环没有 `break` 时执行，正好表示“不是合并权重”                     |
| `getattr(param, "weight_loader")`，没有默认值            | 合并层必须自己知道怎么放。没有就报错，不能静默跳过                                   |
| `getattr(param, "weight_loader", default_weight_loader)` | 普通参数（比如 RMSNorm 的`weight`）直接整体复制                                    |

**这里用的是子串匹配，会不会误伤？** `k in weight_name` 确实是子串判断。但 Qwen3 的权重名里，只有目标权重才含这些片段（`down_proj`、`o_proj` 都不含）。换模型时要重新确认这一点。

`model.get_parameter` 在找不到名字时会抛异常。这正是我们想要的：**模型结构写错，加载时立刻暴露**。

### 检查点 4-A：模型和 HF 对齐

接回引擎之前，先单独确认模型是对的。否则后面出问题，你分不清是模型错了还是引擎错了。

`torch.set_default_device("cuda")` 让之后创建的张量默认都在 GPU 上；`set_default_dtype` 让默认类型变成 bf16。这样模型直接以正确的类型创建在 GPU 上，不用先在 CPU 上构建再 `.to()`。

文件：`tests/test_model_vs_hf.py`

<!-- file: tests/test_model_vs_hf.py -->

```python
import os
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM

from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.loader import load_model

path = os.path.expanduser(os.environ.get("MODEL_PATH", "~/huggingface/Qwen3-0.6B/"))
cfg = AutoConfig.from_pretrained(path)

# qwen3.py 用 dist.get_world_size() 判断并行度，所以单卡也要初始化
dist.init_process_group("nccl", "tcp://localhost:2333", world_size=1, rank=0)
torch.set_default_dtype(cfg.dtype)
torch.set_default_device("cuda")
model = Qwen3ForCausalLM(cfg)
load_model(model, path)
torch.set_default_device("cpu")
torch.set_default_dtype(torch.float32)

ids = torch.randint(0, cfg.vocab_size, (32,), device="cuda")
positions = torch.arange(32, device="cuda")

with torch.inference_mode():
    mine = model.compute_logits(model(ids, positions)).float()
    hf = AutoModelForCausalLM.from_pretrained(path, dtype=cfg.dtype).to("cuda").eval()
    theirs = hf(ids[None]).logits[0].float()

diff = (mine - theirs).abs().max().item()
agree = (mine.argmax(-1) == theirs.argmax(-1)).float().mean().item()
print(f"logits 形状 {tuple(mine.shape)}，最大差异 {diff:.4f}，argmax 一致率 {agree:.1%}")
assert diff < 1.0 and agree > 0.9, "和 HF 差异过大，模型有问题"
dist.destroy_process_group()
print("OK")
```

```bash
python tests/test_model_vs_hf.py
```

期望差异很小（bf16 下的舍入误差），argmax 几乎全部相同。如果不一致，按以下顺序排查：

1. `get_parameter` 报错？模型里的属性名和权重文件名不一致。
2. 差异巨大？检查 `weight_loader` 的 `shard_offset`。
3. 差异中等？检查 RoPE 的基数（0.6B 是 1000000），以及 Q/K Norm 和 RoPE 的先后顺序。
4. 只有第一层附近对？检查 `residual` 的传递。

顺便扩充 `tests/test_parts.py`，在 `print("OK")` 之前加入 RoPE 和 QKV 的测试：

<!-- replace-line: tests/test_parts.py :: print("OK") -->

```python
# 5. RoPE：位置差相同，Q·K 就相同
from nanovllm.layers.linear import QKVParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
rope = get_rope(8, 8, 100, 10000.0)
assert rope is get_rope(8, 8, 100, 10000.0), "应该是同一个对象"
q, k = torch.randn(1, 1, 8), torch.randn(1, 1, 8)
def dot(pq, pk):
    a, _ = rope(torch.tensor([pq]), q, k)
    _, b = rope(torch.tensor([pk]), q, k)
    return (a * b).sum().item()
assert abs(dot(10, 3) - dot(50, 43)) < 1e-3   # 位置差都是 7
assert abs(dot(10, 3) - dot(10, 5)) > 1e-5    # 位置差不同

# 6. QKV 投影：权重必须落到正确的区间
proj = QKVParallelLinear(4, 2, total_num_heads=2, total_num_kv_heads=1)
assert proj.weight.shape == (8, 4)   # (2 + 2×1) × 2 = 8
with torch.no_grad():
    proj.weight.weight_loader(proj.weight, torch.full((4, 4), 1.0), "q")
    proj.weight.weight_loader(proj.weight, torch.full((2, 4), 2.0), "k")
    proj.weight.weight_loader(proj.weight, torch.full((2, 4), 3.0), "v")
assert proj.weight[:, 0].tolist() == [1, 1, 1, 1, 2, 2, 3, 3]
print("OK")
```

## 4.7 接回 ModelRunner：去掉假模型

现在把第 2 章假的 `ModelRunner` 整个替换掉。外层完全不用动，因为接口没变。

启动顺序有讲究：

```text
① init_process_group + set_device     层的构造函数要用 dist，必须最先
② set_default_dtype / default_device  让模型直接以 bf16 建在 GPU 上
③ 创建模型 + load_model
④ 恢复默认 dtype / device             不影响后面的 CPU 代码，比如用户随手创建的张量
```

`run` 是最直接的写法：**逐条序列，每次把整条序列喂进模型。**

文件：`nanovllm/engine/model_runner.py`（替换第 2 章的假版本）

<!-- file: nanovllm/engine/model_runner.py -->

```python
import torch
import torch.distributed as dist

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int = 0):
        self.config = config
        hf_config = config.hf_config
        self.world_size = config.tensor_parallel_size
        self.rank = rank

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

    def exit(self):
        torch.cuda.synchronize()    # 等 GPU 上未完成的工作结束
        dist.destroy_process_group()

    # inference_mode：不记录反向传播信息，省时省显存。推理不需要梯度
    @torch.inference_mode()
    def run(self, seqs: list[Sequence]) -> list[int]:
        token_ids = []
        for seq in seqs:
            input_ids = torch.tensor(seq.token_ids, dtype=torch.int64, device="cuda")
            positions = torch.arange(len(seq), device="cuda")
            logits = self.model.compute_logits(self.model(input_ids, positions))
            temperature = torch.tensor([seq.temperature], dtype=torch.float32, device="cuda")
            # 只有最后一个位置用于预测下一个 token
            token_ids.append(self.sampler(logits[-1:], temperature).item())
        return token_ids
```

### 逐行看 ModelRunner

| 代码                                                             | 为什么                                                                                     |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `dist.init_process_group("nccl", "tcp://localhost:2333", ...)` | 单卡也要调用：层的构造函数要用`dist.get_world_size()`。2333 只是用来交换初始化信息的端口 |
| `torch.cuda.set_device(rank)`                                  | 进程使用第几张卡。多卡时每个进程用不同的卡                                                 |
| `default_dtype = torch.get_default_dtype()`                    | 先记下来，建完模型要恢复                                                                   |
| `set_default_dtype(hf_config.dtype)`                           | 之后`torch.empty(...)` 直接建成 bf16                                                     |
| `set_default_device("cuda")`                                   | 之后创建的张量直接在 GPU 上，不用在 CPU 上建完再搬过去                                     |
| `load_model(self.model, config.model)`                         | 这时才有真实的权重                                                                         |
| 恢复默认值                                                       | 这是进程级的全局设置。不恢复的话，用户之后的`torch.tensor` 都会跑到 GPU 上，还变成 bf16  |
| `@torch.inference_mode()`                                      | 不记录反向传播信息，省时省显存                                                             |
| `logits[-1:]`                                                  | 只取最后一个位置。写`[-1:]` 而不是 `[-1]`，是为了保留批维度：`[1, V]`                |
| `.item()`                                                      | 把只有一个元素的张量变成 Python 整数。这一步要等 GPU 算完                                  |

**`run` 为什么不修改 `seq`？** 只读输入，返回结果。更新序列是 `Scheduler.postprocess` 的职责。这个分工第 11 章多进程时会变得很重要：子进程只负责计算。

## 检查点 4-B：第一次说人话

```bash
python example.py
```

**这是个重要的时刻。** 输入同样是第 2 章的 `example.py`，一行没改，现在模型真的在回答问题了。

### 接下来看症状

写一个脚本来测量每一步的耗时。**之后几章都用它来衡量改进效果。**

文件：`tests/time_steps.py`

<!-- file: tests/time_steps.py -->

```python
import os
from time import perf_counter
from nanovllm import LLM, SamplingParams


def main():
    path = os.path.expanduser(os.environ.get("MODEL_PATH", "~/huggingface/Qwen3-0.6B/"))
    max_tokens = int(os.environ.get("MAX_NEW_TOKENS", 200))
    llm = LLM(path, enforce_eager=True)
    sp = SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=max_tokens)
    llm.add_request("list all prime numbers within 100", sp)
    llm.add_request("introduce yourself", sp)

    begin = perf_counter()
    step = 0
    while not llm.is_finished():
        t = perf_counter()
        llm.step()
        if step % 25 == 0:
            print(f"第 {step:4d} 轮：{(perf_counter() - t) * 1000:8.1f} ms")
        step += 1
    total = perf_counter() - begin
    print(f"共 {step} 轮，{total:.2f} 秒，吞吐 {2 * max_tokens / total:.1f} tok/s")


if __name__ == "__main__":
    main()
```

```bash
python tests/time_steps.py
```

你会看到两个明显的症状：

1. **两条请求是一条接一条算的。** GPU 擅长并行，这样太浪费了。
2. **越到后面，每一轮越慢。**

## 本章小结

下面两章分别解决这两个症状：

```text
症状 1：请求排队算      → 第 5 章 批处理
症状 2：越生成越慢      → 第 6 章 KV Cache
```

先解决第一个。
