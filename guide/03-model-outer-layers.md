# 第 3 章：下一个 token 怎么来？先拆模型外层

**上一章留下的问题**：`ModelRunner.run` 返回的是随机数。

**本章要回答**：真正的 `run` 里要做什么？一路追问下去：

```text
run：下一个 token 怎么来？
├─ token ← logits（每个词一个分数）          → 需要 Sampler
└─ logits ← 模型
   ├─ 模型有哪些部分？看权重文件
   ├─ Qwen3ForCausalLM = 主体 + 输出头        → 需要 Qwen3Model、ParallelLMHead
   ├─ Qwen3Model = 词嵌入 + N 层 + 归一化     → 需要 Embedding、DecoderLayer、RMSNorm
   └─ DecoderLayer = 注意力 + MLP + 残差      → 需要 Qwen3MLP（本章）、Qwen3Attention（下一章）
```

## 本章任务卡

| | |
|---|---|
| **要解决什么** | `run` 只会返回随机数。真实的 token 从哪来？ |
| **要实现什么** | `Sampler`、`Qwen3ForCausalLM`、`Qwen3Model`、`Qwen3DecoderLayer`、`Qwen3MLP`、`RMSNorm`、`SiluAndMul`、嵌入和输出头、合并投影 |
| **要调用什么** | `torch.softmax`、`exponential_`、`F.embedding`、`F.linear`、`F.silu`、`nn.Parameter`、`nn.ModuleList`、`narrow`、`copy_`、`safe_open` |
| **为什么这样做** | 模型结构不是我们设计的，必须和权重文件一一对应。所以先看权重文件，再照着它从外往里写 |

## 3.1 token 从 logits 来：Sampler

模型输出的 logits 是一个长度等于词表大小的向量，每个词一个分数。从它选出一个词，分三步：

1. `logits / temperature`，温度越小分布越尖锐；
2. `softmax` 变成概率；
3. 按概率抽一个。

第 3 步最直观的写法是 `torch.multinomial`，本项目用的是等价的**指数竞争法**：

```text
给每个词抽一个 Eᵢ ~ Exp(1)，然后取 argmax(pᵢ / Eᵢ)
```

**为什么等价**：`Eᵢ / pᵢ` 服从速率为 `pᵢ` 的指数分布。多个独立指数分布“谁最先到达”，第 `i` 个获胜的概率正好是 `pᵢ / Σp = pᵢ`。取 `pᵢ / Eᵢ` 最大，就是取 `Eᵢ / pᵢ` 最小，也就是最先到达的那个。

好处是只用 `argmax`，向量化彻底，也便于 `torch.compile` 优化。

文件：`nanovllm/layers/sampler.py`

<!-- file: nanovllm/layers/sampler.py -->
```python
import torch
from torch import nn


class Sampler(nn.Module):

    # torch.compile：第一次调用时把函数编译优化，之后更快（第 10 章详讲）
    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # logits [B, V]，temperatures [B]。unsqueeze 成 [B, 1] 才能按行相除。
        # float()：logits 可能是 bf16，分辨率不够；同时得到一份可原地修改的新张量
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        # exponential_(1)：填充 Exp(1) 随机数。
        # clamp_min_(1e-10)：防止 E 恰好为 0 导致除零
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
```

### 逐行看 Sampler

用两条序列、词表大小 3 的例子，看每一步的形状：

```text
logits        [2, 3]   [[0.0, 5.0, 1.0], [9.0, 0.0, 1.0]]
temperatures  [2]      [0.6, 1.0]
```

| 代码 | 结果 | 为什么 |
|---|---|---|
| `temperatures.unsqueeze(dim=1)` | `[2, 1]` | 变成一列，才能让第 i 行除以第 i 个温度（广播） |
| `logits.float().div_(...)` | `[2, 3]` | 每行除以自己的温度。同一批的请求温度可以不同，这就是 `SamplingParams` 属于请求的原因 |
| `torch.softmax(logits, dim=-1)` | `[2, 3]` | 每行变成总和为 1 的概率 |
| `torch.empty_like(probs).exponential_(1)` | `[2, 3]` | 每个词抽一个随机数 E。`empty_like` 不初始化，反正马上要填 |
| `.clamp_min_(1e-10)` | `[2, 3]` | E 理论上可能为 0，除以 0 会出错 |
| `probs.div_(E).argmax(dim=-1)` | `[2]` | 每行选 `p/E` 最大的词，一次得到整批的结果 |

**为什么是 `nn.Module`？** 它没有参数，写成函数也可以。写成 Module 只是和其他层保持一致，用法都是 `sampler(...)`。

**温度为什么有效？** 除以小于 1 的数，分数差距被放大，softmax 后高分词的概率更大，结果更确定。温度大于 1 时正好相反。

## 3.2 logits 从模型来：先看模型长什么样

模型不是凭空设计的，**我们要加载别人训练好的权重**，所以结构必须和权重文件对得上。直接看权重文件，就能知道模型的结构：

文件：`tests/inspect_model.py`

<!-- file: tests/inspect_model.py -->
```python
import os
from glob import glob
from safetensors import safe_open
from transformers import AutoConfig

path = os.path.expanduser(os.environ.get("MODEL_PATH", "~/huggingface/Qwen3-0.6B/"))
print(AutoConfig.from_pretrained(path))

for file in glob(os.path.join(path, "*.safetensors")):
    with safe_open(file, "pt", "cpu") as f:
        for name in sorted(f.keys()):
            # 只打印全局权重和第 0 层，其他层结构完全相同
            if ".layers." not in name or ".layers.0." in name:
                print(f"{name:55s} {f.get_slice(name).get_shape()}")
```

```bash
python tests/inspect_model.py
```

把权重名字按点号展开，得到一棵树。**这棵树就是模型结构，也是我们要写的类：**

```text
model                                          Qwen3Model
├─ embed_tokens.weight        [151936, 1024]     词嵌入
├─ layers.0 ~ layers.27                          Qwen3DecoderLayer
│  ├─ input_layernorm.weight            [1024]
│  ├─ self_attn                                  Qwen3Attention
│  │  ├─ q_proj.weight          [2048, 1024]
│  │  ├─ k_proj.weight          [1024, 1024]
│  │  ├─ v_proj.weight          [1024, 1024]
│  │  ├─ o_proj.weight          [1024, 2048]
│  │  ├─ q_norm.weight                  [128]
│  │  └─ k_norm.weight                  [128]
│  ├─ post_attention_layernorm.weight   [1024]
│  └─ mlp                                        Qwen3MLP
│     ├─ gate_proj.weight       [3072, 1024]
│     ├─ up_proj.weight         [3072, 1024]
│     └─ down_proj.weight       [1024, 3072]
└─ norm.weight                          [1024]
```

配置里的关键字段：

| 字段 | 值 | 含义 |
|---|---|---|
| `vocab_size` | 151936 | 词表大小 |
| `hidden_size` | 1024 | 每个 token 的隐藏向量维度 |
| `num_hidden_layers` | 28 | Decoder 层数 |
| `intermediate_size` | 3072 | MLP 中间层宽度 |
| `tie_word_embeddings` | true | 输出层和词嵌入共用权重 |
| `dtype` | bfloat16 | 每个参数 2 字节 |

**权重里没有 `lm_head`**，因为它和 `embed_tokens` 共用权重。

现在**从树根往下写**。

## 3.3 最外层：Qwen3ForCausalLM

`ModelRunner` 需要 `logits = 模型(token)`。这要分成两段：主体把 token 变成隐藏向量，输出头把隐藏向量变成 logits。

**先写 import**。这里面大部分还不存在。没关系，**这份 import 就是接下来的待办清单**，本章和下一章会一个个写出来。

文件：`nanovllm/models/qwen3.py`

<!-- file: nanovllm/models/qwen3.py -->
```python
import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen3Config

from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import QKVParallelLinear, MergedColumnParallelLinear, RowParallelLinear
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class Qwen3ForCausalLM(nn.Module):
    # 权重文件里的名字片段 → (我们的参数名片段, 子权重标识)。作用见 3.6 节和 4.6 节
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config
    ) -> None:
        super().__init__()
        # self.model 这个属性名不能乱取：
        # 它要拼出权重文件里的路径，如 model.layers.0.mlp.down_proj.weight
        self.model = Qwen3Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            # 共用同一块显存。权重文件里没有 lm_head，加载 embed_tokens 时一并生效
            self.lm_head.weight.data = self.model.embed_tokens.weight.data

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
```

**逐行看 Qwen3ForCausalLM：**

- `packed_modules_mapping`：写在类上，加载器用 `getattr(model, ...)` 取到它。模型自己声明“我把哪些权重合并了”，加载器就不用写死任何模型的知识。
- `self.lm_head.weight.data = self.model.embed_tokens.weight.data`：让两个 `Parameter` 指向同一块存储。之后加载 `embed_tokens` 时用 `copy_` 原地写入，`lm_head` 也就有了值。如果没有这行，`lm_head` 就是 `torch.empty` 留下的垃圾数据，因为权重文件里根本没有它。
- `forward` 只调用 `self.model`，`compute_logits` 单独调用 `self.lm_head`：把“算隐藏向量”和“算 logits”拆成两步。

**为什么有 `positions` 参数？** 模型要知道每个 token 在序列的第几位（“狗咬人”和“人咬狗”不一样）。这个信息最终由注意力使用，下一章讲。

**为什么 `forward` 和 `compute_logits` 分开？** 词表有 15 万，算 logits 很贵，不一定每个位置都要算。第 5 章和第 10 章会用到这个分离。

**为什么用 `dist`？** 第 11 章多卡时要用。现在你只需要知道：`world_size=1` 时，这些并行相关的代码都没有效果。

这里用到的新类：`Qwen3Model`、`ParallelLMHead`。继续往下。

## 3.4 第二层：Qwen3Model

数据流很直接：`token → 嵌入 → 经过每一层 → 最终归一化`。

把这个类**追加**到 `qwen3.py` 末尾（类的定义顺序不要紧，运行时才去查找名字）：

<!-- append: nanovllm/models/qwen3.py -->
```python
class Qwen3Model(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        # 把最后一层 MLP 的输出加进残差，再归一化。residual 为什么这样传？看下一节
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
```

**逐行看 Qwen3Model：**

- `nn.ModuleList`：不能用普通的 Python 列表。普通列表里的层不会被注册，参数名就不会有 `layers.0.` 这样的前缀，`get_parameter` 也就找不到它们。
- `for layer in self.layers`：每一层的输出是下一层的输入。`positions` 每层都一样，所以直接传。
- `residual = None`：第 0 层之前还没有残差。下一节解释这个写法。

新需求：`VocabParallelEmbedding`、`ParallelLMHead`。它们都是“token 和向量互转”，放在一个文件。权重形状都是 `[词表, 隐藏维度]`，所以输出头继承嵌入层，这也是它们能共用权重的原因。

文件：`nanovllm/layers/embed_head.py`（第 11 章升级并行）

<!-- file: nanovllm/layers/embed_head.py -->
```python
import torch
from torch import nn
import torch.nn.functional as F


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        # 用 torch.empty 不初始化：反正马上要被权重覆盖，省下初始化时间
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))

    def forward(self, x: torch.Tensor):
        return F.embedding(x, self.weight)


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        # [N, 隐藏维度] × [词表, 隐藏维度]ᵀ → [N, 词表]
        return F.linear(x, self.weight)
```

**逐行看嵌入和输出头：**

- `F.embedding(x, weight)` 就是查表：`x` 形状 `[N]`，每个元素是 token id，结果是 `weight[x]`，形状 `[N, 1024]`。
- `F.linear(x, weight)` 是 `x @ weightᵀ`：`[N, 1024] × [1024, 151936] = [N, 151936]`。每个 logit 等于隐藏向量和这个词的向量做点积，点积越大，这个词的分数越高。这也是嵌入层和输出头能共用权重的直观原因。
- `assert not bias`：只实现了没有偏置的情况，提前拦住不支持的配置。

类名里的 `Parallel` 现在没有意义，第 11 章才会用到。

## 3.5 第三层：DecoderLayer，以及为什么有 residual

Qwen3 的每一层是 **Pre-Norm** 结构：

```text
x = x + Attention(Norm(x))
x = x + MLP(Norm(x))
```

（为什么要加回 `x`？这叫残差连接，让深层网络能够训练。推理时我们只要照做。）

最直接的写法是每步先加再归一化，两步都要做类型转换（bf16 ↔ float32）。本项目把“相加 + 归一化”合并成一次调用，`residual` 把上一步的结果带到下一个 norm 里：

```text
第 0 层入口：  x=嵌入, residual=None
  input_layernorm(x)                → x=norm(嵌入), residual=嵌入
  self_attn(x)                      → x=attn 输出
  post_attention_layernorm(x, r)    → residual=r+x, x=norm(r+x)
  mlp(x)                            → x=mlp 输出，尚未加到 residual
第 1 层入口：
  input_layernorm(x, residual)      → residual=r+x, x=norm(r+x)    先把上一层的 MLP 结果加上
  ...
最后：
  model.norm(x, residual)           → 把最后一层的 MLP 结果加上，再归一化
```

这正好解释了上一节 `Qwen3Model` 里为什么要传 `residual`。

继续追加到 `qwen3.py`：

<!-- append: nanovllm/models/qwen3.py -->
```python
class Qwen3DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen3Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            # getattr 带默认值：不同版本的配置，字段可能缺失
            qkv_bias=getattr(config, 'attention_bias', True),
            head_dim=getattr(config, 'head_dim', None),
            rope_theta=getattr(config, "rope_theta", 1000000),
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.mlp = Qwen3MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            # 右边先求值：左边的 residual 拿到的是归一化前的 hidden_states
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual
```

**逐行看 Qwen3DecoderLayer：**

- `config.xxx`：结构参数全部来自 `config.json`，代码里不写死数字。
- `getattr(config, 'attention_bias', True)`：有些配置缺这个字段，给一个默认值。Qwen3-0.6B 的配置里它是 `false`。
- `input_layernorm`、`post_attention_layernorm`：**属性名必须和权重文件一致**，改了就加载不上。
- `if residual is None`：只有第 0 层走这个分支。这时没有什么可以加，只需要归一化。
- `hidden_states, residual = norm(hidden_states), hidden_states`：Python 先算完等号右边，所以 `residual` 拿到的是**旧的** `hidden_states`。

这一层需要三样东西：`RMSNorm`、`Qwen3MLP`、`Qwen3Attention`。`Qwen3Attention` 是最复杂的，留给下一章。先写另外两个。

### RMSNorm

现在我们很清楚它要提供什么了：**两种调用方式**。

```text
norm(x)            → x / sqrt(mean(x²) + eps) × weight
norm(x, residual)  → 先相加，再归一化，同时返回新的 residual
```

**为什么转成 float32 计算**：模型参数是 bf16，只有约 3 位有效数字。先平方再求和再开方，用 bf16 精度损失明显。

文件：`nanovllm/layers/layernorm.py`

<!-- file: nanovllm/layers/layernorm.py -->
```python
import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps                                       # 防止除零
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 每维一个可学习缩放

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype          # 记住原类型，算完转回去
        x = x.float()                 # bf16 → fp32，会产生新张量
        var = x.pow(2).mean(dim=-1, keepdim=True)  # keepdim 才能和 x 广播
        # rsqrt(a) = 1 / sqrt(a)。mul_ 原地乘，少一次内存分配
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())  # x + residual，用 fp32 算，减小舍入误差
        residual = x.to(orig_dtype)           # 相加结果就是新的残差
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    # forward 不加 compile，它只是个分发器
    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
```

**逐行看 RMSNorm（以 `rms_forward` 为例）：**

| 代码 | 做了什么 | 为什么 |
|---|---|---|
| `orig_dtype = x.dtype` | 记下 bf16 | 最后要转回去，后面的层都按 bf16 工作 |
| `x = x.float()` | 转 fp32，得到新张量 | 平方再求和对精度敏感 |
| `x.pow(2).mean(dim=-1, keepdim=True)` | `[N, d] → [N, 1]` | 每个 token 一个方差。`keepdim` 让它还能和 `x` 广播 |
| `x.mul_(torch.rsqrt(var + self.eps))` | `x / sqrt(方差 + eps)` | `rsqrt` 就是平方根的倒数，`eps` 防止除零 |
| `x.to(orig_dtype).mul_(self.weight)` | 转回 bf16，每维乘一个权重 | `self.weight` 是 bf16，类型要一致 |

**`mul_` 为什么不会改坏输入？** 输入是 bf16 时，`x.float()` 产生了新张量，原地改的是新张量。下面的陷阱说的就是输入本身是 fp32 的情况。

`Q`、`K` 的形状是 `[N, H, d]`，`dim=-1` 同样成立，所以 `q_norm`、`k_norm` 能用同一个类。

> ⚠️ **一个陷阱：这段代码只在模型不是 float32 时安全。** 张量已经是 float32 时，`x.float()` 和 `x.to(float32)` 都**直接返回原张量，不做拷贝**。于是 `residual` 和 `x` 是同一块内存，后面的 `x.mul_` 会把残差也改掉，结果悄悄出错。Qwen3-0.6B 是 bf16，所以不受影响。修复方法见第 12 章。

## 3.6 第四层：Qwen3MLP

看权重树：`gate_proj`、`up_proj`、`down_proj`。运算是：

```text
output = down( SiLU(gate(x)) × up(x) )
```

追加到 `qwen3.py`：

<!-- append: nanovllm/models/qwen3.py -->
```python
class Qwen3MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,   # gate 和 up 各一份
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        # 只实现了 silu，遇到别的激活函数就早点报错，好过静默算错
        assert hidden_act == "silu"
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)   # [N, 1024] → [N, 6144]
        x = self.act_fn(gate_up)         # → [N, 3072]
        x = self.down_proj(x)            # → [N, 1024]
        return x
```

**这里有个问题：权重文件里是 `gate_proj` 和 `up_proj` 两个，代码却写成了一个 `gate_up_proj`。为什么？**

两个线性层就是两次矩阵乘法。输入相同的情况下，合并成一个大矩阵，一次算完再切开，会更快。代价是加载权重时要拼回去。

于是需要两样东西：能把权重放进正确区间的线性层，和能把结果切开的激活函数。

文件：`nanovllm/layers/activation.py`

<!-- file: nanovllm/layers/activation.py -->
```python
import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # chunk(2, -1)：沿最后一维平均切两份，前一半是 gate，后一半是 up
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
```

**合并的线性层怎么加载权重？** 权重文件里的 `gate_proj`、`up_proj` 要放进一个大矩阵的不同区间（`F.linear` 的权重是 `[输出, 输入]`）：

```text
gate_proj.weight  [3072, 1024]  ─┐
                                  ├─→ gate_up_proj.weight [6144, 1024]
up_proj.weight    [3072, 1024]  ─┘    第 0 维：gate 占 0~3071，up 占 3072~6143
```

所以每个参数上挂一个 `weight_loader` 函数，告诉加载器“这个权重要放到我身上的哪里”。

文件：`nanovllm/layers/linear.py`（第 11 章升级并行）

<!-- file: nanovllm/layers/linear.py -->
```python
import torch
from torch import nn
import torch.nn.functional as F


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__()
        # 形状是 [输出, 输入]，这是 F.linear 的约定
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        # nn.Parameter 是对象，可以挂额外属性
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            # 注册为 None，self.bias 这个属性仍然存在，F.linear 能直接用
            self.register_parameter("bias", None)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(LinearBase):
    # 合并多个输出。output_sizes=[3072, 3072] 表示 gate 和 up

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        # loaded_shard_id：第几个子权重（0=gate，1=up）
        shard_offset = sum(self.output_sizes[:loaded_shard_id])
        shard_size = self.output_sizes[loaded_shard_id]
        # narrow(维度, 起点, 长度)：返回共享内存的视图，往视图 copy_ 就写进了原参数
        param.data.narrow(0, shard_offset, shard_size).copy_(loaded_weight)


class RowParallelLinear(LinearBase):
    # 单卡时只是普通线性层。第 11 章会看到它和其他层的差别
    pass
```

**逐行看 linear.py：**

- `self.weight.weight_loader = self.weight_loader`：把**当前对象的方法**挂到参数上。Python 按实际类型查找方法，所以在 `LinearBase.__init__` 里写这一行，拿到的也是子类覆盖后的版本。加载器不需要知道这是哪种层，只管调用 `param.weight_loader(...)`。
- `self.register_parameter("bias", None)`：`self.bias` 依然存在，值是 `None`。`F.linear(x, w, None)` 正好表示没有偏置，`forward` 不用写分支。
- `self.output_sizes = output_sizes` 写在 `super().__init__` 之前：`nn.Module` 只禁止在初始化之前给它赋值参数或子模块，普通的列表没关系。
- `param.data.narrow(0, offset, size).copy_(loaded_weight)`：`narrow` 得到一段视图，`copy_` 写进视图就是写进大矩阵。

用数字走一遍 `up_proj` 的加载（`output_sizes=[3072, 3072]`，`loaded_shard_id=1`）：

```text
shard_offset = sum([3072]) = 3072
shard_size   = 3072
narrow(0, 3072, 3072)  → 大矩阵的第 3072～6143 行
```

## 检查点

模型还差注意力，不能整体运行。但已写完的零件可以单独验证。

文件：`tests/test_parts.py`

<!-- file: tests/test_parts.py -->
```python
import torch
from nanovllm.layers.activation import SiluAndMul
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.linear import MergedColumnParallelLinear
from nanovllm.layers.sampler import Sampler

# 1. Sampler：温度极低时等于 argmax；每行有各自的温度
logits = torch.tensor([[0.0, 5.0, 1.0], [9.0, 0.0, 1.0]])
assert Sampler()(logits, torch.tensor([1e-3, 1e-3])).tolist() == [1, 0]
# 正常温度下，抽样频率应接近 softmax 概率
n = 20000
got = Sampler()(torch.tensor([[0.0, 1.0]]).repeat(n, 1), torch.ones(n)).float().mean().item()
assert abs(got - torch.tensor([0.0, 1.0]).softmax(0)[1].item()) < 0.02

# 2. RMSNorm：权重全 1 时，输出的均方根应该约等于 1
x = torch.randn(4, 8, dtype=torch.bfloat16)
norm = RMSNorm(8).to(torch.bfloat16)
y = norm(x)
rms = y.float().pow(2).mean(-1).sqrt()
assert torch.allclose(rms, torch.ones(4), atol=0.02), rms
# bf16 输入不应被原地修改，否则残差会错
before = x.clone()
norm(x)
assert torch.equal(x, before)
# 残差版：返回的 residual 应该等于 x + r
r = torch.randn(4, 8, dtype=torch.bfloat16)
_, new_r = norm(x, r)
assert torch.allclose(new_r.float(), x.float() + r.float(), atol=0.02)

# 3. SiluAndMul：维度减半
assert SiluAndMul()(torch.randn(2, 8)).shape == (2, 4)

# 4. 合并投影：子权重必须落到正确的区间
proj = MergedColumnParallelLinear(4, [2, 2])
assert proj.weight.shape == (4, 4)
with torch.no_grad():
    proj.weight.weight_loader(proj.weight, torch.full((2, 4), 1.0), 0)
    proj.weight.weight_loader(proj.weight, torch.full((2, 4), 2.0), 1)
assert proj.weight[:, 0].tolist() == [1, 1, 2, 2]
print("OK")
```

```bash
python tests/test_parts.py
```

## 本章小结

```text
ModelRunner.run
├─ Sampler ✓
└─ Qwen3ForCausalLM ✓
   └─ Qwen3Model ✓
      ├─ Embedding ✓  RMSNorm ✓
      └─ Qwen3DecoderLayer ✓
         ├─ Qwen3MLP ✓
         └─ Qwen3Attention ✗  ← 最后的缺口
```

**还缺最核心的一块：注意力。** 下一章补上，然后就能把真模型接回 `ModelRunner`。
