# 第 5 章：请求为什么要排队？批处理

**上一章留下的问题**：`ModelRunner.run` 里有个 `for seq in seqs`，两条请求一条接一条算。GPU 擅长并行，这太浪费了。

**本章要回答**：怎么把多条请求放进模型的一次计算？

```text
先看哪一层要改：
LLMEngine     不用改     run(seqs) 接口没变
Scheduler     不用改
ModelRunner   要改       把多条序列打包成张量        ← 从这里开始
  └ 模型      要改一点   注意力不能跨序列、输出头只要最后一个位置
```

外层完全不用动。这就是第 2 章把接口定义好的好处。

## 本章任务卡

| | |
|---|---|
| **要解决什么** | `run` 里的 `for seq in seqs` 让请求排队算 |
| **要实现什么** | `prepare_prefill`、`prepare_sample`、`run_model`、`Context`、输出头取最后一个位置、用 FlashAttention 的 `Attention` |
| **要调用什么** | `torch.tensor(pin_memory=True)`、`.cuda(non_blocking=True)`、`flash_attn_varlen_func`、`.tolist()` |
| **为什么这样做** | 把不同长度的序列首尾相接，不需要 padding。大多数层是逐 token 独立计算的，所以只需要改真正关心序列边界的两处 |

## 5.1 难点：序列长度不同

假设两条序列，长度 3 和 5。

**传统做法是补齐（padding）**：

```text
[a0 a1 a2 __ __]
[b0 b1 b2 b3 b4]     形状 [2, 5]，__ 的部分白算
```

一批序列长度差异越大，白算得越多。

**本项目的做法是首尾相接**：

```text
input_ids  = [a0 a1 a2 b0 b1 b2 b3 b4]     一维，长度 8，没有一个 padding
positions  = [ 0  1  2  0  1  2  3  4]     每条序列各自从 0 开始！
cu_seqlens = [0, 3, 8]                     第 i 条序列占 [cu[i], cu[i+1])
```

`cu` 是 cumulative（累积）的缩写。

**这样模型要改多少？** 过一遍第 3、4 章写的每个部件：

| 部件 | 要知道序列边界吗？ | 结论 |
|---|---|---|
| 词嵌入、线性层、Norm、MLP | 不需要，逐 token 独立 | 不用改 |
| RoPE | 不需要，靠 `positions` 查表 | 不用改 |
| **Attention** | **需要**：不能看到别的序列 | 要改 |
| **输出头** | **需要**：每条序列只要最后一个位置的 logits | 要改 |

这也解释了为什么第 3、4 章的形状都写成 `[N, ...]`，没有 batch 维。

## 5.2 从外往里：先改 ModelRunner.run

新的 `run` 分成三步：准备张量、跑模型、采样。

替换 `nanovllm/engine/model_runner.py` 里 `ModelRunner` 的 `run` 方法，并加入新方法：

<!-- method: nanovllm/engine/model_runner.py :: ModelRunner -->
```python
    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens = [0]
        max_seqlen = 0
        for seq in seqs:
            input_ids.extend(seq.token_ids)
            positions.extend(range(len(seq)))
            cu_seqlens.append(cu_seqlens[-1] + len(seq))
            max_seqlen = max(len(seq), max_seqlen)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        # FlashAttention 要求这些张量是 int32
        cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(cu_seqlens, cu_seqlens, max_seqlen, max_seqlen)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    # inference_mode：不记录反向传播信息，省时省显存。推理不需要梯度
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor):
        return self.model.compute_logits(self.model(input_ids, positions))

    def run(self, seqs: list[Sequence]) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs)
        temperatures = self.prepare_sample(seqs)
        logits = self.run_model(input_ids, positions)
        # tolist：转成 Python 整数。这里要等 GPU 算完，是一轮的同步点
        token_ids = self.sampler(logits, temperatures).tolist()
        reset_context()
        return token_ids
```

在文件头部的 `load_model` 导入下面，加入：

<!-- insert-after: nanovllm/engine/model_runner.py :: from nanovllm.utils.loader import load_model -->
```python
from nanovllm.utils.context import set_context, reset_context
```

### 走一遍 prepare_prefill

用测试里的三条序列：`a=[11,22,33]`、`b=[44,55,66,77,88]`、`c=[99]`。

| 处理完 | `input_ids` | `positions` | `cu_seqlens` | `max_seqlen` |
|---|---|---|---|---|
| 初始 | `[]` | `[]` | `[0]` | 0 |
| a | `[11,22,33]` | `[0,1,2]` | `[0,3]` | 3 |
| b | `[11,22,33,44,55,66,77,88]` | `[0,1,2,0,1,2,3,4]` | `[0,3,8]` | 5 |
| c | `[...,88,99]` | `[...,4,0]` | `[0,3,8,9]` | 5 |

- `input_ids`、`positions` 用 `int64`：它们要拿去当下标（查嵌入表、查 RoPE 表），`int64` 是 PyTorch 下标的常规类型。
- `cu_seqlens` 用 `int32`：这是 FlashAttention 的要求。
- `set_context(cu_seqlens, cu_seqlens, ...)`：现在 Q 和 K 相同，下面会讲。

**`.tolist()` 为什么是同步点？** GPU 任务是异步提交的，`run_model` 返回时计算可能还没做完。要把结果变成 Python 整数，CPU 就必须等 GPU 算完。

**`prepare_prefill` 这个名字**是“把序列灌入模型”的意思，下一章你会看到它的准确含义。

**`pin_memory=True` + `.cuda(non_blocking=True)` 是什么**：Python 列表先转成 CPU 张量。固定在物理内存（pin）后，GPU 可以用 DMA 直接异步拷贝，CPU 不用等待。这条链路每轮都走，值得优化。

## 5.3 新问题：cu_seqlens 怎么传进模型深处？

`Attention` 和输出头要用 `cu_seqlens`，但是它们在调用链的最深处：

```text
Qwen3ForCausalLM → Qwen3Model → Qwen3DecoderLayer → Qwen3Attention → Attention
```

把参数一层层传下去？中间几层根本用不到。而且第 10 章的 CUDA Graph 只捕获 `model(input_ids, positions)` 这个调用，签名越简单越好。

所以用一个**全局对象**：调用模型前 `set_context`，模型深处 `get_context`，用完 `reset_context`。

文件：`nanovllm/utils/context.py`

<!-- file: nanovllm/utils/context.py -->
```python
from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0):
    global _CONTEXT
    # 每次新建对象，确保不会残留上一轮的字段
    _CONTEXT = Context(cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
```

**逐行看 context.py：**

- `_CONTEXT = Context()`：模块级变量。整个进程只跑一个模型，一份全局状态就够了。
- `global _CONTEXT`：函数里给模块级变量重新赋值时必须声明，否则 Python 会把它当成局部变量。
- `set_context` 每次新建对象，而不是修改旧对象：没有传的字段自动回到默认值，不会留下上一轮的数据。
- `reset_context`：一轮结束就清掉，也让张量的引用及时释放。

**为什么 Q、K 各有一份？** 目前两者相同。第 8 章它们就不同了。FlashAttention 的接口本来就要求分别传。

## 5.4 往里一层：输出头只要最后一个位置

现在输入可能有几千个 token，但每条序列**只需要最后一个位置**的 logits 来预测下一个词。词表有 15 万维，全算一遍浪费显存和时间。

这就是第 3 章把 `compute_logits` 分开的好处。`cu_seqlens_q = [0, 3, 8]` 时，`cu_seqlens_q[1:] - 1 = [2, 7]` 正好是每条序列最后一个 token 的下标。

替换 `nanovllm/layers/embed_head.py` 里 `ParallelLMHead` 的 `forward`：

<!-- method: nanovllm/layers/embed_head.py :: ParallelLMHead -->
```python
    def forward(self, x: torch.Tensor):
        context = get_context()
        last_indices = context.cu_seqlens_q[1:] - 1
        x = x[last_indices].contiguous()
        return F.linear(x, self.weight)
```

在文件头部的导入后面加入：

<!-- insert-after: nanovllm/layers/embed_head.py :: import torch.nn.functional as F -->
```python

from nanovllm.utils.context import get_context
```

**走一遍：** `cu_seqlens_q=[0,3,8,9]` 时，`cu_seqlens_q[1:] - 1 = [2,7,8]`。`x` 的形状是 `[9, 1024]`，`x[[2,7,8]]` 得到 `[3, 1024]`，输出 logits 就是 `[3, V]`，正好每条序列一行。这也满足了 `run` 的约定：输出和输入的序列一一对应。

`.contiguous()` 是防御性写法：用下标张量索引本身就会产生新张量，这里其实已经是连续的了。

## 5.5 最里层：能处理变长序列的注意力

第 4 章的 `scaled_dot_product_attention` 不认识 `cu_seqlens`。我们换成 **FlashAttention**：一个高性能注意力算子库，支持变长序列，还原生支持 GQA（Q 头数是 K/V 头数的倍数即可），映射关系和第 4 章的手写版相同。

文件：`nanovllm/layers/attention.py`（替换第 4 章的版本）

<!-- file: nanovllm/layers/attention.py -->
```python
import torch
from torch import nn

from flash_attn import flash_attn_varlen_func
from nanovllm.utils.context import get_context


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        o = flash_attn_varlen_func(q, k, v,
                                   max_seqlen_q=context.max_seqlen_q,
                                   cu_seqlens_q=context.cu_seqlens_q,
                                   max_seqlen_k=context.max_seqlen_k,
                                   cu_seqlens_k=context.cu_seqlens_k,
                                   softmax_scale=self.scale,
                                   causal=True)
        return o
```

**flash_attn_varlen_func 的每个参数：**

| 参数 | 例子里的值 | 含义 |
|---|---|---|
| `q` | `[9, 16, 128]` | 所有序列的 Query 首尾相接 |
| `k`、`v` | `[9, 8, 128]` | 所有序列的 Key、Value 首尾相接 |
| `cu_seqlens_q` | `[0,3,8,9]` | Q 里每条序列的边界 |
| `cu_seqlens_k` | `[0,3,8,9]` | K、V 里每条序列的边界 |
| `max_seqlen_q`、`max_seqlen_k` | 5、5 | 最长序列的长度，算子用它决定内部处理的方式 |
| `softmax_scale` | `self.scale` | 公式里的 `1/sqrt(d)` |
| `causal=True` | | 不看未来 |

输出形状和 `q` 相同：`[9, 16, 128]`。

它负责了点积、掩码、softmax、加权求和这些事，所以我们不需要自己写注意力的计算。

> ⚠️ **第 4 章的 `tests/test_model_vs_hf.py` 从现在起不能再运行了。** 它没有设置 context，而输出头现在依赖 context。这是正常的，它的任务已经完成。

## 检查点

**批处理的关键性质：一条序列的结果，不应该受同一批其他序列影响。**

文件：`tests/test_batching.py`

<!-- file: tests/test_batching.py -->
```python
import os
from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

path = os.path.expanduser(os.environ.get("MODEL_PATH", "~/huggingface/Qwen3-0.6B/"))
runner = ModelRunner(Config(path))

# 极低温度：采样几乎等于 argmax，结果是确定的
sp = SamplingParams(temperature=1e-4)
a, b, c = [11, 22, 33], [44, 55, 66, 77, 88], [99]
alone = [runner.run([Sequence(p, sp)])[0] for p in (a, b, c)]
together = runner.run([Sequence(p, sp) for p in (a, b, c)])
print("逐条:", alone, "一批:", together)
assert alone == together, "结果被同一批的其他序列影响了"
runner.exit()
print("OK")
```

```bash
python tests/test_batching.py
```

再运行入口和量尺：

```bash
python example.py
```

```bash
python tests/time_steps.py
```

## 本章小结

和上一章比，**同样的两条请求，耗时明显下降**，因为它们现在同时计算。

但另一个症状还在：看 `time_steps.py` 的输出，**越到后面每轮越慢**。下一章解决。
