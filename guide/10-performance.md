# 第 10 章：时间都花在哪了？CUDA Graph

**上一章留下的问题**：逻辑都对了，现在怎么变快？

**本章要回答**：先找到慢在哪，再对症下药。这一章不增加新能力，只改 `ModelRunner`。

## 本章任务卡

| | |
|---|---|
| **要解决什么** | Decode 每轮数据很少，时间却花在 CPU 逐个提交算子上 |
| **要实现什么** | `bench.py`、`capture_cudagraph`、走图的 `run_model`、退出时释放图 |
| **要调用什么** | `torch.cuda.CUDAGraph`、`torch.cuda.graph`、`graph.pool`、`graph.replay`、`torch.compile`、`torch.cuda.synchronize` |
| **为什么这样做** | 只有 Decode 值得用图，它的结构每轮都不变。图要求地址和形状固定，所以用固定缓冲区和 batch 档位去适配 |

## 10.1 先写一把量尺

要优化，先要能衡量。写一个压测脚本：256 条随机长度的请求。

文件：`bench.py`

<!-- file: bench.py -->
```python
import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams
# from vllm import LLM, SamplingParams


def main():
    seed(0)     # 固定随机种子，方便不同引擎、不同版本之间对比
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    # 随机 token：避免前缀缓存影响结果。长度也随机，这才能考验动态批处理
    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    # ignore_eos=True：确保每条都生成到 max_tokens，总 token 数才是确定的
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]
    # uncomment the following line for vllm
    # prompt_token_ids = [dict(prompt_token_ids=p) for p in prompt_token_ids]

    # 短请求预热：触发采样器编译等一次性开销，不计入结果
    llm.generate(["Benchmark: "], SamplingParams())
    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = (time.time() - t)
    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")


if __name__ == "__main__":
    main()
```

先记下现在的成绩（这时还没有实现图模式，`enforce_eager` 不起作用）：

```bash
python bench.py
```

用 profiler 看看时间去了哪里：

```bash
python -m cProfile -s cumtime bench.py
```

## 10.2 慢在哪里

Decode 阶段每轮只算很少的 token，但模型有 28 层，每层有很多算子。每个算子都要经历：Python 调用 → PyTorch 调度 → 提交 GPU 任务。这些都是 **CPU 开销**，和算多少数据无关。

| 阶段 | 数据量 | 主要瓶颈 | 优化手段 |
|---|---|---|---|
| Prefill | 大 | GPU 计算 | CPU 开销占比很小，不用管 |
| Decode | 小（每条 1 个 token） | CPU 调度 | **CUDA Graph** |
| 小算子（Norm、RoPE、采样） | | 反复创建中间张量 | **`torch.compile`** |

### torch.compile：其实已经加了

前面章节已经把 `@torch.compile` 加在这些函数上了：

| 函数 | 文件 |
|---|---|
| `SiluAndMul.forward` | `activation.py` |
| `RMSNorm.rms_forward`、`add_rms_forward` | `layernorm.py` |
| `RotaryEmbedding.forward` | `rotary_embedding.py` |
| `Sampler.forward` | `sampler.py` |

**为什么不直接编译整个模型？** 模型里有 FlashAttention 和 Triton，都是 `torch.compile` 无法追踪的外部代码，还有全局 context 这样的 Python 分支。只编译这些纯粹的张量函数，既能融合算子，又不会遇到这些麻烦。

**这也解释了第一次调用为什么慢**：要做编译。这正是预热的另一个作用。`bench.py` 里也先跑一次短请求，才开始计时。

## 10.3 CUDA Graph 的原理

```text
普通执行：每轮，CPU 逐个提交算子（28 层 × 每层若干算子）
CUDA Graph：
  捕获：跑一遍，记录所有 GPU 操作，形成一张图
  回放：graph.replay()，一个调用，GPU 自己把整张图跑完
```

**代价是限制很严**：图里记录的是**固定的内存地址和固定的形状**。这带来四条设计约束：

| 约束 | 对应的做法 |
|---|---|
| 地址不能变 | 预先分配固定的输入缓冲区（`graph_vars`），每轮**往里写**，不换张量 |
| 形状不能变 | 按 batch size 分档捕获：`[1, 2, 4, 8, 16, 32, ..., 512]` |
| 实际 batch 比档位小 | 向上取档位，多余的行用 padding |
| Python 分支不能变 | 只捕获 Decode；Prefill 走普通路径 |

**这就是 context 的另一个价值**：捕获时，Attention 读到的 `slot_mapping`、`context_lens`、`block_tables` 是 `graph_vars` 里的切片，它们的地址被烧进图里。回放前修改这些张量的内容，图自然读到新数据。而 `model(input_ids, positions)` 的签名始终这么简单。

**padding 行如何不造成破坏：**

| 字段 | padding 行的值 | 效果 |
|---|---|---|
| `slot_mapping` | -1 | KV 写入 kernel 看到 -1 直接跳过（第 6 章留的伏笔），不写脏缓存 |
| `context_lens` | 0 | 读取长度为 0 |
| 输出 | 忽略 | 只取 `outputs[:bs]` |

**为什么 `compute_logits` 不放进图里？** 这就是第 3 章把它和 `forward` 分开的第二个理由。输出只有 `outputs[:bs]` 有效，没必要给 padding 行算 15 万维 logits。放在图外，它的 `F.linear` 只算有效的 `bs` 行。

## 10.4 从外往里改

**① 回放路径：`run_model` 选择走图还是走普通路径。**

替换 `run_model`：

<!-- method: nanovllm/engine/model_runner.py :: ModelRunner -->
```python
    # graph_vars 在 inference mode 里创建，是“推理张量”。
    # 在 inference mode 之外原地修改它们会报错，所以这里的装饰器不能少
    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        # 三种情况走普通路径：Prefill、用户禁用图、batch 超过最大档位
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            # 第一个不小于 bs 的档位。比如 bs=20 用 32 这张图
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            # 所有写入都是往固定地址里写，不换张量
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)        # 先全部标记成“不写”
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            # 第二维只写实际宽度。多余的部分是旧数据，但不会越过 context_lens，读不到
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            # 这一步在图外。context 还没 reset，ParallelLMHead 读到 is_prefill=False
            return self.model.compute_logits(graph_vars["outputs"][:bs])
```

**走一遍：`bs=20`，第 3 轮 Decode**

```text
next(x for x in graph_bs if x >= 20)   → 档位 32，取出 32 这张图

写入固定缓冲区：
  input_ids[:20]     = 这 20 条序列的 last_token
  positions[:20]     = 各自的 len-1
  slot_mapping       = 全部 -1，然后 [:20] 写入真实槽位
  context_lens       = 全部 0，然后 [:20] 写入真实长度
  block_tables[:20, :实际宽度] = 这 20 条的块表

graph.replay()       GPU 跑 32 行。第 20~31 行 slot=-1 不写缓存，读长度 0

compute_logits(outputs[:20])   → 图外只算 20 行 logits
```

第 20～31 行的 `input_ids` 是旧数据，但它们的结果会被丢弃，不会写进缓存，所以没有影响。

**回放时不会走 Python。** 捕获时，`Attention.forward` 里的 `if context.is_prefill` 已经选好了 Decode 分支，这个选择固定在图里。这就是只捕获 Decode 的原因。

导入要多一个 `get_context`：

<!-- replace-line: nanovllm/engine/model_runner.py :: from nanovllm.utils.context import set_context, reset_context -->
```python
from nanovllm.utils.context import set_context, get_context, reset_context
```

**② 这段代码用到的 `self.graphs`、`graph_bs`、`graph_vars`，由捕获时准备好。**

新增方法：

<!-- method: nanovllm/engine/model_runner.py :: ModelRunner -->
```python
    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        # 最大档位。超过 512 的 batch 不走图
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        # 这些就是固定地址的缓冲区。这时默认设备是 cuda
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        # 从大到小捕获：第一张（最大的）图分配的内存池，小图可以复用
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            # 先普通跑一遍：触发 compile、Triton 编译等一次性工作，不让它们混进图里
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
```

**逐行看 capture_cudagraph：**

| 代码 | 为什么 |
|---|---|
| `max_bs = min(max_num_seqs, 512)` | 图不用捕获超过实际需要的大小 |
| `max_num_blocks = (max_model_len + B - 1) // B` | 块表第二维要足够放下最长的序列 |
| `torch.zeros(...)` 那一组 | 这些就是固定地址。设备是 GPU（默认设备还没恢复），类型由 `dtype=` 指定 |
| `outputs = torch.zeros(max_bs, hidden_size)` | 没写 `dtype`，默认类型这时是 bf16，和模型输出一致 |
| `graph_bs = [1,2,4,8] + 16 的倍数` | 档位越密浪费越少，但每张图都要时间和显存。小 batch 多给几档 |
| `graph = torch.cuda.CUDAGraph()` | 一个空的图对象 |
| `set_context(False, slot_mapping=slot_mapping[:bs], ...)` | 切片的地址是固定的。Attention 读 context，这些地址就被录进图里 |
| 第一次 `self.model(...)` | 普通跑一遍，让编译等一次性工作提前做完，不被录进图 |
| `with torch.cuda.graph(graph, self.graph_pool)` | 这个范围内的 GPU 操作被录制，而不是直接执行 |
| `outputs[:bs] = self.model(...)` | 回放时没有 Python 返回值，只能写到固定地址，再从那里读 |
| `graph.pool()` | 第一张图（最大的）留下内存池，后面的图复用，不用各占一份 |
| `torch.cuda.synchronize()` | 等 GPU 做完，再开始下一张图 |
| `self.graph_vars = dict(...)` | 把这些固定缓冲区存起来，`run_model` 每轮往里写 |

**为什么 `input_ids[:bs]` 这样的切片录进图后还能用？** 切片是视图，地址落在固定缓冲区内。之后往缓冲区写入新内容，图读到的就是新内容。

**`@torch.compile` 这个装饰器做了什么？** 第一次调用时分析函数里的张量运算并生成优化版本，之后直接用优化版本。第一次很慢，之后更快。它和 CUDA Graph 解决的问题不同：一个减少中间张量和算子数量，一个减少 CPU 提交的次数，所以两者可以同时用。

**③ 什么时候捕获？** 必须在 `allocate_kv_cache` **之后**，因为图要记录 KV 缓存的地址。还必须在恢复默认设备**之前**，因为缓冲区要建在 GPU 上。

在 `self.allocate_kv_cache()` 下面加入：

<!-- insert-after: nanovllm/engine/model_runner.py :: self.allocate_kv_cache() -->
```python
        if not self.enforce_eager:
            self.capture_cudagraph()
```

在 `self.block_size = config.kvcache_block_size` 下面记下开关（第 2 章 `example.py` 传的这个参数终于用上了）：

<!-- insert-after: nanovllm/engine/model_runner.py :: self.block_size = config.kvcache_block_size -->
```python
        self.enforce_eager = config.enforce_eager
```

**④ 退出时先释放图。** 图持有 GPU 资源，要在销毁进程组之前释放。替换 `exit`：

<!-- method: nanovllm/engine/model_runner.py :: ModelRunner -->
```python
    def exit(self):
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()
```

> ⚠️ **`graph_bs` 的一个坑**：默认 `max_num_seqs=512` 时一切正常。如果你设成不能被 16 整除的值（比如 100），档位最大到 96，`bs` 为 97～100 时 `next(...)` 找不到档位，会抛 `StopIteration`。详见第 12 章。

## 检查点

**① 比较吞吐量。** 把 `bench.py` 里的 `enforce_eager` 依次改为 `True` 和 `False`：

```bash
python bench.py
```

期望 `enforce_eager=False` 明显更快。项目 README 的参考成绩是：RTX 4070 Laptop 8GB 上约 1434 tok/s。你的数字取决于硬件，只要趋势一致即可。

**② 结果不变。** 打开图模式后，逻辑必须和之前一样。把 `tests/test_consistency.py` 里的 `enforce_eager=True` 改为 `False`，再跑一遍：

```bash
python tests/test_consistency.py
```

如果图模式下结果错误而普通模式正确，重点检查：padding 行的 `slot_mapping` 是否为 -1，以及档位向上取整是否正确。

## 本章小结

单卡性能到头了。**如果模型比一张卡的显存还大呢？** 下一章。
