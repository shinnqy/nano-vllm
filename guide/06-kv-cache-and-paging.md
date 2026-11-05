# 第 6 章：为什么越生成越慢？KV Cache 和分页

**上一章留下的问题**：序列越长，每轮越慢。

**本章要回答**：

```text
为什么慢？                → 每轮把整条序列重算一遍
哪些东西不用重算？        → 历史 token 的 K、V
那每轮只算什么？          → 只算最新的 token：Prefill / Decode 两种模式
K、V 存到哪？             → 物理缓存池 + 分页
谁管理这些页？            → BlockManager
```

这个改动会贯穿所有层，我们由外往里一层层改。

## 本章任务卡

| | |
|---|---|
| **要解决什么** | 每轮把整条序列重算一遍，序列越长越慢 |
| **要实现什么** | Prefill/Decode 两种模式、`BlockManager`（最简版）、`allocate_kv_cache`、`prepare_decode`、`prepare_block_tables`、`store_kvcache`、缓存版 `Attention` |
| **要调用什么** | `flash_attn_with_kvcache`、`triton.jit`、`tl.program_id`、`tl.load`、`tl.store`、`tl.arange`、`model.modules()`、`Tensor.stride` |
| **为什么这样做** | 历史 token 的 K、V 不会变，存起来就不用重算。存储照搬操作系统的分页：不要求连续，不用预留，不产生碎片 |

## 6.1 为什么不用重算

回到注意力公式：

```text
output[i] = softmax(Q[i] · K[0..i] / sqrt(d)) · V[0..i]
```

**关键观察：token `i` 的 K、V 只取决于第 0～i 个 token**，后面的 token 影响不到它。所以一旦算过，它们就永远不变。

现在生成第 `n+1` 个 token 时，要把 `n` 个 token 全部重算，总量是 `1+2+...+N`，也就是 O(N²)。

把 K、V 存下来，一轮就分成两种模式：

| 模式 | 什么时候 | 计算什么 | 计算量 |
|---|---|---|---|
| **Prefill** | 第一轮 | prompt 全部 token，存下 K、V | O(prompt 长度) |
| **Decode** | 之后每轮 | **只算最新的 1 个 token**，K、V 追加进缓存 | 每条序列 1 个 token |

`prepare_prefill` 这个名字现在有含义了：**预先把 prompt 灌进缓存。**

## 6.2 最外层：LLMEngine 要知道模式

一轮要么是 Prefill，要么是 Decode，这个决定由调度器做出，传给执行器。

顺便把进度条改进一下，分别显示两种模式的吞吐，因为它们的量级差很多。`num_tokens` 用正负号表示这一轮的类型：正数是 Prefill 的 token 数，负数是 Decode 的序列数。

替换 `LLMEngine` 的两个方法：

<!-- method: nanovllm/engine/llm_engine.py :: LLMEngine -->
```python
    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
```

在 `from dataclasses import fields` 下面加入：

<!-- insert-after: nanovllm/engine/llm_engine.py :: from dataclasses import fields -->
```python
from time import perf_counter
```

## 6.3 下一个问题：K、V 存哪？

先算算有多大。每个 token 的每一层都要存 K 和 V：

```text
每 token 字节数 = 2 (K+V) × 28 层 × 8 个 KV 头 × 128 维 × 2 字节 = 114,688 字节 ≈ 112 KB
```

1000 个 token 就要 112 MB。显存基本都会被它吃掉，所以**怎么管理 KV Cache 是推理引擎的核心问题**。

最直观的做法是每个请求预留 `max_len` 那么长的连续显存，但是：

- 按 4096 长度预留，一条请求就要 448 MiB，而大部分请求用不了这么多，**浪费严重**；
- 请求长短不一，容易产生碎片。

操作系统几十年前就解决了同样的问题：虚拟内存分页。KV Cache 照搬，这就是 **Paged Attention**：

```text
物理缓存池：切成等大的块（默认每块 256 个 token）
┌──────┬──────┬──────┬──────┬──────┬──────┬──────┬──────┐
│ 块 0 │ 块 1 │ 块 2 │ 块 3 │ 块 4 │ 块 5 │ 块 6 │ 块 7 │
└──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────┘

请求 A 的块表 block_table = [7, 2, 5]   （可以不连续！）
  逻辑位置 0~255   → 块 7
  逻辑位置 256~511 → 块 2
  逻辑位置 512~767 → 块 5
```

**整个项目最重要的公式**，逻辑位置 `t` 到物理槽位（`B` 是每块 token 数）：

```text
slot(t) = block_table[t // B] × B + t % B
```

分成两部分来实现：

| 部分 | 负责 | 在哪 |
|---|---|---|
| 账本：哪个块空闲、谁占用了 | 纯 CPU，只有整数 | `BlockManager` |
| 真正的数据 | GPU 显存 | `kv_cache` 张量 |

块表就是两者之间的桥梁。**一定要分清账本和缓存是两回事。**

在 `Config` 里增加配置，`eos: int = -1` 下面加入：

<!-- insert-after: nanovllm/config.py :: eos: int = -1 -->
```python
    kvcache_block_size: int = 256     # 每块多少 token
    num_kvcache_blocks: int = -1      # 块数，由 ModelRunner 写入
```

`__post_init__` 里加一条校验，放在 `assert os.path.isdir(self.model)` 后面。这是 FlashAttention 分页模式的硬性要求：页大小必须是 256 的倍数。

<!-- insert-after: nanovllm/config.py :: assert os.path.isdir(self.model) -->
```python
        assert self.kvcache_block_size % 256 == 0
```

## 6.4 Sequence 要多记什么

一条序列现在有三个长度。这是整个项目最容易搞混的地方：

```text
token_ids:  [t0 t1 t2 ... ........................... tn-1]
             ├── num_cached_tokens ──┤
             已有 KV，可以直接读
                                     ├ num_scheduled_tokens ┤
                                     本轮正要计算 KV

             ├────────────── num_tokens ──────────────────┤
             序列的总长度（包括已经采样出来的 token）
```

**为什么 `num_cached_tokens` 可以小于 `num_tokens`**：刚采样出的新 token 已追加到序列，但它的 K/V 要**下一轮**才能算出来。这就是 Decode 循环的本质：

```text
第 k 轮开始：  num_tokens = n，  num_cached_tokens = n-1
               计算第 n-1 个 token 的 KV，得到 logits，采样出第 n 个 token
第 k 轮结束：  num_tokens = n+1，num_cached_tokens = n
```

在 `Sequence.__init__` 里，`self.num_prompt_tokens = len(token_ids)` 下面加入：

<!-- insert-after: nanovllm/engine/sequence.py :: self.num_prompt_tokens = len(token_ids) -->
```python
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.block_table = []      # 逻辑块号 → 物理块号
```

再加一个类属性记录块大小。把 `counter = count()` 这一行替换成：

<!-- replace-line: nanovllm/engine/sequence.py :: counter = count() -->
```python
    block_size = 256       # 类属性：所有序列相同，由引擎启动时设置
    counter = count()      # 类属性：全局递增，生成唯一 seq_id
```

增加两个块计算方法：

<!-- method: nanovllm/engine/sequence.py :: Sequence -->
```python
    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size      # 向上取整

    @property
    def last_block_num_tokens(self):
        # 范围是 1~B，不是 0~B-1
        return self.num_tokens - (self.num_blocks - 1) * self.block_size
```

在 `LLMEngine.__init__` 里设置块大小，放在 `config = Config(...)` 下面：

<!-- insert-after: nanovllm/engine/llm_engine.py :: config = Config(model, **config_kwargs) -->
```python
        Sequence.block_size = config.kvcache_block_size
```

## 6.5 再往里：Scheduler 决定模式

逻辑很简单：有新请求就先 Prefill；没有才 Decode。

Prefill 要分配块。Decode 时序列变长了，可能需要新块。

**为什么同一轮不混合？** 它们调用的 FlashAttention 接口不同，混合起来实现复杂得多。

替换 `Scheduler` 的三个方法：

<!-- method: nanovllm/engine/scheduler.py :: Scheduler -->
```python
    def __init__(self, config: Config):
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []

        # prefill
        while self.waiting:
            seq = self.waiting.popleft()
            self.block_manager.allocate(seq)
            seq.num_scheduled_tokens = seq.num_tokens
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        for seq in self.running:
            seq.num_scheduled_tokens = 1
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)
        return scheduled_seqs, False

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
```

导入新模块：

<!-- insert-after: nanovllm/engine/scheduler.py :: from nanovllm.engine.sequence import Sequence, SequenceStatus -->
```python
from nanovllm.engine.block_manager import BlockManager
```

**逐行看新的 Scheduler：**

- `schedule` 返回 `(序列列表, 是否 Prefill)`：模式由调度器决定，由引擎转告执行器。
- Prefill 分支把 `num_scheduled_tokens` 设为整条序列长度；Decode 分支设为 1。这个字段是调度器和执行器之间的约定：**本轮要计算多少个 token**。
- `if scheduled_seqs: return ..., True`：只要有新请求就先 Prefill，所以同一轮不会混合。
- `postprocess`：先更新 `num_cached_tokens`，再追加新 token。顺序不能错，因为新 token 的 KV 还没有计算。

## 6.6 BlockManager：账本的最简版本

根据 `Scheduler` 的用法，它要提供 `allocate`、`deallocate`、`may_append`。

**`may_append` 的条件 `len(seq) % B == 1` 为什么是 1？**

Decode 这一轮要计算**最后一个 token** 的 KV。序列长度 `n` 模 B 等于 1，说明最后这个 token（下标 `n-1`，`(n-1) % B == 0`）正好是某个新块的第一个，这个块还不存在，要先分配。

用 B=256 走一遍：

```text
序列长度 256（已占满 1 块），上一轮采样后变成 257
本轮要写下标 256 的 KV → 257 % 256 == 1 → 分配新块 ✓
```

文件：`nanovllm/engine/block_manager.py`（第 8、9 章继续加强）

<!-- file: nanovllm/engine/block_manager.py -->
```python
from collections import deque

from nanovllm.engine.sequence import Sequence


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        # deque：头部弹出 O(1)
        self.free_block_ids: deque[int] = deque(range(num_blocks))

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        for _ in range(seq.num_blocks):
            seq.block_table.append(self.free_block_ids.popleft())
        seq.num_cached_tokens = 0

    def deallocate(self, seq: Sequence):
        for block_id in seq.block_table:
            self.free_block_ids.append(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self.free_block_ids.popleft())
```

**逐行看 BlockManager：**

- `deque(range(num_blocks))`：一开始所有块都空闲。
- `allocate` 给整条序列一次分配够：序列马上要写入这些位置。
- `deallocate` 把块还回去，并清空 `block_table` 和 `num_cached_tokens`，下次分配时状态才是干净的。
- `may_append` 在 Decode 时调用：序列刚跨过块边界，才需要新块。

账本可以不开 GPU 单独验证：

文件：`tests/test_paging.py`

<!-- file: tests/test_paging.py -->
```python
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams

sp = SamplingParams()
# 序列的块计算
s = Sequence(list(range(257)), sp)
assert s.num_blocks == 2 and s.last_block_num_tokens == 1
s = Sequence(list(range(256)), sp)
assert s.num_blocks == 1 and s.last_block_num_tokens == 256

bm = BlockManager(4, 256)
s = Sequence(list(range(256)), sp)
bm.allocate(s)
assert s.block_table == [0]
s.append_token(1)                     # 长度 257：新 token 落在新块
bm.may_append(s)
assert s.block_table == [0, 1]
s.append_token(2)                     # 长度 258：还在同一块，不用新增
bm.may_append(s)
assert s.block_table == [0, 1]
bm.deallocate(s)
assert len(bm.free_block_ids) == 4 and s.block_table == []
print("OK")
```

```bash
python tests/test_paging.py
```

## 6.7 再往里：ModelRunner 准备两种输入

账本决定了块表，现在 `ModelRunner` 要把它变成张量。模型深处要知道这些新信息，放进 `Context`：

| 字段 | 含义 | 谁用 |
|---|---|---|
| `is_prefill` | 这一轮的模式 | Attention、输出头 |
| `slot_mapping` | 本轮每个新 token 的 KV 写到哪个槽位 | Attention |
| `context_lens` | Decode 时每条序列的总长度 | Attention |
| `block_tables` | 每条序列的块表 | Attention |

文件：`nanovllm/utils/context.py`（替换）

<!-- file: nanovllm/utils/context.py -->
```python
from dataclasses import dataclass
import torch


@dataclass(slots=True)
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None  # 本轮 Query 的累积长度
    cu_seqlens_k: torch.Tensor | None = None  # 本轮 Key 的累积长度
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None  # 新 KV 写到哪些物理槽位
    context_lens: torch.Tensor | None = None  # Decode 时每条序列的总长度
    block_tables: torch.Tensor | None = None  # 每条序列的块表

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None):
    global _CONTEXT
    # 每次新建对象，确保不会残留上一轮的字段
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
```

**Prefill 的 slot 怎么算？** 按块算比逐 token 算快。以 `block_table=[7, 2, 9]`、计算 0～599 为例：

```text
位置   0~255 → 块 7 → slot 1792~2047
位置 256~511 → 块 2 → slot  512~767
位置 512~599 → 块 9 → slot 2304~2391
```

只有首尾两块可能不完整：首块的起点可能在块中间，尾块的终点也可能在块中间。

**`positions` 和 `slot_mapping` 要分清**：`positions` 是逻辑位置，用于 RoPE；`slot_mapping` 是物理位置，决定 KV 放在哪。

**为什么现在就写 `start = seq.num_cached_tokens`？** 现在它总是 0。第 8、9 章它会不为 0，提前写成通用形式，那时就不用改了。

替换这些方法（`prepare_decode`、`prepare_block_tables`、`allocate_kv_cache` 是新增的）：

<!-- method: nanovllm/engine/model_runner.py :: ModelRunner -->
```python
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        # 暂时写死。下一章解决“到底该分多少块”
        config.num_kvcache_blocks = 64
        # 多卡时每张卡只存一部分 KV 头（第 11 章）
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # 形状：[K或V, 层, 块, 每块 token 数, KV 头, 维度]
        # 这时默认类型是 bf16、默认设备是 cuda，所以不用另外指定
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        # modules() 按定义顺序遍历。用 hasattr 找到每层的 Attention，不用知道模型结构
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]   # 切片共享内存，不会多占显存
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        # 张量必须是矩形。-1 只用于补齐，读取时 FlashAttention 不会越过序列长度
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None                       # 第 8 章才会用到
        for seq in seqs:
            start = seq.num_cached_tokens         # 从第一个没有 KV 的位置开始
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end                        # 能看到的 Key：从 0 到 end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))   # 逻辑位置，与物理块无关
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size   # 不含 end_block
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size     # 首块：从块中间开始
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    # 尾块：end - i×B 是 end 在这一块内的偏移
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)           # 只需要最后一个 token
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))              # 包含它自己，因为它的 KV 已先写入
            # 最后一块的块号 × B + 最后一块内的偏移
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        return self.model.compute_logits(self.model(input_ids, positions))

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs)
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist()
        reset_context()
        return token_ids
```

### 逐行看 allocate_kv_cache

`torch.empty(2, 28, 块数, 256, 8, 128)` 这六个维度分别是：K 或 V、第几层、第几块、块内第几个 token、第几个头、第几维。

`self.kv_cache[0, layer_id]` 是视图，形状 `[块数, 256, 8, 128]`。每层的 `Attention` 只拿到属于自己的那一片，不需要知道其他层的存在。

用 `hasattr(module, "k_cache")` 找到 Attention，这样 `ModelRunner` 不需要知道模型内部的结构。

### 走一遍 prepare_prefill：slot 怎么算

取 `block_table=[7, 2, 9]`，`B=256`：

| 情况 | `start` | `end` | `start_block` | `end_block` | 写入的 slot 区间 |
|---|---|---|---|---|---|
| 从头算 600 个 | 0 | 600 | 0 | 3 | `1792~2047`、`512~767`、`2304~2391` |
| 从 300 开始算（第 8 章） | 300 | 600 | 1 | 3 | `556~767`、`2304~2391` |
| 从 512 开始算（第 9 章） | 512 | 600 | 2 | 3 | `2304~2391` |

手算第二行：

```text
i=1（首块）：slot_start = 2×256 + 300%256 = 512 + 44 = 556
             i 不是尾块 → slot_end = 2×256 + 256 = 768
i=2（尾块）：slot_start = 9×256 = 2304
             slot_end = 9×256 + 600 - 2×256 = 2304 + 88 = 2392
```

`end_block` 里的 `+ B - 1` 是向上取整：`end=600` 时要用到第 0、1、2 块，所以 `end_block=3`，`range` 不包含 3。

**顺序为什么重要？** `slot_mapping[i]` 和 `input_ids[i]` 一一对应，kernel 靠下标把它们配对。

### 走一遍 prepare_decode

序列长度 601，`block_table=[7, 2, 9]`，本轮要计算下标 600 的 KV：

```text
input_ids     = 最后一个 token
positions     = 600
context_lens  = 601      读 KV 时包括它自己
num_blocks    = (601+255)//256 = 3
last_block_num_tokens = 601 - 2×256 = 89
slot_mapping  = 9×256 + 89 - 1 = 2392
```

用总公式核对：`slot(600) = table[600//256]×256 + 600%256 = 9×256 + 88 = 2392`，结果相同。

每条序列只有 1 个 token，所以不需要 `cu_seqlens`：批大小就是序列条数。块表用 `-1` 补齐，是因为张量必须是矩形，而各序列的块数不同。

`__init__` 里记下块大小，放在 `self.config = config` 下面：

<!-- insert-after: nanovllm/engine/model_runner.py :: self.config = config -->
```python
        self.block_size = config.kvcache_block_size
```

模型加载后分配缓存，放在 `self.sampler = Sampler()` 下面：

<!-- insert-after: nanovllm/engine/model_runner.py :: self.sampler = Sampler() -->
```python
        self.allocate_kv_cache()
```

**创建顺序的新约束：** `num_kvcache_blocks` 由 `ModelRunner` 写入、`Scheduler` 读取。第 2 章写的 `LLMEngine.__init__` 恰好是先 `ModelRunner` 后 `Scheduler`，顺序调换就会得到空账本。

## 6.8 再往里：输出头

Decode 时每条序列只有 1 个 token，不需要再取最后一个位置。

替换 `ParallelLMHead` 的 `forward`：

<!-- method: nanovllm/layers/embed_head.py :: ParallelLMHead -->
```python
    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        return F.linear(x, self.weight)
```

## 6.9 最里层：Attention 读写缓存

每一轮的流程：

```text
① 先把本轮新的 K、V 写进缓存
② Prefill：flash_attn_varlen_func，用本轮算出的 k、v
   Decode：flash_attn_with_kvcache，从缓存读（每条序列只有 1 个 Query）
```

**为什么先写后读？** 第 `i` 个 Query 也要看**自己**的 K/V（公式里是 `0..i`，含 `i`）。先写，当前 token 就和历史一样都在缓存里，读的逻辑不需要特殊处理。这也是 `context_lens` 包含当前 token 的原因。

### 写：为什么用 Triton

**要做什么**：把这一轮的 `key[i]`、`value[i]` 写到物理槽位 `slot_mapping[i]`。

由于 `k_cache` 形状是 `[块, B, Hkv, d]`，内存连续，每个 token 占 `D = Hkv × d` 个元素，所以：

```text
slot 对应的第一个元素的下标 = slot × D
```

用 PyTorch 高级索引也能写：`k_cache.view(-1, Hkv, d)[slot_mapping] = key`。但这个操作每层每轮都跑，第 10 章固定大小的图里还要跳过 padding 行（`-1` 在高级索引里表示最后一个，会写错）。项目用 Triton 写了一个小 kernel。

**Triton 是什么**：用 Python 语法写，编译成 GPU 程序。核心概念是 **program**：

```python
kernel[(N,)](参数...)   # 启动 N 个 program 实例，在 GPU 上并行执行
```

每个实例用 `tl.program_id(0)` 知道自己的编号，这里每个实例负责一个 token。指针算术和 C 语言类似。

**为什么要传 `key_stride`**：`qkv.split` 得到的 K、V 与大张量共享内存，相邻行之间不一定紧贴。`v` 没有经过 RoPE，依然是切片，`v[i]` 的起点是 `i × stride(0)`，`stride(0)` 是整个 QKV 的宽度而不是 D。写死 `i × D` 就会读错。

### 读：flash_attn_with_kvcache

它接收整个缓存池、每条序列的长度和块表，自己按第 6.3 节的公式去找数据。输入形状要 `[batch, 序列长度, 头, 维度]`，Decode 每条序列只有 1 个 token，所以 `unsqueeze(1)`。

输出是 `[B, 1, H, d]`。`Qwen3Attention` 里的 `o.flatten(1, -1)` 把它和 Prefill 的 `[N, H, d]` 都变成 `[N, H×d]`，所以上层不用分情况。

文件：`nanovllm/layers/attention.py`（替换第 5 章的版本）

<!-- file: nanovllm/layers/attention.py -->
```python
import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


# triton.jit：首次调用时为当前 GPU 编译
@triton.jit
def store_kvcache_kernel(
    key_ptr,            # 传入张量，在 kernel 里就是指向首元素的指针
    key_stride,         # 普通数字
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,    # constexpr：编译期常量。tl.arange 的长度必须在编译时确定
):
    idx = tl.program_id(0)                     # 我负责输入中的第几个 token
    slot = tl.load(slot_mapping_ptr + idx)     # 它要写到第几个物理槽位
    if slot == -1: return                      # -1：padding 行，不用写（第 10 章用到）
    # 读取这个 token 的全部 D 个元素。用真实的 stride，不是 D！
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    # 写入位置 = slot × D
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape         # N：本轮新 token 总数
    D = num_heads * head_dim                   # 一个 token 的总元素数
    # 这些断言保护 kernel 的前提：同一个 token 内部的 D 个元素必须连续
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    # [(N,)]：启动 N 个 program。CPU 提交的是一次 kernel，不是 N 次
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


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
        # 先放空张量。真正的缓存由 ModelRunner 创建后挂上来
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        # 缓存还没分配时跳过写入。测试的无缓存基准会用到，下一章的预热也要用
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q,
                                       cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k,
                                       cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale,
                                       causal=True,
                                       block_table=context.block_tables)   # 现在总是 None，第 8 章才用
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1),
                                        k_cache, v_cache,
                                        cache_seqlens=context.context_lens,  # 每条序列读多长
                                        block_table=context.block_tables,    # 去哪里读
                                        softmax_scale=self.scale,
                                        causal=True)
        return o
```

### 逐行看 kernel

假设 `key` 形状 `[2, 8, 128]`，于是 `D = 1024`。启动 2 个 program：

| 代码 | 第 1 号 program 的结果 |
|---|---|
| `idx = tl.program_id(0)` | 1 |
| `slot = tl.load(slot_mapping_ptr + idx)` | `slot_mapping[1]` |
| `if slot == -1: return` | padding 行直接跳过 |
| `key_offsets = idx * key_stride + tl.arange(0, D)` | 第 1 行 1024 个元素的下标 |
| `tl.load(key_ptr + key_offsets)` | 读出一个长度 1024 的向量 |
| `cache_offsets = slot * D + tl.arange(0, D)` | 缓存里目标位置的 1024 个下标 |
| `tl.store(k_cache_ptr + cache_offsets, key)` | 写入 |

**为什么可以用 `slot * D`？** `k_cache` 形状是 `[块数, 256, 8, 128]`，内存连续。把前两维看成一维“槽位”，每个槽位正好 `8 × 128 = D` 个元素。`store_kvcache` 里的 `k_cache.stride(1) == D` 就在检查这个前提。

**`store_kvcache` 里的三条断言**分别保证：同一个头内元素相邻（`stride(-1)==1`）；同一个 token 的各头相邻（`stride(1)==head_dim`）；缓存的槽位相邻。kernel 假设同一 token 的 D 个元素连续，违反时断言会立刻报错，而不是悄悄写错。

### 逐行看 Attention.forward

| 代码 | 为什么 |
|---|---|
| `self.k_cache = self.v_cache = torch.tensor([])` | 普通属性，不是 Parameter，不会被当成权重。先放空张量，避免 `hasattr` 和 `numel()` 出错 |
| `if k_cache.numel() and v_cache.numel()` | 缓存分配之前的路径。第 7 章预热、测试的无缓存基准都会走这里 |
| Prefill 的 `flash_attn_varlen_func(q, k, v, ...)` | 所有 K、V 都是刚算的，直接用，比从缓存读更简单 |
| Decode 的 `q.unsqueeze(1)` | `[B, 16, 128] → [B, 1, 16, 128]`，这个算子要求有序列维 |
| `cache_seqlens=context.context_lens` | 每条序列的历史有多长。它决定读取范围，也决定 padding 的 `-1` 不会被读到 |
| `block_table=context.block_tables` | 算子用第 6.3 节的公式，把逻辑位置换算成物理位置 |

### 把一轮连起来

一条 600 token 的请求生成第一个新 token，再生成第二个：

```text
第 1 轮 Prefill
  schedule        分配 3 块，num_scheduled_tokens=600
  prepare_prefill input_ids=全部 600 个，slot_mapping=600 个
  每层 Attention  写 600 个 KV → 用本轮 k、v 计算
  输出头          只取最后一个位置 → logits [1, V]
  postprocess     num_cached_tokens=600，追加新 token，num_tokens=601

第 2 轮 Decode
  schedule        601%256≠1，不用新块，num_scheduled_tokens=1
  prepare_decode  input_ids=[新 token]，position=600，context_len=601，slot=2392
  每层 Attention  写 1 个 KV → 用 Q 读缓存里的 601 个 KV
  输出头          一条序列一行，不用再挑位置
  postprocess     num_cached_tokens=601，追加新 token，num_tokens=602
```

> `tests/test_batching.py` 用的还是 `run(seqs)` 的旧签名，从现在起不能再运行了。

## 检查点

**从入口跑一遍：**

```bash
python example.py
```

**看症状有没有解决：**

```bash
python tests/time_steps.py
```

每一轮的耗时现在应该**基本不随序列变长而增加**。第 0 轮（Prefill）会比后面的轮次慢。

**精确验证：有缓存和没缓存的结果必须一致。** 这是 KV Cache 正确性的黄金标准。下面手工管理缓存，块表故意不连续，每一步和“不用缓存的完整前向”对比。

第一个文件是工具函数，第 8 章还会用到：

文件：`tests/kv_helpers.py`

<!-- file: tests/kv_helpers.py -->
```python
import os
import torch
import torch.distributed as dist
from transformers import AutoConfig

from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.context import set_context, reset_context
from nanovllm.utils.loader import load_model

path = os.path.expanduser(os.environ.get("MODEL_PATH", "~/huggingface/Qwen3-0.6B/"))
cfg = AutoConfig.from_pretrained(path)
B = 256  # 每块 token 数

dist.init_process_group("nccl", "tcp://localhost:2333", world_size=1, rank=0)
torch.set_default_dtype(cfg.dtype)
torch.set_default_device("cuda")
model = Qwen3ForCausalLM(cfg)
load_model(model, path)
# 手工建 10 个块的物理缓存
num_kv_heads = cfg.num_key_value_heads
head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
kv_cache = torch.empty(2, cfg.num_hidden_layers, 10, B, num_kv_heads, head_dim)
empty = torch.tensor([])
torch.set_default_device("cpu")
torch.set_default_dtype(torch.float32)
attns = [m for m in model.modules() if hasattr(m, "k_cache")]


def use_cache(enabled):
    for i, m in enumerate(attns):
        m.k_cache = kv_cache[0, i] if enabled else empty
        m.v_cache = kv_cache[1, i] if enabled else empty


def to_gpu(x, dtype):
    return torch.tensor(x, dtype=dtype, device="cuda")


@torch.inference_mode()
def no_cache_logits(tokens):
    # 基准：每次把整条序列重算一遍，完全不用缓存
    use_cache(False)
    n = len(tokens)
    cu = to_gpu([0, n], torch.int32)
    set_context(True, cu, cu, n, n)
    logits = model.compute_logits(model(to_gpu(tokens, torch.int64), to_gpu(list(range(n)), torch.int64)))
    reset_context()
    use_cache(True)
    return logits[0].float()


def slots(table, start, end):
    # 最重要的公式：slot(t) = table[t // B] × B + t % B
    return [table[t // B] * B + t % B for t in range(start, end)]


@torch.inference_mode()
def prefill(jobs, use_block_table=False):
    # jobs：[(tokens, table, start, end), ...]
    input_ids, positions, slot_mapping = [], [], []
    cu_q, cu_k = [0], [0]
    for tokens, table, start, end in jobs:
        input_ids += tokens[start:end]
        positions += range(start, end)
        slot_mapping += slots(table, start, end)
        cu_q.append(cu_q[-1] + end - start)
        cu_k.append(cu_k[-1] + end)
    block_tables = None
    if use_block_table:
        max_blocks = max(len(j[1]) for j in jobs)
        block_tables = to_gpu([j[1] + [-1] * (max_blocks - len(j[1])) for j in jobs], torch.int32)
    set_context(
        True, to_gpu(cu_q, torch.int32), to_gpu(cu_k, torch.int32),
        max(j[3] - j[2] for j in jobs), max(j[3] for j in jobs),
        to_gpu(slot_mapping, torch.int32), None, block_tables,
    )
    logits = model.compute_logits(model(to_gpu(input_ids, torch.int64), to_gpu(positions, torch.int64)))
    reset_context()
    return logits.float()


@torch.inference_mode()
def decode(seqs):
    # seqs：[(tokens, table), ...]，tokens 里已包含这一轮要计算的新 token
    ids = [s[0][-1] for s in seqs]
    pos = [len(s[0]) - 1 for s in seqs]
    slot_mapping = [slots(s[1], len(s[0]) - 1, len(s[0]))[0] for s in seqs]
    lens = [len(s[0]) for s in seqs]
    max_blocks = max(len(s[1]) for s in seqs)
    tables = [s[1] + [-1] * (max_blocks - len(s[1])) for s in seqs]
    set_context(
        False, slot_mapping=to_gpu(slot_mapping, torch.int32),
        context_lens=to_gpu(lens, torch.int32), block_tables=to_gpu(tables, torch.int32),
    )
    logits = model.compute_logits(model(to_gpu(ids, torch.int64), to_gpu(pos, torch.int64)))
    reset_context()
    return logits.float()


def check(name, got, tokens):
    expected = no_cache_logits(tokens)
    diff = (got - expected).abs().max().item()
    same = got.argmax().item() == expected.argmax().item()
    print(f"{name:32s} 最大差异 {diff:.4f}  argmax 相同: {same}")
    assert diff < 1.0 and same, name


def random_tokens(n):
    return torch.randint(0, min(cfg.vocab_size, 10000), (n,)).tolist()
```

文件：`tests/test_kv_cache.py`

<!-- file: tests/test_kv_cache.py -->
```python
import torch
from kv_helpers import *

torch.manual_seed(0)
tokens_a, tokens_b = random_tokens(600), random_tokens(50)
# 块表故意不连续、不按顺序，证明物理位置不影响结果
table_a, table_b = [7, 2, 9], [3]
use_cache(True)

# Prefill：两条序列一起
out = prefill([(tokens_a, table_a, 0, 600), (tokens_b, table_b, 0, 50)])
check("Prefill（序列 A）", out[0], tokens_a)
check("Prefill（序列 B）", out[1], tokens_b)

# Decode：每条序列加 1 个 token
for step in range(5):
    tokens_a = tokens_a + [(step * 7 + 11) % 1000]
    tokens_b = tokens_b + [(step * 13 + 5) % 1000]
    out = decode([(tokens_a, table_a), (tokens_b, table_b)])
    check(f"Decode 第 {step} 步（序列 A）", out[0], tokens_a)
    check(f"Decode 第 {step} 步（序列 B）", out[1], tokens_b)
print("OK")
```

```bash
python tests/test_kv_cache.py
```

**如果不一致：**

| 现象 | 可能原因 |
|---|---|
| Prefill 就错 | `cu_seqlens` 或 `max_seqlen` 错 |
| Decode 错 | `context_lens` 必须**包含**新 token；`positions` 要等于 `len-1` |
| 随机错、无规律 | kernel 里写死了 `i × D`，没用真实 stride |

## 本章小结

越生成越慢的症状解决了。留下了一个非常显眼的问题：

```python
config.num_kvcache_blocks = 64   # 暂时写死
```

下一章解决它。
