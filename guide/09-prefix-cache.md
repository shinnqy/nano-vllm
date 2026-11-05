# 第 9 章：相同的开头能复用吗？前缀缓存

**上一章留下的问题**：多个请求开头相同（比如同一段系统提示），却各算一遍。

**本章要回答**：

```text
能不能复用？              → K、V 只依赖前面的 token，前缀相同就完全相同
复用的粒度是什么？        → 块（分页的第二个好处：不同请求可以指向同一个物理块）
怎么判断“算过了”？        → 哈希表，而且必须是链式哈希
一个块被多个请求共用？    → 引用计数
请求结束后还想留着？      → 空闲但保留内容，直到被挪用
```

分页这时显出了它的第二个价值：两个请求的块表可以指向同一个物理块。

```text
请求 A 的块表：[7, 2, 5]
请求 B 的块表：[7, 2, 9]     前两块共享，只有最后一块不同
```

**要改的地方比想象的少：**

```text
Scheduler      加一行：算完后登记哈希
ModelRunner    不用改    num_cached_tokens 不为 0 的路径，第 8 章分块时已经走通了
Attention      不用改    读历史 KV 的路径，第 8 章也走通了
BlockManager   大改      所有新逻辑都在账本里
```

## 本章任务卡

| | |
|---|---|
| **要解决什么** | 相同的开头被反复计算 |
| **要实现什么** | `Sequence.block`、`Block`、`compute_hash`、`hash_blocks`，以及支持命中、引用计数和复活的 `BlockManager` 最终版 |
| **要调用什么** | `xxhash.xxh64`、`int.to_bytes`、`np.array(...).tobytes()`、`deque.remove` |
| **为什么这样做** | 上层的路径第 8 章已经走通了，所以这章几乎只改账本。这正是把账本和 GPU 缓存分开的好处 |

## 9.1 为什么是链式哈希

以块为单位，第 `i` 块的哈希必须包含**它之前全部内容**：

```text
h0 = hash(块 0 的 token)
h1 = hash(h0 + 块 1 的 token)
h2 = hash(h1 + 块 2 的 token)
```

**为什么不能只哈希本块内容？** 因为 K/V 依赖全部历史。两个序列第 1 块的 token 相同，但第 0 块不同，它们第 1 块的 KV 就不同，绝对不能共用。链式哈希把“前缀完全相同”压缩成一个数。

要用到 `xxhash`：极快的非加密哈希。这里不防攻击，只求快。

`Sequence` 要能取出第 `i` 块的 token，新增这个方法：

<!-- method: nanovllm/engine/sequence.py :: Sequence -->
```python
    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]
```

## 9.2 什么时候登记哈希：从 Scheduler 开始

**不能在分配时登记。** 那时 KV 还没算出来，别的请求这时命中，读到的就是垃圾数据。必须等模型**算完之后**。

所以 `postprocess` 开头要加一步。在 `for seq, token_id in zip(seqs, token_ids):` 下面加入：

<!-- insert-after: nanovllm/engine/scheduler.py :: for seq, token_id in zip(seqs, token_ids): -->
```python
            # 必须在更新 num_cached_tokens 之前，它要用旧值
            self.block_manager.hash_blocks(seq)
```

**只登记写满的块。** 没写满的还在变化，不能复用。

```python
start = seq.num_cached_tokens // B
end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // B
```

整除天然只算“本轮新填满的块”。

## 9.3 往里：块的生命周期

一个物理块现在有三种状态：

```text
              allocate/命中                     deallocate，引用归零
  ┌────────────────┐   ──────────→   ┌───────┐  ──────────→  ┌────────────────┐
  │ 空闲，无缓存    │                 │ 使用中 │               │ 空闲，保留缓存  │
  └────────────────┘                 └───────┘               └────────────────┘
          ▲                              ▲       命中，复活          │
          │                              └───────────────────────────┤
          └───────── 被挪用：删除哈希映射，reset ─────────────────────┘
```

“空闲但保留缓存”是关键：**它既算空闲，可以被挪用；又可以被命中，重新复活。** 这实现了一个简单的缓存淘汰策略：

- 空闲队列是 FIFO，先释放的先被挪用，近似 LRU；
- `deallocate` 用 `reversed`，**先释放尾部的块**，所以尾部先被淘汰。前缀只能从头开始匹配，头部块还在，尾部没了也能部分命中；反过来，头部没了，尾部就废了。

## 9.4 can_allocate：两个小陷阱

**为什么 `range(seq.num_blocks - 1)`，最后一块不参与匹配？**

1. 最后一块往往没写满，以后可能要追加 token。别人的块不能写。
2. 就算 prompt 完全命中，也**至少要算一个 token**，才能得到 logits 去采样。

**为什么命中“使用中”的块才减 `num_new_blocks`，命中“空闲保留”的块却不减？** 因为后者还在空闲队列里，复活时会占掉一个空闲名额，空闲块数要算上它，否则会误判有足够的块。

现在回头看第 8 章的调度代码：

```python
num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
```

命中部分不用算，也不占预算。这一行现在才真正起作用。

文件：`nanovllm/engine/block_manager.py`（替换为最终版）

<!-- file: nanovllm/engine/block_manager.py -->
```python
from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1        # -1 表示没有可复用的内容
        self.token_ids = []   # 留一份，哈希碰撞时用来核对

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1    # 只有分配时才调用，所以直接设为 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # set：判断“是否使用中”O(1)
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))   # 把前一块的哈希串进来：链式哈希
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int:
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # 这个块被挪用，旧内容马上要被覆盖，它不能再被命中。
        # 要检查映射确实指向自己：两个相同的 prompt 同时计算时，
        # 同一个哈希会被后者覆盖，先前的块虽然有这个 hash，映射已不属于它。
        # 如果不检查，就会把别人的有效映射删掉。
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id:
            del self.hash_to_block_id[block.hash]
        block.reset()
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int):
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        # 只回到空闲队列，hash 和 GPU 内容都保留，留给后来者
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int:
        # 返回可命中的块数；放不下时返回 -1
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            # 核对 token_ids：防止极小概率的哈希碰撞
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break     # 前缀只能连续匹配，一旦断开，后面的都不算
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        h = -1
        # 先处理命中的块。必须放在前面：否则分配新块时，可能把想命中的块挪用掉
        for i in range(num_cached_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                # 复活：从空闲队列的中间取出。O(n)，但块数不大，而且省下了一次重算
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks):
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence):
        # 先释放尾部的块，尾部先被淘汰，头部保留更久
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence):
        start = seq.num_cached_tokens // self.block_size
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        # 从上一块的哈希继续，无需从头算起
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
```

## 逐行看 BlockManager 的最终版

**`compute_hash`：**

| 代码 | 为什么 |
|---|---|
| `xxhash.xxh64()` | 新建一个 64 位哈希器 |
| `prefix != -1` | 第 0 块没有前缀。`-1` 是“没有”的标记，真实哈希值不会是负数 |
| `prefix.to_bytes(8, "little")` | 哈希器只接受字节。64 位整数正好 8 字节 |
| `np.array(token_ids).tobytes()` | 把整数列表变成连续字节，比逐个转换快得多 |
| `h.intdigest()` | 输出一个整数，当字典的键 |

**`can_allocate` 为什么只查询不修改？** `schedule` 可能在调用它之后才决定 `break`。这时账本必须没有任何变化。

**`hash_blocks` 为什么用旧的 `num_cached_tokens`？** 用整除算出“本轮新填满了哪些块”：

| 情况 | 已缓存 | 本轮 | `start` | `end` | 登记 |
|---|---|---|---|---|---|
| Prefill 600 个 | 0 | 600 | 0 | 2 | 第 0、1 块（第 2 块没写满） |
| 前缀命中 512，算 88 个 | 512 | 88 | 2 | 2 | 无 |
| Decode 写满第 2 块 | 767 | 1 | 2 | 3 | 第 2 块 |
| Decode 没跨边界 | 700 | 1 | 2 | 2 | 无 |

如果先更新 `num_cached_tokens`，`start` 就错了。

**`h = self.blocks[seq.block_table[start - 1]].hash`**：链式哈希的前一环已经存在上一块上，不用从头算。

## 走一遍：两个请求共享前缀

6 个块。请求 A 有 600 个 token。请求 B 的前 512 个和 A 相同，后面 88 个不同。下面是真实账本的运行结果：

| 时刻 | 块表 | 空闲队列（左边先被挪用） | 引用计数不为 0 的块 |
|---|---|---|---|
| A 分配 | A `[0,1,2]` | `3,4,5` | 0、1、2 |
| A 结束 | | `3,4,5,2,1,0` | 无 |
| B 分配 | B `[0,1,3]` | `4,5,2` | 0、1、3 |
| B 结束 | | `4,5,2,3,1,0` | 无 |

逐步看：

1. **A 结束时**，`deallocate` 用 `reversed`，按 2、1、0 的顺序入队。块 0、1 已经登记了哈希，内容还在显存里。块 2 没写满，没有哈希。
2. **B 的 `can_allocate`**：第 0、1 块命中（第 2 块是最后一块，不匹配），返回 2。这两块在空闲队列里，所以 `num_new_blocks` 不减，还是 3。空闲 6 ≥ 3，放得下。
3. **B 的 `allocate`**：把 0、1 从队列中间取出来（复活）。第 2 个逻辑块从队首取到块 3。
4. **调度器**：`num_tokens = 600 - 2×256 = 88`，这轮只算 88 个 token，A 算过的 512 个完全不用算。
5. **`prepare_prefill`**：`start=512`，`seqlen_q=88`，`seqlen_k=600`。K 总数大于 Q 总数，于是传块表。`Attention` 从缓存读 K、V，其中 512 个是 A 留下的。

这就是第 8 章的路径，这里一行都不用改。

## 检查点

### 9-A：账本（不用 GPU）

每个断言都对应一条设计决策。如果某条失败，回头看对应的讲解，比直接改代码更有收获。

文件：`tests/test_block_manager.py`

<!-- file: tests/test_block_manager.py -->
```python
from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence

B = 256


def finish_prefill(bm, seq):
    # 模拟 prefill：调度 → 计算 → 登记。计算部分在这个测试里省略
    n = bm.can_allocate(seq)
    assert n != -1
    bm.allocate(seq, n)
    seq.num_scheduled_tokens = seq.num_tokens - seq.num_cached_tokens
    bm.hash_blocks(seq)
    seq.num_cached_tokens += seq.num_scheduled_tokens
    seq.num_scheduled_tokens = 0
    return n


# 1. 空账本里分配，不会有命中
bm = BlockManager(10, B)
a = Sequence(list(range(600)))        # 需要 3 块
assert finish_prefill(bm, a) == 0
assert a.block_table == [0, 1, 2] and len(bm.free_block_ids) == 7
# 600 个 token 只写满 2 块，第 3 块没有登记
assert len(bm.hash_to_block_id) == 2

# 2. 前缀相同：前 2 块被共享，只有第 3 块是新的
b = Sequence(list(range(512)) + [9] * 88)
assert finish_prefill(bm, b) == 2
assert b.block_table[:2] == [0, 1] and b.block_table[2] != a.block_table[2]
assert b.num_cached_tokens == 600
assert bm.blocks[0].ref_count == 2 and bm.blocks[1].ref_count == 2

# 3. 只改第 0 块，链式哈希保证第 1 块也不能命中，即使第 1 块内容完全相同
c = Sequence([777] * 256 + list(range(256, 512)) + [1])
assert bm.can_allocate(c) == 0

# 4. 释放共用块：只有引用归零才回到空闲队列
bm.deallocate(a)
assert bm.blocks[0].ref_count == 1 and 0 in bm.used_block_ids and 0 not in bm.free_block_ids
assert 2 in bm.free_block_ids     # a 独占的第 3 块已回收
bm.deallocate(b)
assert 0 in bm.free_block_ids and len(bm.free_block_ids) == 10

# 5. 释放后内容还在：可以复活
d = Sequence(list(range(512)) + [5])
assert finish_prefill(bm, d) == 2
assert d.block_table[:2] == [0, 1] and bm.blocks[0].ref_count == 1
bm.deallocate(d)

# 6. 块被挪用后，映射必须失效
bm2 = BlockManager(3, B)
e = Sequence(list(range(257)))        # 占 2 块，第 0 块被登记
finish_prefill(bm2, e)
bm2.deallocate(e)                     # 尾部先释放
f = Sequence([1] * 700)               # 3 块全部挪用
finish_prefill(bm2, f)
assert bm2.can_allocate(Sequence(list(range(257)))) in (-1, 0), "不能命中被覆盖的块"
bm2.deallocate(f)

# 7. 块数不够
g = Sequence([2] * (B * 4))
assert BlockManager(3, B).can_allocate(g) == -1

# 8. 复活的块也要占空闲名额：只有 2 个空闲块，却需要 3 块（1 块复活 + 2 块新分配）
bm3 = BlockManager(3, B)
h = Sequence(list(range(257)))
finish_prefill(bm3, h)                # 占 2 块，第 0 块登记
i = Sequence([3] * 10)
finish_prefill(bm3, i)                # 占第 3 块
bm3.deallocate(h)                     # 空闲 2 块，其中 1 块带缓存
j = Sequence(list(range(256)) + [8] * 512)   # 命中 1 块，共需 3 块
assert bm3.can_allocate(j) == -1
print("OK")
```

```bash
python tests/test_block_manager.py
```

### 9-B：调度器配合前缀缓存

文件：`tests/test_prefix.py`

<!-- file: tests/test_prefix.py -->
```python
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from test_scheduler import make_scheduler, seq, run_all

# 1. 第二个请求前 2 块命中，只需计算剩余 token
s = make_scheduler()
a = seq(600, max_tokens=1)
s.add(a)
run_all(s)
b = Sequence(a.prompt_token_ids[:512] + [9] * 88, SamplingParams(max_tokens=1))
s.add(b)
seqs, _ = s.schedule()
assert b.num_cached_tokens == 512 and b.num_scheduled_tokens == 88
s.postprocess(seqs, [7], True)
assert len(s.block_manager.free_block_ids) == 100, "块泄漏"

# 2. 完全相同的 prompt：至少也要计算最后一块，才能得到 logits
s = make_scheduler()
a = seq(512, max_tokens=1)
s.add(a)
run_all(s)
b = Sequence(a.prompt_token_ids, SamplingParams(max_tokens=1))
s.add(b)
seqs, _ = s.schedule()
assert b.num_cached_tokens == 256 and b.num_scheduled_tokens == 256

# 3. 预算只计算未命中部分
s = make_scheduler(budget=100)
a = seq(600, max_tokens=1)
s.add(a)
run_all(s)
b = Sequence(a.prompt_token_ids[:512] + [9] * 80, SamplingParams(max_tokens=1))
s.add(b)
seqs, _ = s.schedule()
assert seqs == [b] and b.num_scheduled_tokens == 80, "已命中的 512 个不占预算"
print("OK")
```

```bash
python tests/test_prefix.py
```

### 9-C：真实计算下结果不变

命中缓存后，结果必须和不命中时一样。给第 8 章的 `tests/test_consistency.py` 增加一个场景。在 `compare("普通", ...)` 这一行下面加入：

<!-- insert-after: tests/test_consistency.py :: compare("普通", run(llm, prompts, max_new_tokens), expected) -->
```python

    # 第二次运行：相同 prompt，前缀缓存应该命中
    stats.update(hit=0, chunk=0, preempt=0)
    compare("前缀缓存", run(llm, prompts, max_new_tokens), expected)
    print("  ", stats)
    assert stats["hit"] > 0, "前缀缓存没有命中"
```

```bash
python tests/test_consistency.py
```

**如果前缀缓存这一场景失败**，重点检查 `hash_blocks` 的调用时机和 `num_cached_tokens`。

## 本章小结

**到这里，引擎的逻辑就全部完成了。** 单卡下，`LLM.generate` 已经具备本项目的所有功能。

剩下两章不增加新能力，只解决两个新症状：

- 每轮的时间里，有多少花在了 CPU 上？（第 10 章）
- 模型一张卡放不下怎么办？（第 11 章）
