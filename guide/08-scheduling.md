# 第 8 章：资源不够时算谁？真正的调度器

**上一章留下的问题**：现在的 `schedule` 有多天真？

```python
while self.waiting:                       # 一次接纳所有请求
    self.block_manager.allocate(seq)      # 块不够 → IndexError，直接崩溃
```

试试同时提交 1000 条请求，程序就崩了。

**本章要回答：** 把 `schedule` 里的问题一个个列出来，由它们决定要写什么。

```text
schedule：这一轮算谁？
├─ Prefill：
│  ├─ Q1 块够吗？不够怎么办？           → can_allocate，先来先服务
│  ├─ Q2 一轮最多算多少？               → 预算 max_num_batched_tokens
│  └─ Q3 prompt 比预算还长怎么办？      → 分块预填充（要读历史 KV）
└─ Decode：
   ├─ Q4 一轮最多算多少条？             → max_num_seqs
   └─ Q5 越过块边界时没有块了怎么办？    → 抢占
```

顺便，每轮都重新做选择，让短请求结束后新请求能立刻加入，这本身就是**动态批处理（Continuous Batching）**。静态批处理要等一批全部结束才接收新请求，短请求结束后它的位置就白白空转。

## 本章任务卡

| | |
|---|---|
| **要解决什么** | 块用完就崩；预算没人遵守；长 prompt 一轮算不完 |
| **要实现什么** | 新的 `schedule`（Prefill 接纳 + Decode 选择）、`preempt`、`postprocess` 的分块逻辑、`can_allocate`、`can_append`，以及读历史 KV 的路径 |
| **要调用什么** | `deque.appendleft`、`deque.extendleft`、`deque.pop`、`while...else`，以及 `flash_attn_varlen_func` 的 `block_table` 参数 |
| **为什么这样做** | 每轮都重新选择谁参与，新请求就能随时加入。资源不够时先等待，实在不够就抢占，所以不会崩溃，也不会丢结果 |

## 8.1 Prefill 调度

```text
while waiting 非空 且 序列数没超限:
    seq = waiting[0]                  只看不取。可能接纳不了，或者要分多轮
    remaining = 本轮还剩多少 token 预算
    remaining == 0 → 结束
    │
    ├─ 新序列：can_allocate；放不下 → 结束
    └─ 分块进行中：num_tokens = 总长度 - 已缓存
    │
    remaining < num_tokens 且 本轮已经有别的序列 → 结束
    │                                （只允许本轮第一条序列分块）
    新序列才 allocate
    num_scheduled_tokens = min(num_tokens, remaining)
    │
    能在本轮算完 → 状态改为 RUNNING，移入 running
    加入本轮列表
```

**Q1：块不够时为什么直接结束，不看后面的短请求？** 这是先来先服务（FCFS）：否则长请求可能永远排不上队。

**Q3：分块预填充是什么？** prompt 有 700 个 token，预算只有 300。那就分三轮：300、300、100。

**为什么只允许第一条序列分块？** 目的是让任意长度的请求都能取得进展。预算够放下后面的序列就放，不够就下一轮，让它们也去分块只会让 batch 变碎。

**分块时为什么算完这一块后循环能自然结束？** `min(num_tokens, remaining)` 恰好用光预算，下一次循环 `remaining == 0` 就 `break`。

**为什么只有算完才进 `running`？** 没算完的还要继续 Prefill，留在 `waiting` 队首，下一轮还是它。

## 8.2 Decode 调度和抢占

Prefill 只分配了 prompt 需要的块。生成时序列变长，越过块边界需要新块，这时可能已经没有了。

```text
while running 非空 且 序列数没超限:
    seq = running.popleft()
    while 新 token 要新块，但没有空闲块:
        running 还有别的序列 → 抢占队尾那条（最新加入的）
        没有了              → 抢占 seq 自己，跳出
    否则（while-else：循环正常结束）：
        seq.num_scheduled_tokens = 1，加入本轮
```

**被抢占的请求怎么办？** 回到 `waiting` **队首**，释放全部块，`num_cached_tokens` 归零。下次重新 Prefill 整条序列（包括已生成的 token），然后继续生成。已生成的内容不丢，只是多算一次。

**为什么抢占最后加入的？** 它已经算的最少，重算的代价最小，先来的请求也更公平。

**Python 的 `while...else`**：`else` 分支在循环条件变成假时执行；`break` 跳出时不执行。所以 `seq` 被抢占时不会被调度。

**最后为什么 `extendleft(reversed(...))`？** `popleft` 把它们取了出来，现在按原来的顺序放回队头。`extendleft` 是一个个往左塞，顺序会反转，所以先 `reversed`。

## 8.3 写 Scheduler

替换 `Scheduler` 的三个方法，并加入 `preempt`：

<!-- method: nanovllm/engine/scheduler.py :: Scheduler -->
```python
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def schedule(self) -> tuple[list[Sequence], bool]:
        scheduled_seqs = []
        num_batched_tokens = 0

        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0:
                break
            # 块表为空：还没分配过（新请求或被抢占的）
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq
                break
            # 放在“确定要接纳”之后才分配，否则上面 break 时块就白占了
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            seq.num_scheduled_tokens = min(num_tokens, remaining)
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens:
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            seq.num_cached_tokens += seq.num_scheduled_tokens
            seq.num_scheduled_tokens = 0
            # 分块中：这个 token 是中间位置的 logits 采样的，要丢弃
            if is_prefill and seq.num_cached_tokens < seq.num_tokens:
                continue
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
```

### 逐行看 schedule

**Prefill 部分：**

| 代码 | 为什么 |
|---|---|
| `seq = self.waiting[0]` | 只看不取。放不下或者没算完时，它要留在队首 |
| `if remaining == 0: break` | 预算用光。分块时会走到这里 |
| `if not seq.block_table` | 块表为空说明没分配过。分块进行中的序列已经有块，不能再分配 |
| `num_tokens = 总长度 - 命中部分` | 本轮之后还要算多少。这一章命中部分总是 0 |
| `if remaining < num_tokens and scheduled_seqs: break` | 放不下且本轮已有序列，留到下一轮 |
| `allocate` 放在上面判断之后 | 确定接纳才分配，否则 `break` 时块就白占了 |
| `min(num_tokens, remaining)` | 分块的全部实现就是这一个 `min` |
| `cached + scheduled == num_tokens` | 本轮能算完，才进入 `running` |
| `if scheduled_seqs: return ..., True` | Prefill 优先，有 Prefill 就不做 Decode |

**Decode 部分：**

| 代码 | 为什么 |
|---|---|
| `self.running.popleft()` | 先取出来。下面可能要抢占队尾，而抢占不能选到它自己 |
| `while not can_append(seq)` | 块不够就一直抢占，直到够了或者没有可抢的 |
| `self.preempt(self.running.pop())` | 抢占队尾，它释放的块可能够用了，回到循环条件再判断 |
| `self.preempt(seq); break` | 只剩自己了，只能把自己抢占。`break` 跳过 `else`，它不进入本轮 |
| `else:` 分支 | 循环条件变假（块够了）时才执行：真正调度这条序列 |
| `assert scheduled_seqs` | 走到这里还是空的，说明单条请求连整个缓存都放不下。见第 12 章 |
| `extendleft(reversed(...))` | 把取出的序列按原顺序放回队头 |

**`postprocess`：** 分块的中间轮，模型也算出了 logits，采样也做了，但这个 token 是根据 prompt 中间的位置预测的，没有意义，所以用 `continue` 丢弃。

### 走一遍：分块预填充

预算 300。请求 0 有 700 个 token，请求 1 有 100 个。下面是真实调度器的运行结果：

| 轮 | 模式 | 本轮的序列（已缓存, 本轮计算, 总长度） | 轮后 waiting | 轮后 running |
|---|---|---|---|---|
| 0 | P | 请求 0（0, 300, 700） | 0, 1 | |
| 1 | P | 请求 0（300, 300, 700） | 0, 1 | |
| 2 | P | 请求 0（600, 100, 700）、请求 1（0, 100, 100） | | 0, 1 |
| 3 | D | 请求 0（700, 1, 701）、请求 1（100, 1, 101） | | |

- 第 0 轮：请求 0 用光预算，下一次循环 `remaining == 0`，请求 1 进不来。
- 第 2 轮：请求 0 只剩 100，预算还剩 200，够放下请求 1，所以两条一起算。这时 `cu_seqlens_q=[0,100,200]`，`cu_seqlens_k=[0,700,800]`。

### 走一遍：抢占

只有 3 个块。请求 0 有 256 个 token，请求 1 有 250 个，请求 2 有 200 个：

| 轮 | 模式 | 发生了什么 | 块表 | 空闲块 |
|---|---|---|---|---|
| 0 | P | 三条各占 1 块 | `[0]`、`[1]`、`[2]` | 0 |
| 1 | D | 请求 0 长度到 257，需要新块，但没有空闲。抢占队尾的请求 2，把它的块给请求 0 | 请求 0 `[0, 2]`，请求 1 `[1]` | 0 |
| 2 | D | 请求 0 生成到第 3 个 token，结束，释放 2 块 | | 轮后 2 |
| 3 | P | 请求 2 回来重新 Prefill。总长度是 **201**，包括被抢占前生成的 1 个 token | 请求 2 `[2]` | 1 |
| 4～ | D | 请求 1、2 继续 | | |

注意第 3 轮：已生成的 token 没有丢，只是 KV 被重算了一遍。最后一个 token 的 logits 也正好用来采样下一个 token。

**这也是抢占的代价**：Prefill 优先，所以第 3 轮请求 1 被插队，等了一轮。

## 8.4 往里一层：BlockManager 要新增的接口

调度器用到了 `can_allocate`、`can_append`，`allocate` 也多了一个参数。

**`num_cached_blocks` 现在总是 0。** 这个参数和返回值都是下一章前缀缓存要用的：`can_allocate` 返回“有多少块可以复用”，-1 表示放不下。

`can_append` 的写法有点巧：比较结果是 bool，True 相当于 1，意思是“需要 1 个空闲块，否则需要 0 个”。

替换和新增这些方法：

<!-- method: nanovllm/engine/block_manager.py :: BlockManager -->
```python
    def can_allocate(self, seq: Sequence) -> int:
        # 放不下返回 -1。否则返回可复用的块数，下一章才会大于 0
        if len(self.free_block_ids) < seq.num_blocks:
            return -1
        return 0

    def allocate(self, seq: Sequence, num_cached_blocks: int):
        assert not seq.block_table
        for _ in range(seq.num_blocks):
            seq.block_table.append(self.free_block_ids.popleft())
        seq.num_cached_tokens = 0

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
```

> `tests/test_paging.py` 用的是 `allocate` 的旧签名，它的任务已经完成，可以删除。

## 8.5 再往里：分块预填充要读历史 KV

分块的第二轮，要计算位置 300～599。但这些 Query 要看**从 0 开始**的所有 Key，而前 300 个 K、V 不在本轮算出的张量里，只在缓存中。

这就是第 5 章留下的“为什么 Q、K 分开”的答案：

```text
seqlen_q = 300    本轮算 300 个 Query
seqlen_k = 600    这些 Query 能看 600 个 Key（300 已缓存 + 300 新写入）
```

因果掩码要**靠右下角对齐**：最后一个 Query（位置 599）看全部 600 个 Key，前一个 Query 看 599 个，以此类推。FlashAttention 就是这样定义的，所以只要传对这两个长度就够了。举个小例子，`seqlen_q=2, seqlen_k=5` 时：

```text
           k0 k1 k2 k3 k4
位置 3 的 q: 1  1  1  1  0
位置 4 的 q: 1  1  1  1  1
```

对应的累积长度。请求 A 已缓存 512、本轮 88；请求 B 无缓存、本轮 50：

```text
cu_seqlens_q = [0,  88, 138]
cu_seqlens_k = [0, 600, 650]
```

第 6 章的 `prepare_prefill` 已经是这样算的，现在只差把块表传下去。**什么时候需要块表？** Key 总数 > Query 总数，说明有序列要读之前的 KV。

在 `prepare_prefill` 里，循环结束后（`slot_mapping.extend(...)` 这一行下面，注意缩进回到循环外）加入：

<!-- insert-after: nanovllm/engine/model_runner.py :: slot_mapping.extend(range(slot_start, slot_end)) -->
```python
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
```

## 8.6 最里层：Attention 从缓存读 K、V

有块表时，K、V 换成整个缓存池。**本轮新算出的 K、V 刚写进缓存，也能读到**，所以一条序列没有历史也没关系。这又一次体现了“先写后读”的好处。

在 `if context.is_prefill:` 下面加入：

<!-- insert-after: nanovllm/layers/attention.py :: if context.is_prefill: -->
```python
            # block_tables 不为 None，表示本批至少有一条序列需要读取历史，
            # 这时所有 K/V 都从缓存读。本轮的 K/V 刚写进去，也能读到
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
```

## 检查点

### 8-A：调度逻辑（不用 GPU）

调度层不碰 GPU 的好处在这里兑现了。用假的 token 就能测试所有场景，几秒钟跑完。

文件：`tests/test_scheduler.py`

<!-- file: tests/test_scheduler.py -->
```python
from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams

B = 256


def make_scheduler(num_blocks=100, budget=16384, max_seqs=512):
    return Scheduler(SimpleNamespace(
        max_num_seqs=max_seqs, max_num_batched_tokens=budget, eos=2,
        kvcache_block_size=B, num_kvcache_blocks=num_blocks,
    ))


def seq(n, max_tokens=4, base=100):
    return Sequence([base + i for i in range(n)], SamplingParams(max_tokens=max_tokens, ignore_eos=True))


def step(s, log=None):
    # 假的一轮：不算 KV，只更新账本
    seqs, is_prefill = s.schedule()
    if log is not None:
        log.append(("P" if is_prefill else "D", [(q.seq_id, q.num_scheduled_tokens) for q in seqs]))
    s.postprocess(seqs, [7] * len(seqs), is_prefill)
    return seqs, is_prefill


def run_all(s, limit=100000):
    for _ in range(limit):
        if s.is_finished():
            return
        step(s)
    raise SystemExit("死循环：调度器没有进展")


if __name__ == "__main__":
    # 1. 正常流程：一轮 Prefill，之后全是 Decode，并产生正确数量的 token
    s = make_scheduler()
    a = seq(10, max_tokens=3)
    s.add(a)
    seqs, is_prefill = step(s)
    assert is_prefill and a.num_cached_tokens == 10 and len(a) == 11   # KV 比总长度少一个
    seqs, is_prefill = step(s)
    assert not is_prefill and a.num_cached_tokens == 11 and len(a) == 12
    step(s)
    assert a.is_finished and a.num_completion_tokens == 3 and s.is_finished()
    assert len(s.block_manager.free_block_ids) == 100, "块泄漏"

    # 2. 动态批处理：短请求结束后，新请求可以中途加入
    s = make_scheduler()
    short, long = seq(5, max_tokens=2), seq(5, max_tokens=10, base=500)
    s.add(short); s.add(long)
    step(s)                                # 两条一起 Prefill
    step(s)                                # short 结束
    assert short.is_finished and not long.is_finished
    new = seq(5, max_tokens=2, base=900)
    s.add(new)
    seqs, is_prefill = step(s)
    assert is_prefill and seqs == [new], "新请求应该立即进入 Prefill"
    seqs, is_prefill = step(s)
    assert not is_prefill and set(seqs) == {long, new}, "现在两者一起 Decode"
    run_all(s)
    assert len(s.block_manager.free_block_ids) == 100, "块泄漏"

    # 3. EOS 提前结束；ignore_eos 不结束
    s = make_scheduler(); s.eos = 7
    stop = Sequence([1, 2, 3], SamplingParams(max_tokens=100))
    go = Sequence([4, 5, 6], SamplingParams(max_tokens=3, ignore_eos=True))
    s.add(stop); s.add(go)
    step(s)
    assert stop.is_finished and not go.is_finished

    # 4. 序列数限制
    s = make_scheduler(max_seqs=2)
    for i in range(3):
        s.add(seq(5, base=1000 * i))
    seqs, _ = s.schedule()
    assert len(seqs) == 2 and len(s.waiting) == 1

    # 5. 块不够时先等待，不崩溃，等前面的结束后再接纳
    s = make_scheduler(num_blocks=2)
    a, b = seq(300, max_tokens=1, base=1), seq(300, max_tokens=1, base=2000)
    s.add(a); s.add(b)
    seqs, _ = s.schedule()
    assert seqs == [a] and list(s.waiting) == [b], "b 放不下，应该继续等"
    s.postprocess(seqs, [7], True)
    assert a.is_finished
    run_all(s)
    assert b.is_finished

    # 6. 分块预填充：预算 300，prompt 700。分三轮 300、300、100，只产生一个 token
    s = make_scheduler(budget=300)
    a = seq(700, max_tokens=1)
    s.add(a)
    log = []
    step(s, log)
    assert log[-1] == ("P", [(a.seq_id, 300)]) and len(a) == 700 and a.status == SequenceStatus.WAITING
    step(s, log)
    assert log[-1] == ("P", [(a.seq_id, 300)]) and len(a) == 700
    step(s, log)
    assert log[-1] == ("P", [(a.seq_id, 100)]) and len(a) == 701 and a.is_finished
    assert len(s.block_manager.free_block_ids) == 100, "块泄漏"

    # 7. 只有第一条序列可以分块；第二条要等预算够了再来
    s = make_scheduler(budget=300)
    a, b = seq(200, base=1), seq(200, base=2000)
    s.add(a); s.add(b)
    seqs, _ = s.schedule()
    assert seqs == [a] and a.num_scheduled_tokens == 200

    # 8. 抢占：4 块只能勉强放下两条序列的 Prefill，它们 Decode 时会越过块边界
    s = make_scheduler(num_blocks=4)
    a, b = seq(500, max_tokens=300, base=1), seq(500, max_tokens=300, base=5000)
    s.add(a); s.add(b)
    preempted = False
    for _ in range(100000):
        if s.is_finished():
            break
        seqs, is_prefill = s.schedule()
        preempted = preempted or (not is_prefill and len(seqs) == 1 and b.status == SequenceStatus.WAITING)
        s.postprocess(seqs, [7] * len(seqs), is_prefill)
    else:
        raise SystemExit("抢占后没有完成")
    assert preempted, "这个场景应该发生抢占"
    assert a.num_completion_tokens == 300 and b.num_completion_tokens == 300
    assert len(s.block_manager.free_block_ids) == 4, "块泄漏"
    print("OK")
```

```bash
python tests/test_scheduler.py
```

### 8-B：读历史 KV 的路径

文件：`tests/test_chunked_prefill.py`

<!-- file: tests/test_chunked_prefill.py -->
```python
import torch
from kv_helpers import *

torch.manual_seed(0)
tokens_a, tokens_b = random_tokens(600), random_tokens(50)
table_a, table_b = [7, 2, 9], [3]
use_cache(True)

# 第 1 轮：A 的前 300 个 + B 全部。A 这时不是完整序列，只检查 B 的 logits
out = prefill([(tokens_a, table_a, 0, 300), (tokens_b, table_b, 0, 50)])
check("Prefill 无历史（序列 B）", out[1], tokens_b)

# 第 2 轮：A 的后 300 个，要读第 1 轮写入的 KV
out = prefill([(tokens_a, table_a, 300, 600)], use_block_table=True)
check("Prefill 读历史（序列 A）", out[0], tokens_a)

# 接着 Decode，确认后续状态正确
tokens_a = tokens_a + [11]
tokens_b = tokens_b + [5]
out = decode([(tokens_a, table_a), (tokens_b, table_b)])
check("Decode（序列 A）", out[0], tokens_a)
check("Decode（序列 B）", out[1], tokens_b)
print("OK")
```

```bash
python tests/test_chunked_prefill.py
```

### 8-C：极限配置下结果不变

正常跑通不代表抢占和分块是对的，这些路径平时很少触发。这个测试**故意**把资源压到极限，要求输出和 HF 的贪心结果一致。

原理：极低温度下，采样几乎等于 argmax。这里用 HF 跑一遍完整前向作为基准，不用任何缓存逻辑。

压缩块数的办法是直接替换账本：账本的块数比物理缓存少，块号自然合法。

**这个测试里最重要的是探针。** 如果只比较结果，假如抢占根本没发生，测试同样会通过，却什么也没有证明。所以脚本统计了这些路径的实际触发次数，并断言它们大于 0。写测试时要时刻问自己：这条路径真的走到了吗？

文件：`tests/test_consistency.py`

<!-- file: tests/test_consistency.py -->
```python
import gc
import os
import random
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from nanovllm import LLM, SamplingParams
from nanovllm.engine.block_manager import BlockManager


def hf_greedy(path, prompts, max_new_tokens):
    # 基准：不用任何缓存，每一步完整前向，取 argmax
    cfg = AutoConfig.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=cfg.dtype).to("cuda").eval()
    results = []
    with torch.inference_mode():
        for p in prompts:
            tokens = list(p)
            for _ in range(max_new_tokens):
                ids = torch.tensor([tokens], device="cuda")
                tokens.append(model(ids, use_cache=False).logits[0, -1].argmax().item())
            results.append(tokens[len(p):])
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return results


def run(llm, prompts, max_new_tokens):
    sp = SamplingParams(temperature=1e-4, ignore_eos=True, max_tokens=max_new_tokens)
    return [o["token_ids"] for o in llm.generate(prompts, sp, use_tqdm=False)]


def compare(name, got, expected):
    first_diffs = []
    for g, e in zip(got, expected):
        assert len(g) == len(e), f"{name}: 长度不同 {len(g)} != {len(e)}"
        first_diffs.append(next((i for i, (x, y) in enumerate(zip(g, e)) if x != y), len(e)))
    print(f"{name:20s} 每条序列第一个不同 token 的位置（{len(expected[0])} 表示完全相同）: {first_diffs}")
    # bf16 舍入可能让接近平分的 argmax 偶尔翻转，所以只严格要求开头一段相同
    assert min(first_diffs) >= min(8, len(expected[0])), f"{name}: 结果太早出现差异，逻辑可能有错"


def add_probes(scheduler):
    # 统计目标路径有没有真的走到。没有走到的话，“结果相同”就证明不了任何事
    stats = {"hit": 0, "chunk": 0, "preempt": 0}
    schedule, preempt = scheduler.schedule, scheduler.preempt

    def schedule_probe():
        seqs, is_prefill = schedule()
        if is_prefill:
            stats["hit"] += sum(s.num_cached_tokens > 0 and s.num_cached_tokens % 256 == 0 for s in seqs)
            stats["chunk"] += sum(s.num_cached_tokens + s.num_scheduled_tokens < s.num_tokens for s in seqs)
        return seqs, is_prefill

    def preempt_probe(seq):
        stats["preempt"] += 1
        preempt(seq)

    scheduler.schedule, scheduler.preempt = schedule_probe, preempt_probe
    return stats


def main():
    path = os.path.expanduser(os.environ.get("MODEL_PATH", "~/huggingface/Qwen3-0.6B/"))
    tp = int(os.environ.get("TP", 1))
    max_new_tokens = 40
    random.seed(0)
    vocab = min(AutoConfig.from_pretrained(path).vocab_size, 10000)
    shared = [random.randint(0, vocab - 1) for _ in range(520)]    # 超过 2 块的公共前缀
    prompts = [
        shared + [random.randint(0, vocab - 1) for _ in range(30)],
        shared + [random.randint(0, vocab - 1) for _ in range(10)],
        [random.randint(0, vocab - 1) for _ in range(500)],        # 生成时会越过块边界
        [random.randint(0, vocab - 1) for _ in range(240)],        # 生成时也会越过块边界
    ]
    expected = hf_greedy(path, prompts, max_new_tokens)

    llm = LLM(path, enforce_eager=True, tensor_parallel_size=tp, max_model_len=1024, max_num_batched_tokens=4096)
    stats = add_probes(llm.scheduler)
    compare("普通", run(llm, prompts, max_new_tokens), expected)

    # 分块预填充。换一本新账本，排除前面运行留下的影响
    stats.update(hit=0, chunk=0, preempt=0)
    llm.scheduler.max_num_batched_tokens = 300
    llm.scheduler.block_manager = BlockManager(32, 256)
    compare("分块预填充", run(llm, prompts, max_new_tokens), expected)
    print("  ", stats)
    assert stats["chunk"] > 0, "没有发生分块"

    # 抢占：账本只有 9 块，Prefill 正好用完，生成时越过块边界就要抢占
    stats.update(hit=0, chunk=0, preempt=0)
    llm.scheduler.max_num_batched_tokens = 4096
    llm.scheduler.block_manager = BlockManager(9, 256)
    compare("抢占", run(llm, prompts, max_new_tokens), expected)
    print("  ", stats)
    assert stats["preempt"] > 0, "没有发生抢占"
    print("OK")


if __name__ == "__main__":
    main()
```

```bash
python tests/test_consistency.py
```

**如果某个场景失败：**

| 失败场景 | 重点检查 |
|---|---|
| 普通 | 第 6 章，`positions`、`context_lens`、slot |
| 分块预填充 | `postprocess` 的 `continue` 分支，`slot_start` 的块内偏移 |
| 抢占 | `preempt` 是否清空状态；重算时 `num_tokens` 要包括已生成的 token |

## 本章小结

现在请求再多也不会崩了。

但看测试里的 `prompts`：前两条有 520 个相同的 token，它们却各算了一遍。真实场景里，所有请求常常共用同一段系统提示，这种浪费非常严重。

下一章解决它。**你会发现几乎只需要改账本。** 上层用到的 `num_cached_blocks`、读历史 KV 的路径，这章都准备好了。
