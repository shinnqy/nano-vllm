# 第 7 章：缓存该分多少块？

**上一章留下的问题**：`num_kvcache_blocks = 64` 是写死的。

- 小了，显卡大量显存闲置，同时能处理的请求变少；
- 大了，直接 OOM。

**本章要回答**：怎么让引擎自己算出合适的块数？

这一章只改 `ModelRunner`（外层依然不用动）。

## 本章任务卡

| | |
|---|---|
| **要解决什么** | 块数写死成 64：小了浪费显存，大了 OOM |
| **要实现什么** | `warmup_model`、自动估算的 `allocate_kv_cache`，以及 `Config` 的四个资源配置 |
| **要调用什么** | `torch.cuda.empty_cache`、`reset_peak_memory_stats`、`mem_get_info`、`memory_stats`、`dtype.itemsize` |
| **为什么这样做** | 运行时的临时显存算不出来，只能实测。先按最大负载跑一次，再把剩下的显存全部给缓存 |

## 7.1 能给缓存多少显存？

```text
可用 = 总显存 × 利用率 - 已占用 - 运行时还要用的临时显存
```

前两项可以直接查询，**最后一项是难点**：模型运行时的中间张量要占多少？无法直接算，只能**实测**。

办法是：分配缓存之前，用**最大负载**跑一遍模型，记录显存峰值。这就是**预热（warmup）**。

那“最大负载”是多少？一轮最多算多少 token，这本来就应该是一个配置，否则一轮太久或 OOM。

在 `Config` 的 `model: str` 下面加入：

<!-- insert-after: nanovllm/config.py :: model: str -->
```python
    max_num_batched_tokens: int = 16384    # 一轮 Prefill 最多算多少 token
    max_num_seqs: int = 512                # 一轮最多多少条请求
    max_model_len: int = 4096              # 单条序列最大长度
    gpu_memory_utilization: float = 0.9    # 显存预算比例
```

序列长度不能超过模型位置编码支持的长度，在 `self.hf_config = AutoConfig.from_pretrained(self.model)` 下面加入：

<!-- insert-after: nanovllm/config.py :: self.hf_config = AutoConfig.from_pretrained(self.model) -->
```python
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
```

## 7.2 启动顺序

从 `__init__` 往下看，顺序必须是：

```text
创建模型 → 加载权重 → 预热测峰值 → 根据结果分配缓存
```

**为什么预热必须在分配之前？** 缓存要把显存“吃满”，如果先分配，运行时就可能 OOM。

在 `self.sampler = Sampler()` 下面加入（它正好排在上一章的 `self.allocate_kv_cache()` 前面）：

<!-- insert-after: nanovllm/engine/model_runner.py :: self.sampler = Sampler() -->
```python
        self.warmup_model()
```

## 7.3 预热

用**全零 token** 构造一批序列，长度和数量取配置允许的最大负载，跑一次 Prefill：

```python
seq_len = min(max_num_batched_tokens, max_model_len)              # min(16384, 4096) = 4096
num_seqs = min(max_num_batched_tokens // seq_len, max_num_seqs)   # min(4, 512) = 4
```

**为什么是 Prefill？** 一轮 Prefill 的 token 数最多，是运行时显存的峰值。

**这时还没有缓存，代码能跑吗？** 这就是上一章 `Attention` 里 `if k_cache.numel()` 的作用之一：缓存还是空张量时跳过写入。

添加到 `ModelRunner`：

<!-- method: nanovllm/engine/model_runner.py :: ModelRunner -->
```python
    def warmup_model(self):
        torch.cuda.empty_cache()    # 释放 PyTorch 持有但没用的显存，让统计更准确
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()    # 预热的临时张量不再需要，把显存还回去
```

**逐行看 warmup_model：**

| 代码 | 为什么 |
|---|---|
| 第一个 `empty_cache()` | PyTorch 会留着释放过的显存备用。先还给驱动，统计才干净 |
| `reset_peak_memory_stats()` | 峰值清零，后面读到的峰值才是这次预热造成的 |
| `seq_len = min(batched, max_model_len)` | 单条序列不会比这更长 |
| `num_seqs = min(batched // seq_len, max_num_seqs)` | 能放几条就放几条，但总量不超过预算 |
| `Sequence([0] * seq_len)` | 显存占用只看形状，token 内容无所谓 |
| `seq.num_scheduled_tokens = seq_len` | 这些序列没有经过调度器，所以手工设置 `prepare_prefill` 要读的字段 |
| `self.run(seqs, True)` | 走和真实 Prefill 一样的代码，测出来的才是真实占用 |
| 第二个 `empty_cache()` | 临时张量已释放，把它们占的显存还回去，下面的 `mem_get_info` 才看得到 |

这里有两个连带的修改：

**① 预热不关心采样**，所以 `Sequence` 的 `sampling_params` 要有默认值：

<!-- replace-line: nanovllm/engine/sequence.py :: def __init__(self, token_ids: list[int], sampling_params: SamplingParams): -->
```python
    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
```

**② 预热的序列没有块表**，`prepare_prefill` 要跳过 slot 计算。在 `max_seqlen_k = max(seqlen_k, max_seqlen_k)` 下面加入：

<!-- insert-after: nanovllm/engine/model_runner.py :: max_seqlen_k = max(seqlen_k, max_seqlen_k) -->
```python
            if not seq.block_table:    # warmup
                continue
```

## 7.4 用实测结果算块数

```python
free, total = torch.cuda.mem_get_info()
used = total - free
peak = memory_stats["allocated_bytes.all.peak"]         # 预热时的峰值
current = memory_stats["allocated_bytes.all.current"]   # 当前已分配（主要是权重）

可用 = total × 利用率 - used - (peak - current)
       ↑ 预算           ↑ 已经占用   ↑ 运行时临时张量还需要的量
```

`peak - current` 是关键：**峰值减去常驻量，就是运行时的临时张量量**。

一块的字节数：

```text
block_bytes = 2 × 28 层 × 256 × 8 头 × 128 维 × 2 字节 = 29,360,128 字节 = 28 MiB
```

替换 `allocate_kv_cache`：

<!-- method: nanovllm/engine/model_runner.py :: ModelRunner -->
```python
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # 多卡时每张卡只存一部分 KV 头（第 11 章）
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        # 第 2 个参数在属性缺失时才用到，所以 0.6B 用的是配置里的 128
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        # 这时默认类型是 bf16、默认设备是 cuda，所以不用另外指定
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        # modules() 按定义顺序遍历。用 hasattr 找到每层的 Attention，不用知道模型结构
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]   # 切片共享内存，不会多占显存
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
```

### 走一遍估算公式

下面的数字是**假设的**，只用来演示计算。一张 8 GiB 的卡，单位 MiB：

| 项 | 值 | 来源 |
|---|---|---|
| `total` | 8192 | `mem_get_info` |
| `used` | 1300 | `total - free`，主要是模型权重 |
| `peak - current` | 900 | 预热时临时张量的最大占用 |
| 可用 | 8192 × 0.9 − 1300 − 900 = 5172.8 | |
| `block_bytes` | 28 | 第 6.3 节算过 |
| 块数 | 5172 // 28 = 184 | |
| 能放的 token 总数 | 184 × 256 = 47104 | 所有请求共用 |

**几个容易想错的地方：**

- `mem_get_info` 来自驱动，包括其他进程的占用。`memory_stats` 只统计当前进程里 PyTorch 分配的张量。两者口径不同，这里各取所需。
- 为什么不直接减 `peak`？`peak` 包含常驻的权重，而权重已经算在 `used` 里了。减去 `peak - current`，只留下临时部分，才不会重复扣除。
- `hf_config.dtype.itemsize`：bf16 是 2 字节。这里不写死，因为其他模型可能是别的类型。
- `int(...) // block_bytes`：`total * 0.9` 是浮点数，要转成整数。整除向下取整，宁可少一块，也不超预算。

## 检查点

```bash
python example.py
```

运行时用 `nvidia-smi` 观察，显存占用应该接近 `0.9 × 总显存`。

如果报 `assert config.num_kvcache_blocks > 0`，说明显存不够：调小 `max_num_batched_tokens`，或者关掉其他占显存的进程。

> **预热的第二个好处**：`torch.compile`、Triton 的首次编译都发生在这时，不用等到处理真实请求。

## 本章小结

块数现在是自动的，**但它终究有限**。这引出一连串新问题：

- 请求太多，块不够用，上一章的 `free_block_ids.popleft()` 会直接抛 `IndexError`。
- `max_num_batched_tokens` 配了，还没有人遵守它。
- 一条很长的 prompt 超过了预算怎么办？

`Scheduler` 该变聪明了。
