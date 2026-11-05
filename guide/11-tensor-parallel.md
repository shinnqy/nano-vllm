# 第 11 章：一张卡不够？张量并行

**上一章留下的问题**：模型或 KV 缓存放不进一张卡。

**本章要回答**：

```text
LLM(path, tensor_parallel_size=2)
├─ 谁来启动第二张卡的工作？                     → 子进程
├─ 子进程怎么知道每轮要做什么？                  → 共享内存派发命令
├─ 模型怎么拆？                                → 并行线性层
└─ Sequence 怎么高效地发过去？                  → 自定义 pickle
```

第 2 章 `example.py` 传的 `tensor_parallel_size` 终于要用上了。

如果手头只有一张 GPU，这章的代码也可以写完。`tensor_parallel_size=1` 时行为不变，多卡检查点留到有条件时再跑。

## 本章任务卡

| | |
|---|---|
| **要解决什么** | 一张卡放不下，或者想用多张卡分摊 |
| **要实现什么** | 启动子进程、`call`/`loop`/`read_shm`/`write_shm`、`Sequence` 的自定义 pickle、全部并行线性层、并行的嵌入和输出头 |
| **要调用什么** | `mp.get_context("spawn")`、`Event`、`SharedMemory`、`pickle`、`dist.all_reduce`、`dist.gather`、`dist.barrier`、`Tensor.chunk`、`Tensor.narrow` |
| **为什么这样做** | 调度只留在 rank 0，子进程只算。切分方式让注意力完全不用通信，所以 `Attention` 和 KV 缓存的代码一行都不用改 |

## 11.1 总体架构

```text
         主进程（rank 0）                    子进程（rank 1, 2, ...）
 ┌───────────────────────────┐         ┌──────────────────────────┐
 │ LLMEngine                 │         │ ModelRunner.__init__     │
 │  ├ Scheduler + 账本（唯一） │         │   建模型、建缓存          │
 │  └ ModelRunner(rank 0)    │         │   loop()  ← 不会返回      │
 │      call("run", seqs, f) ├─共享内存─→   读命令 → 执行同名方法   │
 │                           │  + Event│                          │
 │      GPU 计算   ←── NCCL ──→ GPU 计算                           │
 │      采样、返回             │         │   不采样、继续等          │
 └───────────────────────────┘         └──────────────────────────┘
```

**数据通道有两条，不能混淆：**

| 通道 | 传什么 | 实现 |
|---|---|---|
| 命令 | “接下来运行 `run`，参数是这些序列” | 共享内存 + `Event` |
| 张量 | 层间的隐藏状态 | NCCL（`dist.all_reduce`、`gather`） |

**为什么 Python 参数不走 NCCL？** NCCL 只传固定形状的 GPU 张量。序列列表是 Python 对象，长度也不固定，用 pickle + 共享内存更合适。

**`Scheduler` 始终只有 rank 0 一份。** 这就是第 4 章要求 `run` 不修改序列状态的原因：子进程只负责计算。

## 11.2 最外层：LLMEngine 启动子进程

**为什么用 spawn 启动？** CUDA 不支持 fork 后继续使用。这也意味着脚本必须有 `if __name__ == "__main__":`，否则子进程导入脚本时会递归启动。第 2 章的 `example.py` 早就这样写了。

替换 `LLMEngine` 的这几个方法。`step` 里只有一处变化：`run` 改成 `call("run", ...)`。

<!-- method: nanovllm/engine/llm_engine.py :: LLMEngine -->
```python
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        # 主进程自己就是 rank 0，所以从 1 开始。target 是类：子进程执行它的 __init__ 并停在 loop 里
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")   # 同时通知所有子进程
        del self.model_runner
        for p in self.ps:
            p.join()                     # 等子进程退出，避免留下僵尸进程

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens
```

在 `from transformers import AutoTokenizer` 下面加入：

<!-- insert-after: nanovllm/engine/llm_engine.py :: from transformers import AutoTokenizer -->
```python
import torch.multiprocessing as mp
```

## 11.3 往里：ModelRunner 的命令派发

**`call` 的设计**：`call("run", a, b)` 在 rank 0 上先写共享内存发给其他 rank，再调用自己的 `run(a, b)`。这样 `LLMEngine` 不需要关心是否多卡。

**`__init__` 末尾的 barrier 是为了什么？** 只有 rank 0 能创建共享内存，别的进程必须在它创建之后才能打开。`dist.barrier()` 恰好保证这个顺序：

```text
rank 0: 创建 → barrier ─┐
                        ├ 都到达才继续
rank 1:      barrier ───┘ → 打开
```

**共享内存格式**：前 4 字节是长度（小端），之后是 pickle 数据。

**为什么 rank 0 能写完直接往下走？** 每个进程都要执行相同的 NCCL 通信，所以各卡会在通信点上自然对齐。子进程慢了，rank 0 就会在 `all_reduce` 上等它。

**`read_shm` 的时序**：等 Event → 读 → `clear`。为什么不会丢命令？rank 0 下一次写命令之前，一定完成了这一轮的 NCCL 通信，而子进程要 `clear` 之后才会开始计算。所以写入时一定已经 clear 过。

替换和新增这些方法：

<!-- method: nanovllm/engine/model_runner.py :: ModelRunner -->
```python
    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event     # rank 0：所有子进程的 Event 列表；子进程：自己的 Event

        # 2333 是各 rank 交换初始化信息的端口，张量通信走 NCCL
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()    # 子进程永远停在这里，直到收到 exit

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()     # 确保大家都关闭了
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        # 只有 rank 0 采样，子进程没有温度，也不需要
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids
```

### 走一遍：2 张卡的一轮

```text
rank 0                                     rank 1
schedule()                                 event.wait() 等命令
call("run", seqs, is_prefill)
  write_shm：pickle → 写共享内存 → event.set()
  run(seqs, is_prefill)                    被唤醒，读命令，event.clear()
                                           call("run", seqs, is_prefill)
    prepare_prefill / prepare_decode         prepare_prefill / prepare_decode
    Embedding → all_reduce ←─────────────→ all_reduce
    每层：
      qkv_proj、注意力（各算各的）            qkv_proj、注意力
      o_proj → all_reduce ←──────────────→ all_reduce
      gate_up、down_proj → all_reduce ←───→ all_reduce
    LM Head → gather ←───────────────────→ gather
    采样，得到 token_ids                    logits=None，不采样
postprocess()                              回到 event.wait()
```

**逐行看：**

| 代码 | 为什么 |
|---|---|
| `event: Event \| list[Event]` | rank 0 要唤醒所有子进程，所以持有全部 Event。子进程只等自己的 |
| `SharedMemory(name="nanovllm", create=True, size=2**20)` | 1 MiB 的共享内存。名字固定，所以子进程能打开它 |
| `self.loop()` 在 `__init__` 里 | 子进程的 `target` 就是这个类。创建对象就进入循环，直到退出 |
| `int.from_bytes(self.shm.buf[0:4], "little")` | 前 4 字节存数据长度，否则不知道 pickle 数据到哪结束 |
| `getattr(self, method_name, None)` | 用字符串找到同名方法。命令就是方法名加参数 |
| `if self.world_size > 1 and self.rank == 0` | 只有 rank 0 发命令。子进程执行 `call` 时不再转发 |
| `temperatures = ... if self.rank == 0 else None` | Sampler 只在 rank 0 运行。子进程的序列也没有温度字段 |
| `self.shm.close()` 和 `unlink()` | 每个进程都要 `close`，但只有创建者 `unlink`（删除） |

**退出流程：** `atexit` 调用 `exit` → `call("exit")` 通知子进程 → 大家在 `dist.barrier()` 处会合，确保都关闭了共享内存 → rank 0 删除共享内存 → 主进程 `p.join()` 等子进程结束。

文件头部要多三个导入。把 `import torch.distributed as dist` 这一行替换成：

<!-- replace-line: nanovllm/engine/model_runner.py :: import torch.distributed as dist -->
```python
import torch.distributed as dist
import pickle
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
```

## 11.4 序列怎么发过去：自定义 pickle

**问题**：Decode 每轮都要把序列发到子进程。序列越长 pickle 越大。但 Decode 只需要最后一个 token，发整条序列是巨大的浪费。

Python 的 pickle 允许自定义：定义 `__getstate__`（决定传什么）和 `__setstate__`（对端如何恢复）。

**用 `is_prefill` 决定发送什么：**

- Prefill：子进程要取 `seq[start:end]`，发完整 `token_ids`。
- Decode：发 `last_token` 一个整数。

增加两个方法：

<!-- method: nanovllm/engine/sequence.py :: Sequence -->
```python
    # pickle 发送时调用。只传子进程计算需要的字段
    def __getstate__(self):
        last_state = self.last_token if not self.is_prefill else self.token_ids
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)

    # pickle 接收时调用。不会经过 __init__，所以字段都在这里设置
    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state = state
        # 用类型区分两种情况：列表是 Prefill，整数是 Decode
        if isinstance(last_state, list):
            self.token_ids = last_state
            self.last_token = self.token_ids[-1]
        else:
            self.token_ids = []
            self.last_token = last_state
```

**子进程收到的序列里有什么？** 只有 `__getstate__` 打包的这几个字段：

| 字段 | 子进程用它做什么 |
|---|---|
| `num_tokens` | `len(seq)`，算位置和块数 |
| `num_cached_tokens`、`num_scheduled_tokens` | `prepare_prefill` 确定 `start`、`end` |
| `block_table` | 算 slot，读缓存 |
| `last_state` | Prefill 时是整个列表，Decode 时只有最后一个 token |

没有 `seq_id`、`temperature`、`status`。子进程根本不需要它们，这也就保证了它们不可能在子进程里被用错。

**一个容易忽略的点**：`num_blocks`、`last_block_num_tokens` 用的是类属性 `block_size`。这个类属性不在 pickle 里，第 12 章第 3 条讲的就是它带来的隐患。

在 `self.num_scheduled_tokens = 0` 下面加入这个字段：

<!-- insert-after: nanovllm/engine/sequence.py :: self.num_scheduled_tokens = 0 -->
```python
        self.is_prefill = True     # 决定跨进程时传整条序列，还是只传最后一个
```

**谁来维护这个标记？** 调度器。序列第一次进入 Decode 时改为 False：

<!-- insert-after: nanovllm/engine/scheduler.py :: seq.num_scheduled_tokens = 1 -->
```python
                seq.is_prefill = False
```

被抢占后要重新 Prefill，子进程又需要完整序列了，所以改回 True：

<!-- insert-after: nanovllm/engine/scheduler.py :: seq.status = SequenceStatus.WAITING -->
```python
        seq.is_prefill = True
```

## 11.5 最里层：模型怎么拆

每一层的矩阵按行或列切开，每张卡算一部分，用少量通信合并。这就是**张量并行（TP）**。

核心是一个二人组（第 3、4 章类名里的 `Parallel` 终于有了意义）：

```text
ColumnParallelLinear：输出维度切开。 输入完整 → 每张卡输出一部分
                 ↓ 中间运算（每张卡独立）
RowParallelLinear：  输入维度切开。 输入部分 → 每张卡算出完整形状的“部分和” → all_reduce 相加
```

**为什么 Row 这样切是对的？** 把矩阵按列切开：

```text
y = W x = [W₁ W₂] [x₁] = W₁x₁ + W₂x₂
                  [x₂]
卡 0 算 W₁x₁，卡 1 算 W₂x₂，all_reduce 求和，就得到 y
```

### 每一层怎么切（2 张卡）

```text
隐藏状态 [N, 1024]：每张卡都有完整一份（复制状态）
│
├─ Embedding         词表切开：每张卡查自己的范围，其他位置置零，all_reduce 求和
│
├─ Attention
│   ├─ qkv_proj      Column：每张卡得到 8 个 Q 头 + 4 个 K 头 + 4 个 V 头
│   ├─ 注意力         每张卡独立计算自己的头。无需通信！
│   │                每张卡的 KV 缓存也只存自己的 4 个头 ← 显存减半的来源
│   └─ o_proj        Row：输入 1024 → 输出 1024，all_reduce  ← 通信
│
├─ MLP
│   ├─ gate_up_proj  Column：每张卡 [N, 3072]（gate 和 up 各 1536）
│   ├─ SiluAndMul    每张卡独立
│   └─ down_proj     Row，all_reduce                        ← 通信
│
└─ LM Head           词表切开，每张卡算部分 logits，gather 到 rank 0
```

**神奇的地方**：Attention 按头切开后，不同的头本来就互相独立，所以注意力本身**不用通信**。`Attention`、`store_kvcache` 都不用改，因为它们只看到自己的头。这就是第 4 章 `Qwen3Attention` 里头数要除以 `tp_size` 的原因，第 7 章分配缓存时除以 `world_size` 也是这个原因。

**每层通信 2 次**，再加上首尾各 1 次。通信量和 `hidden_size` 有关，而和词表、KV 缓存都无关，所以可以接受。

### 并行线性层

**加载权重**：每张卡只取自己那一片。

- Column：沿第 0 维（输出）切，`narrow(0, rank × 片大小, 片大小)`。
- Row：沿第 1 维（输入）切。偏置不切，但只有 rank 0 使用，否则相加时会加 N 次。

合并层要做两步切分：先把权重切出当前卡的那片，再放进合并参数的对应区间。

文件：`nanovllm/layers/linear.py`（替换为最终版）

<!-- file: nanovllm/layers/linear.py -->
```python
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist


def divide(numerator, denominator):
    # 切分必须能整除，否则直接报错
    assert numerator % denominator == 0
    return numerator // denominator


class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,      # 沿权重的哪一维切：0=Column，1=Row
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):
    # 每张卡一份完整权重。本模型没用到，留给可能需要的模型

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


class ColumnParallelLinear(LinearBase):

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


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        # 偏移和大小都按“每张卡的份额”计算
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size)
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank]
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

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

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
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


class RowParallelLinear(LinearBase):

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
        # 偏置是一维，形状是输出大小，不需要切
        if param_data.ndim == 1:
            param_data.copy_(loaded_weight)
            return
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 偏置只让 rank 0 加，否则 all_reduce 后会变成 N 倍
        y = F.linear(x, self.weight, self.bias if self.tp_rank == 0 else None)
        if self.tp_size > 1:
            dist.all_reduce(y)
        return y
```

**用数字看每一层的切分（2 张卡）：**

| 层 | 单卡权重形状 | 每张卡的权重形状 | 输入 → 输出（每张卡） |
|---|---|---|---|
| `qkv_proj` | `[4096, 1024]` | `[2048, 1024]` | `[N,1024] → [N,2048]` |
| 其中 Q / K / V | 2048 / 1024 / 1024 行 | 1024 / 512 / 512 行 | 8 个 Q 头、4 个 K 头、4 个 V 头 |
| `o_proj` | `[1024, 2048]` | `[1024, 1024]` | `[N,1024] → [N,1024]`，再求和 |
| `gate_up_proj` | `[6144, 1024]` | `[3072, 1024]` | `[N,1024] → [N,3072]` |
| `down_proj` | `[1024, 3072]` | `[1024, 1536]` | `[N,1536] → [N,1024]`，再求和 |

**以 rank 1 加载 `k_proj` 为例（`[1024, 1024]`）：**

```text
num_heads=8，num_kv_heads=4，head_size=128
shard_offset = 8 × 128 = 1024      合并参数里 K 从第 1024 行开始
shard_size   = 4 × 128 = 512
param_data.narrow(0, 1024, 512)    目标位置
loaded_weight.chunk(2, 0)[1]       取原权重的后一半，[512, 1024]
```

**`qkv_proj` 为什么用 `chunk` 而不是 `narrow`？** 这时形状已经知道了，直接平分取第 rank 份，写起来更简单，结果相同。

### 词嵌入和输出层

**Embedding 的前向**：把 token id 换成本卡索引；不属于本卡的，索引置 0（随便查一个），结果再乘 mask 清零。all_reduce 后，每个位置只有一张卡贡献了真实向量。

**LM Head 的前向**：`F.linear(x, weight)` 得到本卡负责的那部分词的分数。`dist.gather` 收集到 rank 0，再按最后一维拼接。其他卡返回 `None`，它们不采样。

文件：`nanovllm/layers/embed_head.py`（替换为最终版）

<!-- file: nanovllm/layers/embed_head.py -->
```python
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


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
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        logits = F.linear(x, self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits
```

**走一遍 Embedding：** 2 张卡，每张负责 75968 个词。token id 是 100000，属于 rank 1：

| | rank 0（0～75967） | rank 1（75968～151935） |
|---|---|---|
| `mask` | False | True |
| `x = mask * (x - start)` | 0 | 24032 |
| `F.embedding(x, weight)` | 查到第 0 个词的向量（无意义） | 查到正确的向量 |
| `mask.unsqueeze(1) * y` | 全 0 | 正确的向量 |
| `all_reduce` 求和后 | 正确的向量 | 正确的向量 |

`x` 先乘 `mask`，是因为减完可能是负数或超出范围，不能拿去查表。

**走一遍 LM Head：** 每张卡的 `logits` 是 `[B, 75968]`。`gather` 把它们收集到 rank 0，按最后一维拼接成 `[B, 151936]`，按 rank 的顺序拼接正好就是词表的顺序。这一步通信量最大，但只发生一次，而且只取了每条序列的最后一个位置。

> `tests/test_parts.py` 里的线性层测试没有初始化进程组，从现在起不能再运行了。它的任务也已经完成。

## 检查点

**① 单卡回归**：

```bash
python example.py
```

```bash
python tests/test_block_manager.py
```

```bash
python tests/test_scheduler.py
```

```bash
python tests/test_prefix.py
```

```bash
python tests/test_kv_cache.py
```

```bash
python tests/test_chunked_prefill.py
```

```bash
python tests/test_consistency.py
```

**② 2 卡一致性**（需要至少 2 张 GPU）：

```bash
TP=2 python tests/test_consistency.py
```

结果应该和单卡一样通过。

**③ 观察显存**：运行时用 `nvidia-smi` 看，每张卡的模型权重显存大约减半，能分配到的 KV 块数大约翻倍（每个块的字节数减半了）。

如果 2 卡时卡住不动：

- 通常是某个 rank 执行了不同的通信序列。用 `py-spy dump --pid <子进程 pid>` 看它停在哪一行。
- 检查 `RowParallelLinear` 和 `gather` 的调用条件，所有 rank 必须走同样的通信路径。

## 本章小结

**你已经实现了本项目的全部功能。** 再回头看一眼第 2 章的 `example.py`：它从第一天起就一行没变，而里面已经从一个随机数发生器，长成了一个完整的推理引擎。
