# 附录 A：API 速查

这里列出项目用到的所有外部 API：它做什么，这里为什么用它。写代码时遇到不认识的调用，先来这里查。括号里是第一次出现的章节。

## Python 标准库

| API | 做什么 | 这里为什么用它 |
|---|---|---|
| `@dataclass(slots=True)`（2） | 自动生成 `__init__`；`slots` 禁止随意增加属性 | 字段名写错会直接报错 |
| `dataclasses.fields`（2） | 列出 dataclass 的所有字段 | 从 `**kwargs` 里筛出 `Config` 认识的参数 |
| `__post_init__`（2） | dataclass 生成的 `__init__` 结束后自动调用 | 校验参数、读取模型配置 |
| `collections.deque`（2） | 两端增删都是 O(1) 的队列 | `waiting`、`running`、空闲块队列 |
| `deque.appendleft` / `extendleft`（8） | 从左边加入。`extendleft` 逐个加入，顺序会反转 | 被抢占的请求回到队首；把本轮序列放回队头 |
| `deque.remove`（2） | 按值删除，O(n) | 结束的请求离开 `running`；复活的块离开空闲队列 |
| `enum.Enum`、`auto`（2） | 定义一组命名常量 | 请求的三种状态 |
| `itertools.count`（2） | 无限递增的计数器 | 生成唯一的 `seq_id` |
| `copy.copy`（2） | 浅拷贝 | 不影响调用方传入的列表 |
| `atexit.register`（2） | 注册解释器退出时运行的函数 | 自动释放 GPU 和子进程 |
| `time.perf_counter`（6） | 高精度计时 | 计算吞吐 |
| `functools.lru_cache`（4） | 按参数缓存函数结果 | 28 层共用一张 RoPE 表 |
| `glob.glob`（4） | 按通配符找文件 | 找到所有 `.safetensors` |
| `pickle.dumps` / `loads`（11） | Python 对象和字节互转 | 把命令发给子进程 |
| `Sequence.__getstate__` / `__setstate__`（11） | 自定义 pickle 的内容 | Decode 只传最后一个 token |
| `multiprocessing.shared_memory.SharedMemory`（11） | 多个进程共用一块内存 | 命令通道 |
| `multiprocessing.synchronize.Event`（11） | 进程间的信号：`set`、`wait`、`clear` | 通知子进程有新命令 |
| `int.to_bytes` / `from_bytes`（9） | 整数和字节互转 | 链式哈希；共享内存的长度头 |

## PyTorch：创建张量

| API | 做什么 | 这里为什么用它 |
|---|---|---|
| `torch.tensor(data, dtype=, pin_memory=True)`（5） | 从列表创建张量。`pin_memory` 把它固定在物理内存 | 之后 GPU 可以用 DMA 异步拷贝 |
| `.cuda(non_blocking=True)`（5） | 放到 GPU，不等拷贝完成 | 每轮都要做，不让 CPU 白等 |
| `torch.empty`（3） | 分配内存但不初始化 | 反正马上要被覆盖，省时间 |
| `torch.empty_like`（3） | 分配和给定张量同形状的内存 | 采样的随机数 |
| `torch.zeros`（10） | 全零张量 | CUDA Graph 的固定缓冲区 |
| `torch.ones`（3） | 全一张量 | `RMSNorm.weight` 的初值 |
| `torch.arange`（4） | 等差序列 | RoPE 的频率和位置 |
| `torch.set_default_device`、`set_default_dtype`、`get_default_dtype`（4） | 设置新张量默认的设备和类型 | 模型和缓存直接以 bf16 建在 GPU 上 |

## PyTorch：形状和切分

| API | 做什么 | 这里为什么用它 |
|---|---|---|
| `Tensor.view`（4） | 改变形状，共用内存 | `[N, 2048] → [N, 16, 128]` |
| `Tensor.split(sizes, dim)`（4） | 按给定长度切开，返回视图 | Q、K、V 长度不同 |
| `Tensor.chunk(n, dim)`（3） | 平均切成 n 份 | gate 和 up；取本卡的权重分片 |
| `Tensor.narrow(dim, start, len)`（3） | 取一段，返回视图 | 把权重写进合并参数的指定区间 |
| `Tensor.flatten(a, b)`（4） | 把第 a～b 维压平 | 多头输出合并 |
| `Tensor.unsqueeze`、`unsqueeze_`（3、4） | 增加长度为 1 的维度 | 广播；FlashAttention 要求的序列维 |
| `Tensor.contiguous`（5） | 保证内存连续 | 防御性写法 |
| `Tensor.stride`（6） | 查询相邻元素的内存间隔 | 切片的内存不一定紧贴，kernel 要用真实间隔 |
| `Tensor.numel`（6） | 元素总数 | 判断缓存是否已分配 |
| `Tensor.size`（10） | 某一维的长度 | 取实际 batch 大小 |

## PyTorch：计算

| API | 做什么 | 这里为什么用它 |
|---|---|---|
| `F.embedding(x, w)`（3） | 查表，等于 `w[x]` | 词嵌入 |
| `F.linear(x, w, b)`（3） | `x @ wᵀ + b` | 所有线性层 |
| `F.silu`（3） | 激活函数 | MLP 的门控 |
| `F.scaled_dot_product_attention`（4） | 注意力 | 只在第 4 章临时使用 |
| `torch.softmax`（3） | 分数变概率 | 采样 |
| `Tensor.exponential_`（3） | 填充指数分布的随机数 | 按概率采样 |
| `Tensor.clamp_min_`（3） | 限制下界 | 防止除零 |
| `Tensor.argmax`（3） | 最大值的下标 | 选出 token |
| `torch.rsqrt`（3） | 平方根的倒数 | RMSNorm |
| `Tensor.pow`、`Tensor.mean`（3） | 平方、求均值 | RMSNorm |
| `Tensor.float()`、`Tensor.to(dtype)`（3） | 类型转换。类型已经相同时**不拷贝** | bf16 和 fp32 互转。第 12 章的隐患就来自“不拷贝” |
| `Tensor.mul_`、`add_`、`div_`、`copy_`、`fill_`、`zero_`（3、10） | 原地操作 | 少分配内存；写入固定地址 |
| `torch.einsum("i,j -> ij", a, b)`（4） | 外积 | 所有位置 × 所有频率 |
| `Tensor.cos`、`sin`（4） | 三角函数 | RoPE 表 |
| `torch.cat`（4） | 沿某一维拼接 | RoPE；拼接各卡的 logits |
| `.tolist()`、`.item()`（4、5） | 张量变 Python 数字。要等 GPU 算完 | 把 token 交回调度器 |

## PyTorch：模块

| API | 做什么 | 这里为什么用它 |
|---|---|---|
| `nn.Module`（3） | 模型零件的基类，`__call__` 调用 `forward` | 所有层 |
| `nn.Parameter`（3） | 注册为权重的张量 | 能被 `get_parameter` 按名字找到 |
| `nn.ModuleList`（3） | 能注册子模块的列表 | 参数名才有 `layers.0.` 前缀 |
| `register_parameter(name, None)`（3） | 声明一个值为 None 的参数 | `self.bias` 总是存在 |
| `register_buffer(name, t, persistent=False)`（4） | 注册不训练的张量；不保存 | RoPE 表 |
| `model.modules()`（6） | 遍历所有子模块 | 找到每层的 Attention，挂上缓存 |
| `model.get_parameter(name)`（4） | 按完整名字取参数，找不到就抛异常 | 模型写错了立刻暴露 |
| `Parameter.data`（3） | 绕过梯度追踪访问数据 | 写入权重；共享权重 |

## PyTorch：运行和显存

| API | 做什么 | 这里为什么用它 |
|---|---|---|
| `torch.inference_mode()`（4） | 不记录梯度信息 | 省时省显存 |
| `torch.compile`（3） | 编译优化一个函数 | Norm、RoPE、采样这些小算子 |
| `torch.cuda.set_device`（4） | 选择当前进程用哪张卡 | 每个 rank 一张 |
| `torch.cuda.synchronize`（4） | 等 GPU 上的任务全部完成 | 退出前、捕获后 |
| `torch.cuda.empty_cache`（7） | 把 PyTorch 缓存的显存还给驱动 | 让显存统计准确 |
| `torch.cuda.reset_peak_memory_stats`（7） | 清零峰值统计 | 测量预热的峰值 |
| `torch.cuda.memory_stats`（7） | 当前进程的显存统计 | `peak`、`current` |
| `torch.cuda.mem_get_info`（7） | 驱动看到的 `(空闲, 总量)` | 计算预算 |
| `dtype.itemsize`（7） | 这种类型占几个字节 | 算一块有多大 |
| `torch.cuda.CUDAGraph`（10） | 图对象 | 录制 GPU 操作 |
| `torch.cuda.graph(graph, pool)`（10） | 上下文管理器，范围内的操作被录制 | 捕获 Decode |
| `graph.pool()`（10） | 取出图的内存池 | 多张图共用，省显存 |
| `graph.replay()`（10） | 回放整张图 | 一个调用跑完所有算子 |

## PyTorch：分布式和多进程

| API | 做什么 | 这里为什么用它 |
|---|---|---|
| `dist.init_process_group("nccl", url, world_size, rank)`（4） | 建立进程组。所有 rank 到齐才返回 | 单卡也要调用，因为层要读 `world_size` |
| `dist.destroy_process_group`（4） | 销毁进程组 | 退出 |
| `dist.get_rank` / `get_world_size`（3） | 自己的编号；总进程数 | 决定切分和通信 |
| `dist.all_reduce(t)`（11） | 所有卡求和，结果原地写回每张卡 | 行并行的结果相加 |
| `dist.gather(t, list, 0)`（11） | 收集到 rank 0 | 收集各卡的 logits |
| `dist.barrier`（11） | 所有 rank 都到达才继续 | 共享内存的创建和关闭顺序 |
| `mp.get_context("spawn")`（11） | 用 spawn 方式启动进程 | CUDA 不支持 fork 后继续使用 |
| `ctx.Process`、`ctx.Event`（11） | 创建子进程和信号 | 启动其他 rank |

## FlashAttention

| API | 做什么 | 这里为什么用它 |
|---|---|---|
| `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale=, causal=, block_table=)`（5） | 变长序列的注意力。`block_table` 不为空时，`k`、`v` 是整个缓存池 | Prefill |
| `flash_attn_with_kvcache(q, k_cache, v_cache, cache_seqlens=, block_table=, softmax_scale=, causal=)`（6） | 用缓存的 K、V 做注意力。`q` 要有序列维 | Decode |

## Triton

| API | 做什么 | 这里为什么用它 |
|---|---|---|
| `@triton.jit`（6） | 把函数编译成 GPU 程序 | 写 KV 的 kernel |
| `kernel[(N,)](...)`（6） | 启动 N 个 program | 每个 program 负责一个 token |
| `tl.program_id(0)`（6） | 当前 program 的编号 | 知道自己处理第几个 token |
| `tl.load`、`tl.store`（6） | 按指针读写 | 读 K、V，写缓存 |
| `tl.arange(0, D)`（6） | 生成偏移向量，长度要在编译时确定 | 一次处理一个 token 的 D 个元素 |
| `tl.constexpr`（6） | 把参数标记为编译期常量 | `D` 要用于 `tl.arange` |

## 其他库

| API | 做什么 | 这里为什么用它 |
|---|---|---|
| `AutoConfig.from_pretrained`（2） | 读取 `config.json` | 模型结构参数 |
| `AutoTokenizer.from_pretrained`（2） | 加载分词器 | 文字和 token 互转 |
| `tokenizer.apply_chat_template`（2） | 套上对话格式 | 模型是按这个格式训练的 |
| `tokenizer.encode` / `decode`（2） | 文字和 token 互转 | 引擎的入口和出口 |
| `tokenizer.eos_token_id`（2） | 结束符的编号 | 判断生成结束 |
| `safetensors.safe_open`（3） | 打开权重文件，按需读取 | 不会一次把整个文件读进内存 |
| `f.keys`、`f.get_tensor`（3） | 列出权重名；读取一个权重 | 逐个加载 |
| `xxhash.xxh64`、`h.update`、`h.intdigest`（9） | 极快的 64 位非加密哈希：喂入字节，输出整数 | 块的链式哈希，结果当字典的键 |
| `np.array(...).tobytes()`（9） | 整数列表变成连续字节 | 喂给哈希器 |
| `tqdm`、`pbar.update`、`pbar.set_postfix`（2、6） | 进度条；在条尾附加信息 | 显示进度和吞吐 |
