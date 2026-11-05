# 第 1 章：全局地图

## 1.1 我们要实现什么

最终目标只有这几行：

```python
from nanovllm import LLM, SamplingParams

llm = LLM("/path/to/model")
outputs = llm.generate(["你好"], SamplingParams(temperature=0.6, max_tokens=256))
print(outputs[0]["text"])
```

用户给一批文字，拿回一批文字。整个项目的所有代码，都是为了让这几行**又快又省显存**。

## 1.2 第一层拆解：generate 必须做什么

大模型不会一次写出整段话。它**一次只产生一个 token**（可以理解成一个词的片段）：

```text
"你好" ─分词器→ [108386] ─模型→ 每个词一个分数 ─采样→ 下一个 token ─┐
                  ▲                                              │
                  └───────────── 追加到序列，再来一次 ────────────┘
```

所以 `generate` 的主体必然是一个循环。如果同时有多条请求，每转一圈，要回答三个问题：

| 问题                                    | 负责的部件                |
| --------------------------------------- | ------------------------- |
| ① 这一圈算哪些请求？                   | `Scheduler`（调度器）   |
| ② 算出它们的下一个 token               | `ModelRunner`（执行器） |
| ③ 算完后每个请求到哪一步了，谁结束了？ | `Scheduler`             |

把循环和这几件事串起来的，就是 `LLMEngine`。

```text
 llm.generate(prompts, sampling_params)
          │
          ▼
 ┌─ LLMEngine ───────────────────────────────┐
 │  把每个请求放进队列                         │
 │  while 还有请求没结束:                      │
 │    ① Scheduler.schedule()     这轮算谁      │
 │    ② ModelRunner.run()        算下一个 token│
 │    ③ Scheduler.postprocess()  更新进度      │
 └───────────────────────────────────────────┘
```

**这张图就是整个项目的骨架。** 后面所有内容，都是把这三个方框里面逐步填实。

## 1.3 这个骨架会遇到的问题

下面这些问题你现在不用懂，只需要知道它们**是按顺序冒出来的**。这就是后面章节的顺序：

```text
generate
└─ 循环：schedule → run → postprocess        第 2 章 先用假模型跑通
   │
   ├─ run 要真正产生 token
   │   ├─ logits → token：采样               第 3 章
   │   └─ token → logits：模型
   │       ├─ 外层：嵌入、Decoder 层、输出头   第 3 章
   │       └─ 核心：注意力、位置编码、加载权重  第 4 章
   │
   ├─ 症状：请求一条条排队算 → 批处理          第 5 章
   ├─ 症状：越生成越慢 → KV Cache、分页        第 6 章
   ├─ 症状：缓存大小是写死的 → 自动估算         第 7 章
   │
   ├─ schedule 要变聪明
   │   └─ 症状：请求一多就崩 → 真正的调度      第 8 章
   │
   ├─ 症状：相同开头反复计算 → 前缀缓存        第 9 章
   ├─ 症状：CPU 拖后腿 → CUDA Graph           第 10 章
   └─ 症状：一张卡不够 → 张量并行              第 11 章
```

注意：我们不会先写“KV Cache”这个零件。而是等程序真的慢下来，你亲眼看到原因之后，才去解决它。

## 1.4 开始前要认识的几个词

后面的代码里，这些词每一页都会出现。这里先建立直觉，不用背。

| 词 | 是什么 | 在代码里的样子 |
|---|---|---|
| **token** | 模型处理文字的最小单位，通常是一个词或词的片段 | 一个整数，比如 `108386` |
| **tokenizer（分词器）** | 文字和 token 互相转换的工具 | `tokenizer.encode("你好")`、`tokenizer.decode(ids)` |
| **词表大小** | 一共有多少种 token | `vocab_size = 151936` |
| **隐藏向量（hidden state）** | 模型内部表示一个 token 的一串数字 | 形状 `[N, 1024]`，N 个 token，每个 1024 维 |
| **logits** | 模型给词表里每个词打的分，分数越高越可能是下一个词 | 形状 `[批大小, 151936]` |
| **权重（weights）** | 别人训练好的那些矩阵。我们只负责加载，不训练 | 目录里的 `*.safetensors` 文件 |
| **推理（inference）** | 用训练好的权重生成结果，不再更新权重 | 不需要梯度，所以能省很多时间和显存 |

**形状的字母约定**，全书统一：

| 字母 | 含义 | Qwen3-0.6B 的值 |
|---|---|---|
| `N` | 本轮喂给模型的 token 总数 | 每轮不同 |
| `H` / `Hkv` | Q 的头数 / K、V 的头数 | 16 / 8 |
| `d` | 每个头的维度 | 128 |
| `L` | 层数 | 28 |
| `B` | 每个缓存块放多少 token（第 6 章） | 256 |
| `V` | 词表大小 | 151936 |

**如果你不熟悉 PyTorch，只需要先知道这五点：**

1. `nn.Module` 是“模型零件”的基类。写 `layer(x)` 时，实际执行的是它的 `forward(x)`。
2. `nn.Parameter` 是挂在零件上的权重。零件嵌套时，参数名会按属性名拼起来，比如 `layers.0.mlp.down_proj.weight`。第 4 章靠这个名字把权重文件对应到模型上。
3. 名字以下划线结尾的方法（`mul_`、`copy_`、`div_`）是**原地操作**，直接改张量本身，不分配新内存。
4. `view`、`narrow`、`split`、切片返回的是**视图**：和原张量共用一块内存，写视图就是写原张量。第 3 章放权重、第 6 章放缓存都靠这一点。
5. `torch.tensor([...])` 默认在 CPU 上，算之前要放到 GPU（`.cuda()`）。

代码里每个外部 API 的作用和选择理由，都收在[附录 A](appendix-a-api.md)。遇到不认识的调用，先去那里查。

**看代码时，始终问自己四个问题：**

```text
这段代码要解决什么问题？   → 看每章开头的“任务卡”
为什么非这样写不可？       → 看代码后面的“逐行看”
它调用了什么？             → 看任务卡和附录 A
跑起来数据长什么样？       → 看“走一遍”，那些数字可以自己手算核对
```

## 1.5 准备环境

后面第 2 章不需要 GPU，但第 3 章开始就要用，所以先一次装好。

```bash
conda create -n nanovllm python=3.11 -y
```

```bash
conda activate nanovllm
```

去 [pytorch.org](https://pytorch.org/get-started/locally/) 选择和你的 CUDA 驱动匹配的安装命令。`cu124` 只是示例，要换成你那条命令里的版本：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

**一定要先装好 torch，再装下面这些。** `flash-attn` 是要用 CUDA 编译的 C++ 扩展，安装时需要找到 CUDA 版的 torch。

```bash
pip install "transformers>=4.56" triton xxhash numpy tqdm safetensors einops
```

`--no-build-isolation` 的意思是：编译时直接用当前环境的 torch，而不是让 pip 另外隔离下载一份（可能是 CPU 版）：

```bash
pip install flash-attn --no-build-isolation
```

下载模型。这个模型只有 0.6B 参数，普通显卡就能跑，很适合学习：

```bash
hf download Qwen/Qwen3-0.6B --local-dir ~/huggingface/Qwen3-0.6B
```

如果没有 `hf` 命令，就用 `huggingface-cli download`，参数相同。

> **关于 transformers 版本**：项目的 `pyproject.toml` 声明 `>=4.51.0`，但代码用到了 `hf_config.dtype`。老版本读取这个模型的配置时可能没有这个属性，所以这里写成 `>=4.56`。

每个依赖负责什么，等用到时再讲。

### 检查点：环境可用

创建一个空目录 `my-vllm/`，然后创建 `tests/`。

文件：`tests/check_gpu.py`

<!-- file: tests/check_gpu.py -->

```python
import torch
import triton
import transformers
import xxhash  # 只验证能导入
from flash_attn import flash_attn_varlen_func

print("torch", torch.__version__, "transformers", transformers.__version__, "triton", triton.__version__)
assert torch.cuda.is_available(), "没有可用的 CUDA：检查驱动和 torch 版本"
print("GPU:", torch.cuda.get_device_name(0), "数量:", torch.cuda.device_count())

# 假装有 1 条长度为 8 的序列，4 个头，每头 64 维。这些含义第 5 章会讲
q = torch.randn(8, 4, 64, dtype=torch.bfloat16, device="cuda")
cu_seqlens = torch.tensor([0, 8], dtype=torch.int32, device="cuda")
o = flash_attn_varlen_func(q, q, q, cu_seqlens, cu_seqlens, 8, 8, causal=True)
assert o.shape == q.shape
print("FlashAttention OK, 输出形状", tuple(o.shape))
```

```bash
python tests/check_gpu.py
```

能看到 `FlashAttention OK` 就算通过。下一章开始写入口。
