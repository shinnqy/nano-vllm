# 第 2 章：从入口写起，先让骨架跑通

**本章的问题**：目录是空的。入口怎么写？它需要哪些部件？

**本章的做法**：先写用户的代码，然后运行它。报错缺什么，就写什么。里面暂时做不了的（模型），就写一个**假的**，先把整条链路打通。

```text
你在这里：
example.py → LLM → LLMEngine.generate → step → Scheduler / ModelRunner（假）
```

## 本章任务卡

|                        |                                                                                                                           |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **要解决什么**   | 目录是空的，不知道从哪下手                                                                                                |
| **要实现什么**   | `example.py`、`LLM`、`SamplingParams`、`LLMEngine`、`Config`、`Sequence`、`Scheduler`、假的 `ModelRunner` |
| **要调用什么**   | `dataclass`、`deque`、`itertools.count`、`tqdm`、`AutoConfig`、`AutoTokenizer`、`apply_chat_template`       |
| **为什么这样做** | 先把外壳和接口定下来。之后每一章都只是把里面换得更真实，外层几乎不用动                                                    |

## 2.1 先写用户要写的代码

项目的打包文件先准备好，让 `import nanovllm` 在任何目录都能工作：

文件：`pyproject.toml`

<!-- file: pyproject.toml -->

```toml
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "nano-vllm"
version = "0.2.0"
requires-python = ">=3.10,<3.13"
dependencies = [
    "torch>=2.4.0",
    "triton>=3.0.0",
    "transformers>=4.56.0",
    "flash-attn",
    "xxhash",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["nanovllm*"]
```

```bash
mkdir -p nanovllm/engine nanovllm/layers nanovllm/models nanovllm/utils
```

```bash
pip install -e . --no-deps
```

`-e` 是可编辑安装，之后改代码不用重装。`--no-deps` 是因为依赖已经装好，不要让 pip 再去动 torch。子目录**不写** `__init__.py` 也能导入，本项目就是这样做的。

现在写入口。**这是全项目第一个要写的文件：**

文件：`example.py`

<!-- file: example.py -->

```python
import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("~/huggingface/Qwen3-0.6B/")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
```

`apply_chat_template` 把用户的话套上模型训练时用的对话格式。少了这步，模型后面会胡言乱语。

逐行看 `example.py`：

- `AutoTokenizer.from_pretrained(path)`：这里加载分词器，只为套对话模板。引擎内部还会自己加载一份，用来编码和解码。
- `tokenize=False`：只拼出字符串，不转 token。`generate` 接收字符串后会自己调用 `tokenizer.encode`，这里转了就重复了。
- `add_generation_prompt=True`：在末尾加上“轮到 assistant 说话了”的标记。少了它，模型可能接着用户的话往下写，而不是回答。
- `if __name__ == "__main__":`：现在只是习惯写法，第 11 章多进程时就是必需的了。

`enforce_eager`、`tensor_parallel_size` 现在不用管，第 10、11 章才有用，先接受这两个参数。

运行它，当然会报错：

```bash
python example.py
```

```text
ImportError: cannot import name 'LLM' from 'nanovllm'
```

**这就是待办清单。** 这行代码告诉我们需要 `LLM` 和 `SamplingParams`，一个个写。

## 2.2 第一层：对外的名字

文件：`nanovllm/__init__.py`

<!-- file: nanovllm/__init__.py -->

```python
from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams
```

`LLM` 只是个对外的名字，真正的逻辑放在 `LLMEngine`：

文件：`nanovllm/llm.py`

<!-- file: nanovllm/llm.py -->

```python
from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    pass
```

`SamplingParams` 表示“这**一个**请求想怎么生成”。为什么它属于请求，而不属于引擎？因为同一批里的请求可以有不同的温度和长度。

文件：`nanovllm/sampling_params.py`

<!-- file: nanovllm/sampling_params.py -->

```python
from dataclasses import dataclass


# slots=True：不用 __dict__ 存属性。写错字段名（如 sp.temprature = 1）会直接报错，
# 而不是悄悄多出一个新属性，让 bug 难以发现。
@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0   # 越小越保守，越大越随机
    max_tokens: int = 64       # 最多生成多少 token（不含 prompt）
    ignore_eos: bool = False   # True 时即使生成了结束符也继续，压测用

    # dataclass 生成的 __init__ 结束后会自动调用这个方法，适合校验
    def __post_init__(self):
        # 采样时要做 logits / temperature，温度为 0 会除零。
        # 所以直接禁止，而不是把 0 当作“贪心”特殊处理，这样采样器能保持最简单。
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
```

## 2.3 第二层：LLMEngine

先想 `generate` 怎么写：

1. 把每个 prompt 变成一个请求，放进调度器；
2. 只要还有请求没结束，就走一步；
3. 所有请求结束后，按提交顺序返回。

“走一步”单独写成 `step`，内容就是第 1 章的三件事。**写到这里，我们立刻知道还需要什么：**

| `LLMEngine` 用到了 | 它必须提供什么                                          |
| -------------------- | ------------------------------------------------------- |
| `Config`           | 模型路径等引擎级设置                                    |
| `Sequence`         | 保存一个请求的 token 和状态                             |
| `Scheduler`        | `add`、`schedule`、`postprocess`、`is_finished` |
| `ModelRunner`      | `run(seqs)` 返回每条序列的下一个 token                |

文件：`nanovllm/engine/llm_engine.py`

<!-- file: nanovllm/engine/llm_engine.py -->

```python
import atexit
from dataclasses import fields
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        # 只把 Config 认识的字段传过去
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.model_runner = ModelRunner(config)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id    # 调度器要靠它判断结束，所以先于调度器设置
        self.scheduler = Scheduler(config)
        # Python 退出时自动清理，用户不需要显式调用
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.exit()
        del self.model_runner

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        # 同时接受文本和 token 列表（压测直接传 token）
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs = self.scheduler.schedule()                   # ① 这一轮算谁
        token_ids = self.model_runner.run(seqs)            # ② 算出下一个 token
        self.scheduler.postprocess(seqs, token_ids)        # ③ 更新进度
        return [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        # 只传一个参数时，所有请求共用
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        while not self.is_finished():
            for seq_id, token_ids in self.step():
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        # 完成顺序不固定，按 seq_id 排序才能和输入一一对应
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
```

**为什么提交和执行要分开（`add_request` 和 `step`）？** 调度器要在一轮开始前，综合考虑所有等待和运行中的请求。第 8 章会充分用到这个结构。

### 逐行看 LLMEngine

**`__init__`：顺序不能乱。**

| 代码                                          | 为什么                                                                                                                                  |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `fields(Config)` 那三行                     | `LLM(path, enforce_eager=True, ...)` 的参数原样传进来。只挑 `Config` 有的字段，这样接口和 vLLM 相似，多传一个不认识的参数也不会报错 |
| `Config(model, **config_kwargs)`            | 所有引擎级设置都集中到一个对象里，后面的部件只接收它                                                                                    |
| `ModelRunner(config)` 在 `Scheduler` 之前 | 第 7 章会由`ModelRunner` 算出块数写回 `config`，`Scheduler` 再读。现在先按这个顺序写                                              |
| `config.eos = tokenizer.eos_token_id`       | 结束符属于分词器。调度器要靠它判断请求结束，所以要在创建调度器之前设置                                                                  |
| `atexit.register(self.exit)`                | Python 退出时自动释放 GPU 资源，用户不用记得调用                                                                                        |

**`step`：一轮循环，完全对应第 1 章那张图。** 最后一行只返回**本轮刚结束**的序列，让上层把结果收走。

**`generate`：**

- `[sampling_params] * len(prompts)`：列表里是同一个对象的多个引用。这没问题，因为 `Sequence` 只读取它的值。
- `outputs` 用字典，键是 `seq_id`：短请求先结束，完成顺序和提交顺序不一样。
- `sorted(outputs.keys())`：`seq_id` 是递增的，排序后就是提交顺序。多次调用 `generate` 时 `seq_id` 会接着增长，排序依然正确。
- `tokenizer.decode`：这里才把 token 变回文字，引擎内部全程只处理整数。

下面按需求一个个往里写。

## 2.4 第三层：Config

模型的层数、头数都写在模型目录的 `config.json` 里，引擎不要写死，直接读取。

现在只放用得到的字段。后面的章节遇到新需求时再来加。

文件：`nanovllm/config.py`

<!-- file: nanovllm/config.py -->

```python
import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1

    def __post_init__(self):
        # 要求本地目录：后面要找权重文件，不支持写 repo id
        assert os.path.isdir(self.model)
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
```

**逐行看 Config：**

- `@dataclass`：自动生成 `__init__`，字段写一行就够了。
- `hf_config: AutoConfig | None = None`：用户不传，由 `__post_init__` 填入。`AutoConfig.from_pretrained` 读取模型目录的 `config.json`，层数、头数都从这里来。
- `eos: int = -1`：先占位。`-1` 不是合法的 token，所以不会误判结束，由 `LLMEngine` 填入真值。
- `assert 1 <= tensor_parallel_size <= 8`：配置一错就立刻报错，好过跑到一半才崩。

## 2.5 第三层：Sequence

一个请求要记住什么？目前只要这些：

- 全部 token（prompt + 已生成）；
- prompt 有多长（用来切出输出）；
- 现在是什么状态；
- 它的采样参数。

**为什么 prompt 和输出放在同一个列表？** 模型看到的就是一条连续的序列，它并不区分两者。

文件：`nanovllm/engine/sequence.py`

<!-- file: nanovllm/engine/sequence.py -->

```python
from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    counter = count()      # 类属性：全局递增，生成唯一 seq_id

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        # copy：之后要往里追加 token，不能影响调用方传入的列表
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    # 实现这两个方法后，就可以写 len(seq)、seq[start:end]
    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
```

**逐行看 Sequence：**

| 代码                  | 为什么                                                                                                      |
| --------------------- | ----------------------------------------------------------------------------------------------------------- |
| `counter = count()` | 写在类上，所有序列共用一个计数器，`next()` 一次得到一个唯一编号                                           |
| `copy(token_ids)`   | 后面要往里追加。不拷贝就会改掉调用方传进来的列表                                                            |
| `last_token`        | 现在看起来多余。第 6 章 Decode 每轮只喂最后一个 token；第 11 章子进程只收到这一个值，那时已经没有完整列表了 |
| `num_tokens` 单独存 | 同样是为了第 11 章：子进程的`token_ids` 可能是空的，长度却还要用                                          |
| `num_prompt_tokens` | 切分输入和输出的分界线。`completion_token_ids` 就靠它切片                                                 |
| `@property`         | 让`seq.is_finished` 看起来是属性，实际每次现算，不用担心和 `status` 不同步                              |

## 2.6 第三层：最简单的 Scheduler

现在只做最简单的策略：**所有请求每轮都算。**

两个队列：`waiting` 放新来的，`running` 放正在生成的。为什么现在就分两个？新请求和运行中的请求，处理方式本来就不一样，到第 6 章你就能看到。

文件：`nanovllm/engine/scheduler.py`

<!-- file: nanovllm/engine/scheduler.py -->

```python
from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus


class Scheduler:

    def __init__(self, config: Config):
        self.eos = config.eos
        # deque：两端弹出都是 O(1)。list.pop(0) 是 O(n)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> list[Sequence]:
        while self.waiting:
            seq = self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
        return list(self.running)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]):
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.running.remove(seq)
```

**逐行看 Scheduler：**

- `schedule` 里的 `return list(self.running)`：**这里必须复制一份。** 后面 `postprocess` 遍历 `seqs` 时，会调用 `self.running.remove(seq)`。如果 `seqs` 就是 `running` 本身，边遍历边删除，deque 会抛 `RuntimeError`。
- `postprocess` 里的结束条件：遇到 EOS，或达到 `max_tokens`。`ignore_eos` 用于压测，让每条请求都生成到指定长度。
- `zip(seqs, token_ids)`：两个列表一一对应，这是 `run` 接口的约定：**输出顺序必须和输入顺序相同。** 后面所有版本的 `run` 都要遵守。

## 2.7 第三层：假的 ModelRunner

真正的模型是下一章的事。现在让它**随机返回一个 token**。这是为了先确认外面整条链路是通的。

**接口才是这里的重点**：输入一批序列，输出每条序列的下一个 token。只要这个接口不变，外层的代码后面就不用动。

文件：`nanovllm/engine/model_runner.py`

<!-- file: nanovllm/engine/model_runner.py -->

```python
from random import randrange

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence


class ModelRunner:

    def __init__(self, config: Config):
        self.vocab_size = config.hf_config.vocab_size

    def exit(self):
        pass

    def run(self, seqs: list[Sequence]) -> list[int]:
        return [randrange(self.vocab_size) for _ in seqs]
```

## 检查点

```bash
python example.py
```

你会看到进度条走完，然后打印出**一堆乱码般的随机词**。这就对了！

为了确认结构正确，再写个小测试：

文件：`tests/test_skeleton.py`

<!-- file: tests/test_skeleton.py -->

```python
import os
from nanovllm import LLM, SamplingParams


def main():
    path = os.path.expanduser(os.environ.get("MODEL_PATH", "~/huggingface/Qwen3-0.6B/"))
    llm = LLM(path)
    # 长度不同的请求：短的先结束，但结果要和输入顺序对应
    outputs = llm.generate(
        [[1, 2, 3], [4, 5], [6]],
        [SamplingParams(max_tokens=n, ignore_eos=True) for n in (9, 2, 5)],
        use_tqdm=False,
    )
    assert [len(o["token_ids"]) for o in outputs] == [9, 2, 5]
    assert llm.is_finished()
    print("OK")


if __name__ == "__main__":
    main()
```

```bash
python tests/test_skeleton.py
```

### 走一遍：测试里的三条请求

三条请求的 `max_tokens` 分别是 9、2、5。每一行是一次 `step()`：

| 轮次 | `schedule` 返回              | 本轮结束的序列         | `running` 里还剩         |
| ---- | ------------------------------ | ---------------------- | -------------------------- |
| 1    | 三条全部（从`waiting` 移入） | 无                     | 三条                       |
| 2    | 三条                           | 第 2 条（已生成 2 个） | 第 1、3 条                 |
| 3～4 | 第 1、3 条                     | 无                     | 第 1、3 条                 |
| 5    | 第 1、3 条                     | 第 3 条（已生成 5 个） | 第 1 条                    |
| 6～8 | 第 1 条                        | 无                     | 第 1 条                    |
| 9    | 第 1 条                        | 第 1 条（已生成 9 个） | 空，`is_finished()` 为真 |

完成顺序是 2、3、1。按 `seq_id` 排序后，输出恢复成 1、2、3。

## 本章小结：现在有什么，缺什么

```text
example.py ✓
LLMEngine  ✓  主循环已经是真实的
Sequence   ✓  目前够用
Scheduler  △  策略太简单
ModelRunner ✗ 假的！
```

**新的症状：输出是随机词。** 下一章把假的 `ModelRunner` 换成真的。
