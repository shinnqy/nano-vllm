# 第 12 章：排错、已知局限和下一步

## 12.1 常见错误速查

| 现象 | 原因 | 解决办法 |
|---|---|---|
| `flash-attn` 安装时报找不到 torch | 隔离环境里没有 torch | 先装 torch，加 `--no-build-isolation` |
| `Config` 没有 `dtype` 属性 | transformers 太旧 | 升级，见 12.2 第 1 条 |
| `Address already in use`（端口 2333） | 另一个引擎还在运行，或上次没退干净 | 结束旧进程。同一机器要跑多个实例，得把代码里的端口改成可配置 |
| `FileExistsError: nanovllm`（共享内存） | 上次多卡运行异常退出 | 删除 `/dev/shm/nanovllm` |
| `assert config.num_kvcache_blocks > 0` | 显存不足 | 调小 `max_num_batched_tokens` 或 `max_model_len`，清理其他进程 |
| 输出胡言乱语，形状和逻辑都没报错 | 没有用 `apply_chat_template`，或 RoPE 基数错误 | 先过第 4 章的 HF 对比 |
| `get_parameter` 报 `no attribute` | 模型属性名和权重名不对应 | 对照第 3 章打印的权重名 |
| `Inplace update to inference tensor` | 在 inference mode 之外修改 `graph_vars` | `run_model` 要加 `@torch.inference_mode()` |
| CUDA Graph 模式下结果错 | padding 行写脏了缓存 | 检查 `slot_mapping.fill_(-1)` |
| 多卡时启动就不断重启进程 | 脚本没写 `if __name__ == "__main__"` | 补上 |

**调试技巧：**

- 一律先用 `enforce_eager=True` 排除 CUDA Graph 的影响。
- 调度问题先用 `tests/test_scheduler.py` 的假模型复现，不用等 GPU。
- 逻辑问题用 `temperature=1e-4` + `ignore_eos=True`，让结果确定，才好对比。

## 12.2 项目的已知局限

在通读代码、写这份指南的过程中，发现了以下这些点。指南的最终版代码和项目**保持一致**，并没有修复它们，这样方便你对照阅读。但你应该知道它们的存在。**可以把它们当练习来修复。**

这些点都是读代码分析得到的。第 2、3、5、7、10 条的现象已用 CPU 上的小脚本确认，其余的是逻辑推断，还没有实际复现。

**1. `pyproject.toml` 的 transformers 版本下限偏低。** 声明的是 `>=4.51.0`，但代码用了 `hf_config.dtype`，老版本可能只有 `torch_dtype`。第 1 章把下限写成 `>=4.56`。

**2. RMSNorm 对 float32 不安全（第 3 章提到）。** 张量已是 float32 时，`x.float()` 和 `.to()` 返回同一个对象，后面的 `mul_` 会把残差和输入一起改掉。而且第 3 章的采样器也有同样的情况：`logits.float()` 不产生新张量，就要在 inference mode 外原地改推理张量，会直接报错。修复：

```python
# layernorm.py：rms_forward 和 add_rms_forward 里，用强制拷贝替换 x.float()
x = x.to(torch.float32, copy=True)
# layernorm.py：add_rms_forward 里，残差也要是独立的一份
residual = x.to(orig_dtype, copy=True)
# sampler.py
logits = logits.to(torch.float32, copy=True).div_(temperatures.unsqueeze(dim=1))
```

**3. 多卡时，`kvcache_block_size` 不是 256 就会算错槽位。** `LLMEngine.__init__` 里的 `Sequence.block_size = ...` 只修改了主进程，spawn 启动的子进程重新导入模块后还是默认的 256，`prepare_decode` 里的 `last_block_num_tokens` 就错了。修复：在 `ModelRunner.__init__` 里也设置一次 `Sequence.block_size`。

**4. 各 rank 的块数可能不同。** 每个 rank 按自己卡的剩余显存算 `num_kvcache_blocks`，而账本只用 rank 0 的结果。如果子进程的卡上还有其他程序，块数更少，就会越界。修复：`dist.all_reduce(..., op=MIN)` 取最小值。

**5. 共享内存只有 1 MiB。** Prefill 要传完整的 `token_ids`，序列多、又都很长时（比如大量前缀命中的长序列，已缓存的部分不占预算，却仍然要传输），pickle 结果可能超过 1 MiB，赋值时会报错。修复：加大 `size`，或者发送前检查长度。

**6. 没有校验长度上限。** `add_request` 不检查 `prompt 长度 + max_tokens` 是否超过 `max_model_len`。超过时 CUDA Graph 的 `block_tables` 放不下，超过 `max_position_embeddings` 时 RoPE 查表越界。修复：在 `add_request` 里断言。

**7. 超长请求会让调度器崩溃。** 如果一条请求需要的块比整个缓存池还多，`can_allocate` 一直返回 -1，`running` 又是空的，就会走到 `assert scheduled_seqs`，直接崩溃。抢占自己后同样可能发生。修复：提前拒绝这种请求。

**8. Prefill 优先可能饿死 Decode。** 持续有新请求到来时，`running` 里的请求一直得不到调度。这是有意的简化。`LLM.generate` 是一次性提交，所以不成问题，但改成在线服务就要处理（比如 Prefill 和 Decode 轮流）。

**9. EOS 只有一个。** 只用了 `tokenizer.eos_token_id`，而 `generation_config.json` 可能列出多个结束符。

**10. `max_num_seqs` 不能被 16 整除时，图模式可能崩溃（第 10 章提到）。** 修复：

```python
self.graph_bs = [x for x in [1, 2, 4, 8] if x < max_bs] + list(range(16, max_bs, 16)) + [max_bs]
```

**11. 输出里带着结束符文本。** `tokenizer.decode` 没有传 `skip_special_tokens=True`，所以结果会以 `<|im_end|>` 结尾。

**12. 没有处理排队顺序的公平性。** `running` 超过 `max_num_seqs` 时，总是调度队头的序列，后面的要等它们结束。

## 12.3 全量自查

能回答下面这些问题，才算真的掌握了。答不上来，回到对应章节。

**模型（第 3、4 章）**

1. 为什么 `head_dim × num_heads` 可以不等于 `hidden_size`？
2. 为什么权重文件里是 `q_proj`，模型里是 `qkv_proj`？合并的好处是什么？
3. `v` 没有做 RoPE，为什么？这又给 KV 写入带来什么影响？

**缓存（第 6 章）**

4. 写出 `slot(t)` 公式。`positions` 和 `slot_mapping` 的区别是什么？
5. 为什么先写 KV 再做 Attention？
6. `cu_seqlens_q` 和 `cu_seqlens_k` 什么时候不同？
7. Decode 的 `context_lens` 为什么包含当前 token？

**账本（第 6、9 章）**

8. 为什么哈希必须是链式的？
9. 为什么最后一块不参与前缀匹配？
10. “空闲且保留缓存”的块，复活时为什么要占用空闲名额？
11. `deallocate` 为什么是 `reversed`？
12. `len(seq) % B == 1` 为什么是 1？
13. `hash_blocks` 为什么放在 `num_cached_tokens` 更新之前？

**调度（第 8 章）**

14. 一条 Prefill 分块进行时，中间轮次采样出来的 token 怎么处理？为什么？
15. 被抢占的请求重新运行时，已经生成的 token 怎么办？
16. 为什么不把 Prefill 和 Decode 放进同一个 batch？

**性能（第 10 章）**

17. CUDA Graph 为什么要预分配缓冲区？为什么要分档？
18. 为什么大图要先捕获？
19. padding 行的 `slot_mapping` 为什么是 -1？如果是 0 会怎样？

**并行（第 11 章）**

20. `RowParallelLinear` 的偏置为什么只在 rank 0 使用？
21. 注意力计算为什么不需要通信？
22. Decode 时，子进程为什么只收到 `last_token` 也能工作？

## 12.4 下一步练习

按难度递增：

1. **修复 12.2 的局限**：从第 6、7、11 条开始。
2. **加 Top-K / Top-P 采样**：改 `SamplingParams`、`Sampler`，再把新参数传过 `prepare_sample`。
3. **流式输出**：把 `generate` 改成生成器，每轮 `yield` 新 token。
4. **统计前缀缓存命中率**：在 `BlockManager` 里加计数器。
5. **在线服务**：用 FastAPI 接收请求，后台循环调用 `step()`。解决第 8 条的饥饿问题。
6. **混合 Prefill 和 Decode**：调度器允许同一轮两者并存。提示：可以都走 `flash_attn_varlen_func`。
7. **支持新模型**：写 `models/llama.py`，用 `architectures` 字段选择模型类。

## 12.5 和真正的 vLLM 对照

写完这个项目后再读 vLLM 会很轻松，主要概念都能一一对应：

| 本项目 | vLLM 里对应的概念 |
|---|---|
| `Scheduler` | `Scheduler`（多了优先级、并发上限等策略） |
| `BlockManager` | KV cache 管理器 |
| `ModelRunner` | Worker 中的模型执行部分 |
| `set_context` / `get_context` | 注意力元数据 |
| `QKVParallelLinear` 等 | 同名的并行层 |
| `graph_bs` | CUDA Graph 档位 |

vLLM 额外解决的是在线服务、分布式、更多模型、更多硬件等工程问题，核心思路和你刚写的是一样的。

## 指南本身的验证范围

这份指南是在**没有 GPU 的机器**上写的。下面如实列出验证过什么、没验证什么。

**已实际运行并通过的：**

| 内容 | 方式 |
|---|---|
| 最终代码与项目源码一致 | 用脚本把所有章节的修改依次应用，得到的 21 个 Python 文件逐一与项目做 AST 比对（忽略注释和定义顺序），全部相同 |
| 第 1～9、11 章的检查点 | 每章修改应用后，在 CPU 上运行当章脚本 |
| 第 2 章起入口一直能跑 | 第 2、4、5、6、7、9、11 章结束时都实际运行了 `example.py` |
| 前缀缓存、分块预填充、抢占 | `test_consistency.py` 用探针确认都发生了，结果和 HF 逐 token 相同 |
| 张量并行的逻辑 | 2 个进程，Gloo 代替 NCCL，权重切分和通信结果与 HF 相同 |
| 图模式的缓冲区和 padding 逻辑 | 单进程，第 10 章和最终代码各跑一次，用“重新执行模型”代替 `replay()` |
| Triton kernel 的指针和 stride 计算 | 在 Python 里逐行执行 kernel 函数体，用真实存储模拟指针 |

**运行环境里的替代品（所以上面的结果有局限）：**

- 模型是**结构相同的微型随机模型**（2 层、hidden 32），不是真实的 Qwen3-0.6B；
- FlashAttention 是自己写的 Python 参考实现，不是真实算子；
- `torch.compile` 被替换成了空操作；
- 版本是 torch 2.15.0、transformers 5.22.1。transformers 4.56 只验证了 `dtype` 属性可用。

**这次修订（增加任务卡、逐行讲解、走查和附录）又重新检查了这些：**

| 内容 | 方式和结果 |
|---|---|
| 最终代码与项目源码一致 | 重新把各章代码依次应用，21 个文件全部 AST 相同（比对时忽略注释、字符串注释和定义顺序） |
| 不依赖 GPU 的检查点 | `test_parts`（第 3、4 章版本）、`test_paging`、`test_scheduler`、`test_block_manager`、`test_prefix` 全部通过，运行时给 `flash_attn`、`triton`、`Qwen3Config` 放了只为导入成功的桩 |
| 第 8、9 章走查里的数字 | 由真实的 `Scheduler` 和 `BlockManager` 跑出来，不是手写的 |
| 第 6 章的 slot 数字 | 用和 `prepare_prefill` 相同的算法计算 |

上面“已实际运行”那张表是之前编写时的记录。这次只重跑了本表列出的部分。

**没有验证、需要你在 GPU 机器上第一次跑时确认的：**

- 真实的 FlashAttention、Triton 编译、NCCL、`torch.compile`；
- 真实的 CUDA Graph 捕获和回放，以及显存估算（`mem_get_info`、`memory_stats`）；
- 图模式下的多卡运行（这种组合连替代版也没跑通过：子进程拿不到替代的 Graph）；
- 第 1 章的安装命令；
- 各章“耗时变化”的描述，这是按原理推断的，微型模型上测不出来；
- 真实模型的输出质量和吞吐量，以及测试里的差异阈值。这些阈值是按经验设的，bf16 下如果只是阈值附近略有超出，先看 argmax 和首个差异位置，再判断是不是 bug。

`example.py`、`bench.py` 的逻辑和项目完全相同，而项目 README 给出了它们在 GPU 上的参考成绩，可以作为对照。
