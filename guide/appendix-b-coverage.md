# 附录 B：功能覆盖和完整调用链

用这份附录确认两件事：项目的每个功能都讲到了；每个方法都有来处。

## B.1 功能在哪一章

| 功能 | 解决什么问题 | 第一次出现 | 主要代码 |
|---|---|---|---|
| 同步的 `generate` 接口 | 用户怎么用 | 2 | `LLMEngine.generate` |
| 每个请求独立的采样参数 | 同一批温度和长度可以不同 | 2、3 | `SamplingParams`、`prepare_sample` |
| 采样（温度 + 指数竞争） | logits 怎么变成 token | 3 | `Sampler` |
| Qwen3 模型结构 | token 怎么变成 logits | 3、4 | `models/qwen3.py` |
| RoPE 位置编码 | 模型怎么知道顺序 | 4 | `RotaryEmbedding` |
| GQA | K、V 头比 Q 头少，省缓存 | 4、5 | `Qwen3Attention`、FlashAttention |
| 合并投影 | 省矩阵乘法 | 3、4 | `MergedColumnParallelLinear`、`QKVParallelLinear` |
| 权重加载 | 训练好的权重放进模型 | 4 | `load_model`、`weight_loader` |
| 变长序列批处理 | 不用 padding | 5 | `prepare_prefill`、`Context` |
| FlashAttention | 变长注意力、读缓存 | 5、6 | `Attention` |
| KV Cache（Prefill / Decode） | 不重复计算 | 6 | `prepare_decode`、`store_kvcache` |
| 分页 | 不用预留连续显存 | 6 | `kv_cache`、`block_table`、`slot_mapping` |
| Triton 写缓存 | 高效散写，能跳过 padding | 6 | `store_kvcache_kernel` |
| 自动估算缓存大小 | 显存用足又不 OOM | 7 | `warmup_model`、`allocate_kv_cache` |
| 动态批处理 | 请求随时加入和退出 | 8 | `Scheduler.schedule` |
| 分块预填充 | 超长 prompt | 8 | `min(num_tokens, remaining)`、`postprocess` |
| 抢占 | 缓存不够 | 8 | `Scheduler.preempt` |
| 前缀缓存 | 相同开头不重算 | 9 | `can_allocate`、`hash_blocks` |
| 引用计数和缓存淘汰 | 块被共用；释放后还能复用 | 9 | `Block`、`BlockManager` |
| `torch.compile` | 优化小算子 | 3、10 | `@torch.compile` |
| CUDA Graph | 减少 CPU 提交开销 | 10 | `capture_cudagraph`、`run_model` |
| 压测 | 衡量性能 | 10 | `bench.py` |
| 张量并行 | 多卡分摊 | 11 | `layers/linear.py`、`layers/embed_head.py` |
| 多进程派发 | 子进程怎么知道要做什么 | 11 | `call`、`loop`、`SharedMemory` |
| 高效序列化 | Decode 不传整条序列 | 11 | `__getstate__`、`__setstate__` |

## B.2 从入口出发的完整调用链

把全部章节写完后，一次 `llm.generate(...)` 里发生的所有事情：

```text
LLM(path, ...)                                     第 2、11 章
└─ LLMEngine.__init__
   ├─ Config(...)                                  读 config.json，校验
   ├─ 启动 rank 1..n-1 的子进程                     它们停在 ModelRunner.loop()
   ├─ ModelRunner(config, 0, events)
   │  ├─ dist.init_process_group
   │  ├─ Qwen3ForCausalLM(hf_config)               全部层
   │  ├─ load_model                                weight_loader 把权重放到位
   │  ├─ warmup_model                              测显存峰值
   │  ├─ allocate_kv_cache                         算块数，分配并挂到每层
   │  └─ capture_cudagraph                         逐档录制 Decode
   ├─ AutoTokenizer.from_pretrained                config.eos
   └─ Scheduler(config) → BlockManager(...)        账本

llm.generate(prompts, sampling_params)             第 2、6 章
├─ add_request × N                                 encode → Sequence → waiting
└─ while not is_finished():
   └─ step()
      ├─ ① Scheduler.schedule()                    第 8 章
      │    有 waiting：Prefill
      │      can_allocate → allocate → num_scheduled_tokens = min(...)
      │    否则：Decode
      │      can_append 不够 → preempt
      │      may_append，num_scheduled_tokens = 1
      │
      ├─ ② ModelRunner.call("run", seqs, is_prefill)  第 11 章
      │    write_shm                               通知子进程
      │    run
      │    ├─ prepare_prefill / prepare_decode     第 5、6 章
      │    │    → set_context(...)
      │    ├─ prepare_sample
      │    ├─ run_model                            第 10 章
      │    │    Prefill：直接执行
      │    │    Decode：写 graph_vars → graph.replay()
      │    │    Qwen3ForCausalLM
      │    │    └─ Qwen3Model
      │    │       ├─ VocabParallelEmbedding       all_reduce
      │    │       ├─ Qwen3DecoderLayer × L
      │    │       │  ├─ input_layernorm
      │    │       │  ├─ Qwen3Attention
      │    │       │  │  ├─ qkv_proj → split → view
      │    │       │  │  ├─ q_norm、k_norm
      │    │       │  │  ├─ rotary_emb
      │    │       │  │  ├─ Attention
      │    │       │  │  │  ├─ store_kvcache       先写
      │    │       │  │  │  └─ flash_attn_*        再读
      │    │       │  │  └─ o_proj                 all_reduce
      │    │       │  ├─ post_attention_layernorm
      │    │       │  └─ Qwen3MLP
      │    │       │     ├─ gate_up_proj
      │    │       │     ├─ SiluAndMul
      │    │       │     └─ down_proj              all_reduce
      │    │       └─ norm
      │    │    compute_logits → ParallelLMHead    gather 到 rank 0
      │    ├─ Sampler                              只在 rank 0
      │    └─ reset_context
      │
      └─ ③ Scheduler.postprocess()                 第 8、9 章
           hash_blocks → num_cached_tokens += ...
           分块中间轮：continue
           append_token → 结束：deallocate、离开 running
└─ decode → 按 seq_id 排序返回

程序退出（atexit）
└─ LLMEngine.exit → call("exit") → 释放共享内存、图、进程组 → join 子进程
```

## B.3 每个方法在哪一章

这张表是脚本生成的：把各章代码依次应用，记录每个定义第一次出现和之后变化的章节。“之后修改”只统计代码变化，不统计注释。

| 文件 | 类或方法 | 第一次写 | 之后修改 |
|---|---|---|---|
| `config.py` | `Config` | 2 | 6、7 |
| `config.py` | `Config.__post_init__` | 2 | 6、7 |
| `engine/block_manager.py` | `Block` | 9 | 不变 |
| `engine/block_manager.py` | `Block.__init__` | 9 | 不变 |
| `engine/block_manager.py` | `Block.update` | 9 | 不变 |
| `engine/block_manager.py` | `Block.reset` | 9 | 不变 |
| `engine/block_manager.py` | `BlockManager` | 6 | 8、9 |
| `engine/block_manager.py` | `BlockManager.__init__` | 6 | 9 |
| `engine/block_manager.py` | `BlockManager.compute_hash` | 9 | 不变 |
| `engine/block_manager.py` | `BlockManager._allocate_block` | 9 | 不变 |
| `engine/block_manager.py` | `BlockManager._deallocate_block` | 9 | 不变 |
| `engine/block_manager.py` | `BlockManager.can_allocate` | 8 | 9 |
| `engine/block_manager.py` | `BlockManager.allocate` | 6 | 8、9 |
| `engine/block_manager.py` | `BlockManager.deallocate` | 6 | 9 |
| `engine/block_manager.py` | `BlockManager.can_append` | 8 | 不变 |
| `engine/block_manager.py` | `BlockManager.may_append` | 6 | 9 |
| `engine/block_manager.py` | `BlockManager.hash_blocks` | 9 | 不变 |
| `engine/llm_engine.py` | `LLMEngine` | 2 | 6、11 |
| `engine/llm_engine.py` | `LLMEngine.__init__` | 2 | 6、11 |
| `engine/llm_engine.py` | `LLMEngine.exit` | 2 | 11 |
| `engine/llm_engine.py` | `LLMEngine.add_request` | 2 | 不变 |
| `engine/llm_engine.py` | `LLMEngine.step` | 2 | 6、11 |
| `engine/llm_engine.py` | `LLMEngine.is_finished` | 2 | 不变 |
| `engine/llm_engine.py` | `LLMEngine.generate` | 2 | 6 |
| `engine/model_runner.py` | `ModelRunner` | 2 | 4、5、6、7、8、10、11 |
| `engine/model_runner.py` | `ModelRunner.__init__` | 2 | 4、6、7、10、11 |
| `engine/model_runner.py` | `ModelRunner.exit` | 2 | 4、10、11 |
| `engine/model_runner.py` | `ModelRunner.loop` | 11 | 不变 |
| `engine/model_runner.py` | `ModelRunner.read_shm` | 11 | 不变 |
| `engine/model_runner.py` | `ModelRunner.write_shm` | 11 | 不变 |
| `engine/model_runner.py` | `ModelRunner.call` | 11 | 不变 |
| `engine/model_runner.py` | `ModelRunner.warmup_model` | 7 | 不变 |
| `engine/model_runner.py` | `ModelRunner.allocate_kv_cache` | 6 | 7 |
| `engine/model_runner.py` | `ModelRunner.prepare_block_tables` | 6 | 不变 |
| `engine/model_runner.py` | `ModelRunner.prepare_prefill` | 5 | 6、7、8 |
| `engine/model_runner.py` | `ModelRunner.prepare_decode` | 6 | 不变 |
| `engine/model_runner.py` | `ModelRunner.prepare_sample` | 5 | 不变 |
| `engine/model_runner.py` | `ModelRunner.run_model` | 5 | 6、10 |
| `engine/model_runner.py` | `ModelRunner.run` | 2 | 4、5、6、11 |
| `engine/model_runner.py` | `ModelRunner.capture_cudagraph` | 10 | 不变 |
| `engine/scheduler.py` | `Scheduler` | 2 | 6、8、9、11 |
| `engine/scheduler.py` | `Scheduler.__init__` | 2 | 6、8 |
| `engine/scheduler.py` | `Scheduler.is_finished` | 2 | 不变 |
| `engine/scheduler.py` | `Scheduler.add` | 2 | 不变 |
| `engine/scheduler.py` | `Scheduler.schedule` | 2 | 6、8、11 |
| `engine/scheduler.py` | `Scheduler.preempt` | 8 | 11 |
| `engine/scheduler.py` | `Scheduler.postprocess` | 2 | 6、8、9 |
| `engine/sequence.py` | `SequenceStatus` | 2 | 不变 |
| `engine/sequence.py` | `Sequence` | 2 | 6、7、9、11 |
| `engine/sequence.py` | `Sequence.__init__` | 2 | 6、7、11 |
| `engine/sequence.py` | `Sequence.__len__` | 2 | 不变 |
| `engine/sequence.py` | `Sequence.__getitem__` | 2 | 不变 |
| `engine/sequence.py` | `Sequence.is_finished` | 2 | 不变 |
| `engine/sequence.py` | `Sequence.num_completion_tokens` | 2 | 不变 |
| `engine/sequence.py` | `Sequence.prompt_token_ids` | 2 | 不变 |
| `engine/sequence.py` | `Sequence.completion_token_ids` | 2 | 不变 |
| `engine/sequence.py` | `Sequence.num_blocks` | 6 | 不变 |
| `engine/sequence.py` | `Sequence.last_block_num_tokens` | 6 | 不变 |
| `engine/sequence.py` | `Sequence.block` | 9 | 不变 |
| `engine/sequence.py` | `Sequence.append_token` | 2 | 不变 |
| `engine/sequence.py` | `Sequence.__getstate__` | 11 | 不变 |
| `engine/sequence.py` | `Sequence.__setstate__` | 11 | 不变 |
| `layers/activation.py` | `SiluAndMul` | 3 | 不变 |
| `layers/activation.py` | `SiluAndMul.forward` | 3 | 不变 |
| `layers/attention.py` | `store_kvcache_kernel` | 6 | 不变 |
| `layers/attention.py` | `store_kvcache` | 6 | 不变 |
| `layers/attention.py` | `Attention` | 4 | 5、6、8 |
| `layers/attention.py` | `Attention.__init__` | 4 | 6 |
| `layers/attention.py` | `Attention.forward` | 4 | 5、6、8 |
| `layers/embed_head.py` | `VocabParallelEmbedding` | 3 | 11 |
| `layers/embed_head.py` | `VocabParallelEmbedding.__init__` | 3 | 11 |
| `layers/embed_head.py` | `VocabParallelEmbedding.weight_loader` | 11 | 不变 |
| `layers/embed_head.py` | `VocabParallelEmbedding.forward` | 3 | 11 |
| `layers/embed_head.py` | `ParallelLMHead` | 3 | 5、6、11 |
| `layers/embed_head.py` | `ParallelLMHead.__init__` | 3 | 不变 |
| `layers/embed_head.py` | `ParallelLMHead.forward` | 3 | 5、6、11 |
| `layers/layernorm.py` | `RMSNorm` | 3 | 不变 |
| `layers/layernorm.py` | `RMSNorm.__init__` | 3 | 不变 |
| `layers/layernorm.py` | `RMSNorm.rms_forward` | 3 | 不变 |
| `layers/layernorm.py` | `RMSNorm.add_rms_forward` | 3 | 不变 |
| `layers/layernorm.py` | `RMSNorm.forward` | 3 | 不变 |
| `layers/linear.py` | `divide` | 11 | 不变 |
| `layers/linear.py` | `LinearBase` | 3 | 11 |
| `layers/linear.py` | `LinearBase.__init__` | 3 | 11 |
| `layers/linear.py` | `LinearBase.forward` | 3 | 11 |
| `layers/linear.py` | `ReplicatedLinear` | 11 | 不变 |
| `layers/linear.py` | `ReplicatedLinear.__init__` | 11 | 不变 |
| `layers/linear.py` | `ReplicatedLinear.weight_loader` | 11 | 不变 |
| `layers/linear.py` | `ReplicatedLinear.forward` | 11 | 不变 |
| `layers/linear.py` | `ColumnParallelLinear` | 11 | 不变 |
| `layers/linear.py` | `ColumnParallelLinear.__init__` | 11 | 不变 |
| `layers/linear.py` | `ColumnParallelLinear.weight_loader` | 11 | 不变 |
| `layers/linear.py` | `ColumnParallelLinear.forward` | 11 | 不变 |
| `layers/linear.py` | `MergedColumnParallelLinear` | 3 | 11 |
| `layers/linear.py` | `MergedColumnParallelLinear.__init__` | 3 | 不变 |
| `layers/linear.py` | `MergedColumnParallelLinear.weight_loader` | 3 | 11 |
| `layers/linear.py` | `QKVParallelLinear` | 4 | 11 |
| `layers/linear.py` | `QKVParallelLinear.__init__` | 4 | 11 |
| `layers/linear.py` | `QKVParallelLinear.weight_loader` | 4 | 11 |
| `layers/linear.py` | `RowParallelLinear` | 3 | 11 |
| `layers/linear.py` | `RowParallelLinear.__init__` | 11 | 不变 |
| `layers/linear.py` | `RowParallelLinear.weight_loader` | 11 | 不变 |
| `layers/linear.py` | `RowParallelLinear.forward` | 11 | 不变 |
| `layers/rotary_embedding.py` | `apply_rotary_emb` | 4 | 不变 |
| `layers/rotary_embedding.py` | `RotaryEmbedding` | 4 | 不变 |
| `layers/rotary_embedding.py` | `RotaryEmbedding.__init__` | 4 | 不变 |
| `layers/rotary_embedding.py` | `RotaryEmbedding.forward` | 4 | 不变 |
| `layers/rotary_embedding.py` | `get_rope` | 4 | 不变 |
| `layers/sampler.py` | `Sampler` | 3 | 不变 |
| `layers/sampler.py` | `Sampler.forward` | 3 | 不变 |
| `llm.py` | `LLM` | 2 | 不变 |
| `models/qwen3.py` | `Qwen3Attention` | 4 | 不变 |
| `models/qwen3.py` | `Qwen3Attention.__init__` | 4 | 不变 |
| `models/qwen3.py` | `Qwen3Attention.forward` | 4 | 不变 |
| `models/qwen3.py` | `Qwen3MLP` | 3 | 不变 |
| `models/qwen3.py` | `Qwen3MLP.__init__` | 3 | 不变 |
| `models/qwen3.py` | `Qwen3MLP.forward` | 3 | 不变 |
| `models/qwen3.py` | `Qwen3DecoderLayer` | 3 | 不变 |
| `models/qwen3.py` | `Qwen3DecoderLayer.__init__` | 3 | 不变 |
| `models/qwen3.py` | `Qwen3DecoderLayer.forward` | 3 | 不变 |
| `models/qwen3.py` | `Qwen3Model` | 3 | 不变 |
| `models/qwen3.py` | `Qwen3Model.__init__` | 3 | 不变 |
| `models/qwen3.py` | `Qwen3Model.forward` | 3 | 不变 |
| `models/qwen3.py` | `Qwen3ForCausalLM` | 3 | 不变 |
| `models/qwen3.py` | `Qwen3ForCausalLM.__init__` | 3 | 不变 |
| `models/qwen3.py` | `Qwen3ForCausalLM.forward` | 3 | 不变 |
| `models/qwen3.py` | `Qwen3ForCausalLM.compute_logits` | 3 | 不变 |
| `sampling_params.py` | `SamplingParams` | 2 | 不变 |
| `sampling_params.py` | `SamplingParams.__post_init__` | 2 | 不变 |
| `utils/context.py` | `Context` | 5 | 6 |
| `utils/context.py` | `get_context` | 5 | 不变 |
| `utils/context.py` | `set_context` | 5 | 6 |
| `utils/context.py` | `reset_context` | 5 | 不变 |
| `utils/loader.py` | `default_weight_loader` | 4 | 不变 |
| `utils/loader.py` | `load_model` | 4 | 不变 |

项目里的每个类和方法都在这张表里。第 11 章结束时，定义的集合和项目源码完全相同。
