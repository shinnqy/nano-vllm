"""GPU 课程的分阶段验收入口，不是另一份推理引擎。

在 GPU 机器的项目根目录运行。--package 指向被检查的 Python 包：
  python docs/learning/gpu_acceptance.py --case environment
  python docs/learning/gpu_acceptance.py --case attention --package myvllm
  torchrun --standalone --nproc-per-node=2 docs/learning/gpu_acceptance.py --case tp --package myvllm

默认 nanovllm 用于取得原项目基线；自己的功能必须再用 --package myvllm 验收。
真实权重检查使用 --model 或 MODEL_DIR。每个 case 单独运行，避免重复进程组。
"""

import argparse
import atexit
import gc
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import tempfile
from contextlib import contextmanager
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn.functional as F

# 让从原项目根目录执行脚本时可以找到 nanovllm；myvllm 应自行 editable install。
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def module(args, suffix):
    return importlib.import_module(f"{args.package}.{suffix}")


def report(name, **values):
    print(json.dumps({"case": name, **values}, ensure_ascii=False), flush=True)


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("本验收入口需要在 CUDA GPU 机器执行")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch.device("cuda", local_rank)


@contextmanager
def process_group():
    """层级检查用 torchrun 环境；单进程检查用临时 rendezvous 文件。"""
    require_cuda()
    if dist.is_initialized():
        raise RuntimeError("每个 case 必须在独立进程中执行")
    with tempfile.TemporaryDirectory(prefix="nano-learning-") as directory:
        options = dict(backend="nccl", timeout=timedelta(minutes=5))
        if "RANK" in os.environ:
            dist.init_process_group(init_method="env://", **options)
        else:
            rendezvous = Path(directory, "rendezvous").as_uri()
            dist.init_process_group(init_method=rendezvous, rank=0, world_size=1, **options)
        try:
            yield
        finally:
            dist.destroy_process_group()


def check_environment(args):
    device = require_cuda()
    flash = importlib.import_module("flash_attn")
    assert callable(flash.flash_attn_varlen_func)
    assert callable(flash.flash_attn_with_kvcache)
    versions = {}
    for name in ("torch", "triton", "transformers", "flash-attn", "safetensors", "xxhash"):
        versions[name] = importlib.metadata.version(name)
    report("environment", device=torch.cuda.get_device_name(device),
           capability=torch.cuda.get_device_capability(device),
           cuda_runtime=torch.version.cuda, device_count=torch.cuda.device_count(),
           bf16_supported=torch.cuda.is_bf16_supported(), versions=versions)


def reference_attention(q, k, v, positions):
    """显式因果 mask 与 GQA，作为 GPU 库调用的独立数学参照。"""
    ratio = q.size(1) // k.size(1)
    qh = q.transpose(0, 1).float()
    kh = k.repeat_interleave(ratio, 1).transpose(0, 1).float()
    vh = v.repeat_interleave(ratio, 1).transpose(0, 1).float()
    scores = qh @ kh.transpose(-1, -2) * q.size(-1) ** -0.5
    allowed = torch.arange(k.size(0), device=q.device)[None, :] <= positions[:, None]
    probabilities = scores.masked_fill(~allowed[None], float("-inf")).softmax(-1)
    return (probabilities @ vh).transpose(0, 1).to(q.dtype)


@torch.inference_mode()
def check_scatter(args):
    device = require_cuda()
    store = module(args, "layers.attention").store_kvcache
    torch.manual_seed(23)
    # split 后 token 之间的 stride 大于一行 KV 的元素数，避免只验证连续输入。
    packed = torch.randn(7, 256, device=device, dtype=torch.float16)
    _, k, v = packed.split((128, 64, 64), -1)
    k, v = k.view(7, 2, 32), v.view(7, 2, 32)
    slots = torch.tensor([2, 511, -1, 0, 769, 128, 20], device=device, dtype=torch.int32)
    kc = torch.full((4, 256, 2, 32), -9.0, device=device, dtype=k.dtype)
    vc = kc.clone()
    expected_k, expected_v = kc.clone(), vc.clone()
    valid = slots >= 0
    expected_k.view(-1, 2, 32)[slots[valid].long()] = k[valid]
    expected_v.view(-1, 2, 32)[slots[valid].long()] = v[valid]
    store(k, v, kc, vc, slots)
    torch.testing.assert_close(kc, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(vc, expected_v, rtol=0, atol=0)
    report("scatter", passed=True, input_row_stride=k.stride(0), kv_row_elements=64)


@torch.inference_mode()
def check_attention(args):
    device = require_cuda()
    Attention = module(args, "layers.attention").Attention
    context = module(args, "utils.context")
    torch.manual_seed(29)
    lengths = (260, 17)
    tables_list = [[5, 1], [2, -1]]
    tables = torch.tensor(tables_list, device=device, dtype=torch.int32)
    q = torch.randn(277, 4, 32, device=device, dtype=torch.float16)
    k = torch.randn(277, 2, 32, device=device, dtype=torch.float16)
    v = torch.randn_like(k)
    slots_list = [tables_list[r][p // 256] * 256 + p % 256
                  for r, length in enumerate(lengths) for p in range(length)]
    slots = torch.tensor(slots_list, device=device, dtype=torch.int32)
    cu = torch.tensor([0, 260, 277], device=device, dtype=torch.int32)
    layer = Attention(4, 32, 32 ** -0.5, 2)
    layer.k_cache = torch.zeros(8, 256, 2, 32, device=device, dtype=k.dtype)
    layer.v_cache = torch.zeros_like(layer.k_cache)
    try:
        # 无历史的完整 Prefill：写缓存，但 Attention 直接消费新 K/V。
        context.set_context(True, cu, cu, 260, 260, slots, None, None)
        actual = layer(q, k, v)
        expected = torch.cat([reference_attention(q[:260], k[:260], v[:260], torch.arange(260, device=device)),
                              reference_attention(q[260:], k[260:], v[260:], torch.arange(17, device=device))])
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)

        # 含历史的 chunk：第一个请求跨越 256-token 边界，Q/K 长度不同。
        layer.k_cache.zero_()
        layer.v_cache.zero_()
        old_indices = list(range(253)) + list(range(260, 274))
        new_indices = torch.tensor(list(range(253, 260)) + list(range(274, 277)), device=device)
        layer.k_cache.view(-1, 2, 32)[slots[old_indices].long()] = k[old_indices]
        layer.v_cache.view(-1, 2, 32)[slots[old_indices].long()] = v[old_indices]
        cuq = torch.tensor([0, 7, 10], device=device, dtype=torch.int32)
        context.set_context(True, cuq, cu, 7, 260, slots[new_indices], None, tables)
        actual = layer(q[new_indices], k[new_indices], v[new_indices])
        expected = torch.cat([reference_attention(q[253:260], k[:260], v[:260], torch.arange(253, 260, device=device)),
                              reference_attention(q[274:], k[260:], v[260:], torch.arange(14, 17, device=device))])
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)

        # Decode 只输入末位置，传入的有效 KV 长度包含刚写入的 token。
        last = torch.tensor([259, 276], device=device)
        lens = torch.tensor(lengths, device=device, dtype=torch.int32)
        context.set_context(False, slot_mapping=slots[last], context_lens=lens, block_tables=tables)
        actual = layer(q[last], k[last], v[last]).reshape(2, 4, 32)
        expected = torch.cat([reference_attention(q[259:260], k[:260], v[:260], torch.tensor([259], device=device)),
                              reference_attention(q[276:277], k[260:], v[260:], torch.tensor([16], device=device))])
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
        report("attention", passed=True, scenarios=["prefill", "paged_chunk", "decode"],
               decode_max_abs_error=(actual.float() - expected.float()).abs().max().item())
    finally:
        context.reset_context()


@torch.inference_mode()
def check_layers(args):
    device = require_cuda()
    RMSNorm = module(args, "layers.layernorm").RMSNorm
    SiluAndMul = module(args, "layers.activation").SiluAndMul
    get_rope = module(args, "layers.rotary_embedding").get_rope
    torch.manual_seed(31)
    x = torch.randn(11, 16, device=device, dtype=torch.float16)
    residual = torch.randn_like(x)
    norm = RMSNorm(16).to(device=device, dtype=x.dtype)
    expected = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)
    torch.testing.assert_close(norm(x), expected, rtol=0.01, atol=0.01)
    total = x.float() + residual.float()
    expected = (total * torch.rsqrt(total.square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)
    actual, saved_residual = norm(x, residual)
    torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.01)
    torch.testing.assert_close(saved_residual, total.to(x.dtype), rtol=0, atol=0)
    gate_up = torch.randn(11, 64, device=device, dtype=x.dtype)
    gate, up = gate_up.chunk(2, -1)
    torch.testing.assert_close(SiluAndMul()(gate_up), F.silu(gate) * up, rtol=0.01, atol=0.01)
    rope = get_rope(32, 32, 512, 1000000).to(device)
    positions = torch.tensor([0, 1, 7, 255, 256], device=device)
    q = torch.randn(5, 4, 32, device=device, dtype=x.dtype)
    k = torch.randn(5, 2, 32, device=device, dtype=x.dtype)
    angles = positions.float()[:, None] * (1000000.0 ** (-torch.arange(0, 32, 2, device=device).float() / 32))[None]
    cosine, sine = angles.cos()[:, None], angles.sin()[:, None]
    def rotate(tensor):
        first, second = tensor.float().chunk(2, -1)
        return torch.cat((first * cosine - second * sine, second * cosine + first * sine), -1).to(tensor.dtype)
    actual_q, actual_k = rope(positions, q, k)
    torch.testing.assert_close(actual_q, rotate(q), rtol=0.01, atol=0.01)
    torch.testing.assert_close(actual_k, rotate(k), rtol=0.01, atol=0.01)
    report("layers", passed=True, covered=["RMSNorm", "add_RMSNorm", "SiluAndMul", "RoPE", "compiled_forwards"])


@torch.inference_mode()
def check_sampling(args):
    device = require_cuda()
    Sampler = module(args, "layers.sampler").Sampler
    torch.manual_seed(37)
    distribution = torch.tensor([0.2, 0.3, 0.5], device=device)
    logits = distribution.log().repeat(20000, 1)
    temperatures = torch.ones(20000, device=device)
    sampled = Sampler()(logits, temperatures)
    observed = torch.bincount(sampled, minlength=3).float() / sampled.numel()
    torch.testing.assert_close(observed, distribution, rtol=0, atol=0.02)
    temperatures = torch.cat((torch.full((20000,), 0.5, device=device),
                              torch.full((20000,), 2.0, device=device)))
    sampled = Sampler()(distribution.log().repeat(40000, 1), temperatures)
    per_temperature = []
    for group, temperature in enumerate((0.5, 2.0)):
        expected = (distribution.log() / temperature).softmax(-1)
        observed_group = torch.bincount(sampled[group * 20000:(group + 1) * 20000], minlength=3).float() / 20000
        torch.testing.assert_close(observed_group, expected, rtol=0, atol=0.02)
        per_temperature.append(dict(temperature=temperature, observed=observed_group.tolist()))
    report("sampling", passed=True, expected=distribution.tolist(), observed=observed.tolist(),
           mixed_temperatures=per_temperature)


def check_state(args):
    Sequence = module(args, "engine.sequence").Sequence
    Scheduler = module(args, "engine.scheduler").Scheduler
    SamplingParams = module(args, "sampling_params").SamplingParams
    Sequence.block_size = 256
    cfg = SimpleNamespace(max_num_seqs=4, max_num_batched_tokens=256, eos=999,
                          kvcache_block_size=256, num_kvcache_blocks=16)
    scheduler = Scheduler(cfg)
    prompt = list(range(600))
    a = Sequence(prompt, SamplingParams(max_tokens=2, ignore_eos=True))
    scheduler.add(a)
    for expected_cached, expected_scheduled in ((0, 256), (256, 256), (512, 88)):
        seqs, prefill = scheduler.schedule()
        assert prefill and seqs == [a]
        assert (a.num_cached_tokens, a.num_scheduled_tokens) == (expected_cached, expected_scheduled)
        scheduler.postprocess(seqs, [11], prefill)
    assert a.num_tokens == 601 and a.num_cached_tokens == 600
    b = Sequence(prompt[:512] + [77] * 88, SamplingParams(max_tokens=1, ignore_eos=True))
    scheduler.add(b)
    seqs, prefill = scheduler.schedule()
    assert seqs == [b] and prefill and b.num_cached_tokens == 512
    assert all(scheduler.block_manager.blocks[i].ref_count == 2 for i in a.block_table[:2])
    scheduler.postprocess(seqs, [12], prefill)
    assert all(scheduler.block_manager.blocks[i].ref_count == 1 for i in a.block_table[:2])
    seqs, prefill = scheduler.schedule()
    scheduler.postprocess(seqs, [13], prefill)
    assert scheduler.is_finished() and len(scheduler.block_manager.free_block_ids) == 16
    assert scheduler.block_manager.can_allocate(Sequence(prompt, SamplingParams())) == 2
    # 强制喂入 EOS，避免等随机采样碰巧出现 EOS 才验证停止分支。
    for ignore_eos in (False, True):
        eos_scheduler = Scheduler(cfg)
        eos_seq = Sequence([1, 2], SamplingParams(max_tokens=2, ignore_eos=ignore_eos))
        eos_scheduler.add(eos_seq)
        seqs, prefill = eos_scheduler.schedule()
        eos_scheduler.postprocess(seqs, [cfg.eos], prefill)
        assert eos_seq.is_finished == (not ignore_eos)
        if ignore_eos:
            seqs, prefill = eos_scheduler.schedule()
            eos_scheduler.postprocess(seqs, [7], prefill)
        assert eos_scheduler.is_finished()
    # 两块池、两条恰好满块的 prompt：采样后都需要新块，迫使尾请求被抢占。
    cfg.num_kvcache_blocks = 2
    cfg.max_num_batched_tokens = 512
    scheduler = Scheduler(cfg)
    a = Sequence([1] * 256, SamplingParams(max_tokens=2, ignore_eos=True))
    b = Sequence([2] * 256, SamplingParams(max_tokens=2, ignore_eos=True))
    scheduler.add(a)
    scheduler.add(b)
    seqs, prefill = scheduler.schedule()
    scheduler.postprocess(seqs, [3, 4], prefill)
    seqs, prefill = scheduler.schedule()
    assert not prefill and seqs == [a] and b in scheduler.waiting and not b.block_table
    scheduler.postprocess(seqs, [5], prefill)
    for _ in range(4):
        if scheduler.is_finished():
            break
        seqs, prefill = scheduler.schedule()
        scheduler.postprocess(seqs, [6] * len(seqs), prefill)
    assert scheduler.is_finished() and len(scheduler.block_manager.free_block_ids) == 2
    report("state", passed=True, covered=["chunk_progress", "prefix_refcounts", "idle_prefix", "EOS", "ignore_EOS", "cross_block", "preemption_recompute", "release"])


@torch.inference_mode()
def check_tp(args):
    device = require_cuda()
    with process_group():
        rank, size = dist.get_rank(), dist.get_world_size()
        assert size >= 2, "使用 torchrun --nproc-per-node=2 执行 TP 检查"
        linear = module(args, "layers.linear")
        embedding = module(args, "layers.embed_head")
        context = module(args, "utils.context")
        torch.manual_seed(41)
        # 从相同 CPU 随机序列取得参考数据，每个 rank 看到同一份完整权重。
        x = torch.randn(3, 16).to(device)
        weight = torch.randn(32, 16).to(device)
        bias = torch.randn(32).to(device)
        column = linear.ColumnParallelLinear(16, 32, bias=True).to(device)
        column.weight_loader(column.weight, weight)
        column.weight_loader(column.bias, bias)
        local = column(x)
        pieces = [torch.empty_like(local) for _ in range(size)]
        dist.all_gather(pieces, local)
        torch.testing.assert_close(torch.cat(pieces, -1), F.linear(x, weight, bias), rtol=1e-4, atol=1e-4)
        row = linear.RowParallelLinear(16, 32, bias=True).to(device)
        row.weight_loader(row.weight, weight)
        row.weight_loader(row.bias, bias)
        torch.testing.assert_close(row(x.chunk(size, -1)[rank].contiguous()), F.linear(x, weight, bias), rtol=1e-4, atol=1e-4)
        qkv = linear.QKVParallelLinear(16, 16, 8, 4).to(device)
        qweight, kweight, vweight = [torch.randn(n, 16).to(device) for n in (128, 64, 64)]
        for label, values in zip(("q", "k", "v"), (qweight, kweight, vweight)):
            qkv.weight_loader(qkv.weight, values, label)
        expected = torch.cat([F.linear(x, values.chunk(size, 0)[rank]) for values in (qweight, kweight, vweight)], -1)
        torch.testing.assert_close(qkv(x), expected, rtol=1e-4, atol=1e-4)
        merged = linear.MergedColumnParallelLinear(16, [32, 32]).to(device)
        gate, up = torch.randn(32, 16).to(device), torch.randn(32, 16).to(device)
        merged.weight_loader(merged.weight, gate, 0)
        merged.weight_loader(merged.weight, up, 1)
        expected = torch.cat([F.linear(x, values.chunk(size, 0)[rank]) for values in (gate, up)], -1)
        torch.testing.assert_close(merged(x), expected, rtol=1e-4, atol=1e-4)
        vocab_weight = torch.randn(64, 16).to(device)
        ids = torch.tensor([0, 33, 63], device=device)
        embed = embedding.VocabParallelEmbedding(64, 16).to(device)
        embed.weight_loader(embed.weight, vocab_weight)
        torch.testing.assert_close(embed(ids), F.embedding(ids, vocab_weight), rtol=0, atol=0)
        head = embedding.ParallelLMHead(64, 16).to(device)
        head.weight_loader(head.weight, vocab_weight)
        context.set_context(False)
        actual = head(x)
        if rank == 0:
            torch.testing.assert_close(actual, F.linear(x, vocab_weight), rtol=1e-4, atol=1e-4)
            report("tp", passed=True, world_size=size, covered=["column", "row_bias", "qkv", "gate_up", "embedding", "lm_head"])
        else:
            assert actual is None
        context.reset_context()


def model_path(args):
    path = args.model or os.environ.get("MODEL_DIR")
    if not path or not Path(path).is_dir():
        raise ValueError("请通过 --model 或 MODEL_DIR 指定本地 Qwen3 权重目录")
    return str(Path(path).resolve())


@torch.inference_mode()
def check_logits(args):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    device, path = require_cuda(), model_path(args)
    with process_group():
        rank = dist.get_rank()
        config = AutoConfig.from_pretrained(path)
        dtype = getattr(config, "dtype", None) or config.torch_dtype
        ids = AutoTokenizer.from_pretrained(path).encode("The capital of France is")
        ids = torch.tensor(ids, device=device, dtype=torch.int64)
        expected = None
        if rank == 0:
            reference = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="eager").to(device).eval()
            expected = reference(ids[None], use_cache=False).logits[0, -1].float().cpu()
            del reference
            gc.collect()
            torch.cuda.empty_cache()
        dist.barrier()
        Model = module(args, "models.qwen3").Qwen3ForCausalLM
        context = module(args, "utils.context")
        load_model = module(args, "utils.loader").load_model
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(dtype)
            with torch.device(device):
                model = Model(config).eval()
            load_model(model, path)
        finally:
            torch.set_default_dtype(original_dtype)
        positions = torch.arange(ids.numel(), device=device)
        cu = torch.tensor([0, ids.numel()], device=device, dtype=torch.int32)
        context.set_context(True, cu, cu, ids.numel(), ids.numel(), torch.empty(0, device=device, dtype=torch.int32))
        try:
            actual = model.compute_logits(model(ids, positions))
            if rank == 0:
                actual = actual[0].float().cpu()
                torch.testing.assert_close(actual, expected, rtol=args.rtol, atol=args.atol)
                report("logits", passed=True, world_size=dist.get_world_size(),
                       max_abs_error=(actual - expected).abs().max().item(),
                       expected_top1=expected.argmax().item(), actual_top1=actual.argmax().item())
        finally:
            context.reset_context()


@torch.inference_mode()
def check_graph(args):
    require_cuda()
    assert int(os.environ.get("WORLD_SIZE", "1")) == 1, "graph case 使用普通 python 单进程启动"
    Config = module(args, "config").Config
    Runner = module(args, "engine.model_runner").ModelRunner
    Sequence = module(args, "engine.sequence").Sequence
    BlockManager = module(args, "engine.block_manager").BlockManager
    context = module(args, "utils.context")
    config = Config(model_path(args), max_num_seqs=args.max_seqs, max_num_batched_tokens=256,
                    max_model_len=512, gpu_memory_utilization=args.memory, enforce_eager=False,
                    kvcache_block_size=args.block_size)
    Sequence.block_size = config.kvcache_block_size
    runner = Runner(config, 0, [])
    blocks = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
    try:
        sizes = sorted({n for n in (1, 3, 9, 17, 32, args.max_seqs) if n <= args.max_seqs})
        for size in sizes:
            seqs = [Sequence([1, 2, 3, 4]) for _ in range(size)]
            for seq in seqs:
                blocks.allocate(seq, 0)
                seq.num_scheduled_tokens = len(seq)
            ids, positions = runner.prepare_prefill(seqs)
            runner.run_model(ids, positions, True)
            for seq in seqs:
                seq.num_cached_tokens = len(seq)
                seq.append_token(5)
                seq.num_scheduled_tokens = 1
            ids, positions = runner.prepare_decode(seqs)
            runner.enforce_eager = True
            expected = runner.run_model(ids, positions, False).clone()
            runner.enforce_eager = False
            actual = runner.run_model(ids, positions, False)
            torch.testing.assert_close(actual, expected, rtol=args.rtol, atol=args.atol)
            for seq in seqs:
                blocks.deallocate(seq)
            context.reset_context()
            report("graph", batch_size=size, passed=True,
                   max_abs_error=(actual.float() - expected.float()).abs().max().item())
    finally:
        runner.enforce_eager = False
        context.reset_context()
        runner.exit()


@torch.inference_mode()
def reference_completion_logits(path, prompts, next_id, device):
    """先计算已知下一 token 下的参考分数，再释放模型，给引擎让出显存。"""
    from transformers import AutoConfig, AutoModelForCausalLM
    config = AutoConfig.from_pretrained(path)
    dtype = getattr(config, "dtype", None) or config.torch_dtype
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="eager").to(device).eval()
    expected = []
    for prompt in prompts:
        ids = torch.tensor(prompt + [next_id], device=device, dtype=torch.int64)
        # 前一行是 prompt 末尾的 logits；后一行是固定 next_id 经过 Decode 后的 logits。
        expected.append(model(ids[None], use_cache=False).logits[0, -2:].float().cpu())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return expected


def check_engine(args):
    device = require_cuda()
    assert int(os.environ.get("WORLD_SIZE", "1")) == 1, "LLM 自行 spawn worker，不要再用 torchrun 启动 engine case"
    path = model_path(args)
    prompt_length = 600 if args.block_size == 256 else 850
    prompt = ([17, 42, 81, 9] * ((prompt_length + 3) // 4))[:prompt_length]
    other_prompt = prompt[:512] + [23] * (prompt_length - 512)
    fixed_token = 7
    reference_logits = reference_completion_logits(path, [prompt, other_prompt], fixed_token, device)
    LLM = module(args, "llm").LLM
    SamplingParams = module(args, "sampling_params").SamplingParams
    llm = LLM(path, tensor_parallel_size=args.tp, enforce_eager=not args.graphs,
              max_model_len=1024, max_num_batched_tokens=256, max_num_seqs=16,
              gpu_memory_utilization=args.memory, kvcache_block_size=args.block_size)
    trace, active, comparisons = [], [], []
    expected_by_id = {}
    original_schedule = llm.scheduler.schedule
    def schedule():
        seqs, prefill = original_schedule()
        active[:] = seqs
        trace.append([dict(id=s.seq_id, prefill=prefill, cached=s.num_cached_tokens,
                           scheduled=s.num_scheduled_tokens, total=len(s)) for s in seqs])
        return seqs, prefill
    llm.scheduler.schedule = schedule
    original_run_model = llm.model_runner.run_model
    original_sampler = llm.model_runner.sampler
    def run_model(input_ids, positions, is_prefill):
        logits = original_run_model(input_ids, positions, is_prefill)
        for row, seq in enumerate(active):
            if seq.seq_id not in expected_by_id:
                continue
            if is_prefill and seq.num_cached_tokens + seq.num_scheduled_tokens < len(seq):
                continue
            expected = expected_by_id[seq.seq_id][0 if is_prefill else 1]
            actual = logits[row].float().cpu()
            torch.testing.assert_close(actual, expected, rtol=args.rtol, atol=args.atol)
            comparisons.append(dict(id=seq.seq_id, prefill=is_prefill,
                                    max_abs_error=(actual - expected).abs().max().item()))
        return logits
    class FixedSampler(torch.nn.Module):
        def forward(self, logits, temperatures):
            return torch.full((logits.size(0),), fixed_token, device=logits.device, dtype=torch.int64)
    # 固定生成 token 来比较缓存数值；真实随机采样另外在 sampling case 和后续 API 检查覆盖。
    llm.model_runner.run_model = run_model
    llm.model_runner.sampler = FixedSampler()
    try:
        params = SamplingParams(max_tokens=2, ignore_eos=True, temperature=0.6)
        llm.add_request(prompt, params)
        first_id = llm.scheduler.waiting[-1].seq_id
        expected_by_id[first_id] = reference_logits[0]
        assert not llm.step()[0]
        assert not llm.step()[0]  # 两轮 256-token 预算后，长 prompt 仍未完成。
        llm.add_request(other_prompt, params)  # 在第一条仍运行时插入共享前缀的请求。
        second_id = llm.scheduler.waiting[-1].seq_id
        expected_by_id[second_id] = reference_logits[1]
        outputs = {}
        for _ in range(32):
            if llm.is_finished():
                break
            completed, _ = llm.step()
            outputs.update(completed)
        assert llm.is_finished() and len(outputs) == 2
        assert all(len(ids) == 2 for ids in outputs.values())
        assert len(comparisons) == 4  # 两条请求各比较最终 Prefill 与一次 Decode。
        second_events = [event for batch in trace for event in batch if event["id"] == second_id]
        assert second_events[0]["prefill"] and second_events[0]["cached"] == 512
        assert not llm.scheduler.block_manager.used_block_ids
        llm.model_runner.run_model = original_run_model
        llm.model_runner.sampler = original_sampler
        # 另外覆盖字符串入参、tokenizer、generate 收集和实际返回类型。
        generated = llm.generate(["Name one color."], params, use_tqdm=False)
        assert len(generated) == 1 and isinstance(generated[0]["text"], str)
        assert len(generated[0]["token_ids"]) == 2
        per_request = [SamplingParams(max_tokens=n, ignore_eos=True) for n in (1, 3, 2)]
        ordered = llm.generate(["One.", "Two.", "Three."], per_request, use_tqdm=False)
        assert [len(output["token_ids"]) for output in ordered] == [1, 3, 2]
        assert not llm.scheduler.block_manager.used_block_ids
        report("engine", passed=True, tp=args.tp, graphs=args.graphs, block_size=args.block_size,
               trace=trace, logit_comparisons=comparisons,
               covered=["chunk", "dynamic_admission", "prefix_hit", "decode_logits", "length_stop", "release", "text_API", "parameter_list", "output_order"])
    finally:
        # 原项目注册 atexit；显式关闭前撤销注册，避免同一对象重复 exit。
        atexit.unregister(llm.exit)
        llm.exit()


def main():
    cases = dict(environment=check_environment, layers=check_layers, sampling=check_sampling,
                 scatter=check_scatter, attention=check_attention, state=check_state,
                 tp=check_tp, logits=check_logits, graph=check_graph, engine=check_engine)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, choices=cases)
    parser.add_argument("--package", default="nanovllm")
    parser.add_argument("--model")
    parser.add_argument("--tp", type=int, default=1, help="仅 engine case 使用；引擎自行启动 worker")
    parser.add_argument("--graphs", action="store_true", help="engine case 启用 CUDA Graph")
    parser.add_argument("--memory", type=float, default=0.6)
    parser.add_argument("--block-size", type=int, choices=(256, 512), default=256)
    parser.add_argument("--max-seqs", type=int, default=32, help="graph case 的最大 batch 档位")
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.1, help="真实低精度 logits 的初始比较阈值，需要结合误差诊断")
    args = parser.parse_args()
    if not 0 < args.memory <= 1 or args.max_seqs <= 0 or args.tp <= 0:
        parser.error("memory 必须在 (0,1]，max-seqs 与 tp 必须为正数")
    cases[args.case](args)


if __name__ == "__main__":
    main()
