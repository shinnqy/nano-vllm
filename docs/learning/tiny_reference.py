"""逐行学习用的单请求 Qwen3 风格参考模型。

运行：python docs/learning/tiny_reference.py --device cuda
只需要 PyTorch；默认使用 CUDA、FP32 和随机权重，不下载模型。
它用于理解计算和验证缓存等价性，随机权重没有语言能力。
这是完整 GPU 引擎课程的数学参考阶段；最终功能验收另见 gpu_acceptance.py。
配套讲解：docs/from-zero-inference-engine.zh-CN.md
"""

from dataclasses import dataclass  # 自动生成配置对象的初始化方法。
import argparse  # 明确选择执行设备，GPU 课程默认使用 CUDA。

import torch  # 张量、矩阵乘法和设备操作来自 PyTorch。
from torch import nn  # Module 注册子模块，Parameter 注册可加载的权重。
import torch.nn.functional as F  # 使用没有独立状态的函数式算子。


@dataclass
class TinyConfig:
    vocab_size: int = 32  # 词表有多少种 token；这里使用玩具整数 ID。
    hidden_size: int = 24  # 残差通路中每个 token 的向量宽度。
    num_heads: int = 4  # Query 头数。
    num_kv_heads: int = 2  # K/V 头数；两个 Query 头共享一组 K/V。
    head_dim: int = 8  # 刻意让 4 * 8 != 24，避免误认为两者必须相等。
    intermediate_size: int = 48  # MLP 内部先扩展到这个宽度。
    num_layers: int = 2  # 两层足以检验每层独立的 KV Cache。
    rms_norm_eps: float = 1e-6  # 防止归一化分母为零。
    rope_theta: float = 10000.0  # 控制 RoPE 不同维度的旋转频率。
    tie_word_embeddings: bool = True  # 输入词向量和输出投影共享权重。


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()  # 初始化 Module 的参数和子模块登记机制。
        self.weight = nn.Parameter(torch.ones(width))  # 每个特征一个缩放系数。
        self.eps = eps  # eps 是超参数，不是待学习的权重。

    def forward(self, x):
        original_dtype = x.dtype  # 输出应回到调用者使用的数据类型。
        xf = x.float()  # 平方和均值用 FP32，降低低精度累积误差。
        mean_square = xf.square().mean(dim=-1, keepdim=True)  # 最后一维归一化。
        normalized = xf * torch.rsqrt(mean_square + self.eps)  # 除以均方根。
        return normalized.to(original_dtype) * self.weight  # 再施加训练好的缩放。


class RoPE(nn.Module):
    def __init__(self, head_dim, theta):
        super().__init__()
        assert head_dim % 2 == 0  # 旋转需要把特征两两配对。
        index = torch.arange(0, head_dim, 2, dtype=torch.float32)  # d/2 个频率。
        inv_freq = theta ** (-index / head_dim)  # 高频到低频的逆波长。
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # 随模型移动设备。

    def forward(self, x, positions):
        angles = positions.float()[:, None] * self.inv_freq[None, :]  # [T, d/2]。
        cos = angles.cos()[:, None, :]  # [T, 1, d/2]，沿 head 维广播。
        sin = angles.sin()[:, None, :]  # 同一个位置的各个 head 使用相同角度。
        first, second = x.float().chunk(2, dim=-1)  # 使用前后半区配对的排列。
        rotated = torch.cat((first * cos - second * sin,
                             second * cos + first * sin), dim=-1)  # 二维旋转。
        return rotated.to(x.dtype)  # 保持 Q/K 原来的 dtype。


def reference_attention(q, k, v, query_positions):
    """一个请求；q=[Tq,Hq,d]，k/v=[Tk,Hkv,d]，KV 从逻辑位置 0 连续开始。"""
    assert q.size(1) % k.size(1) == 0  # GQA 的头数比例必须是整数。
    repeats = q.size(1) // k.size(1)  # 每个 KV head 服务多少个 Query head。
    k_for_q = k.repeat_interleave(repeats, dim=1)  # 教学版显式扩展 KV 头。
    v_for_q = v.repeat_interleave(repeats, dim=1)  # 生产内核可避免这次复制。
    qh = q.transpose(0, 1)  # [Hq,Tq,d]，让每个 head 独立做矩阵乘法。
    kh = k_for_q.transpose(0, 1)  # [Hq,Tk,d]。
    vh = v_for_q.transpose(0, 1)  # [Hq,Tk,d]。
    scores = (qh.float() @ kh.float().transpose(-1, -2)) * q.size(-1) ** -0.5
    key_positions = torch.arange(k.size(0), device=q.device)  # 历史的逻辑位置。
    allowed = key_positions[None, :] <= query_positions[:, None]  # 不能读取未来。
    scores = scores.masked_fill(~allowed[None, :, :], float("-inf"))  # 广播到各头。
    probabilities = scores.softmax(dim=-1)  # 每个 Query 对可见 Key 的权重和为 1。
    output = probabilities @ vh.float()  # 对 Value 按注意力权重求和。
    return output.transpose(0, 1).to(q.dtype)  # 回到 [Tq,Hq,d]。


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # 保存拆头所需的形状信息。
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)  # 每个 Q head 内归一化。
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)  # 每个 K head 内归一化。
        self.rope = RoPE(config.head_dim, config.rope_theta)  # Q/K 共享位置编码规则。
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_size, bias=False)

    def forward(self, x, positions, past=None):
        c = self.config  # 简写配置名，仅为缩短后面的表达式。
        q = self.q_proj(x).view(-1, c.num_heads, c.head_dim)  # [T,Hq,d]。
        k = self.k_proj(x).view(-1, c.num_kv_heads, c.head_dim)  # [T,Hkv,d]。
        v = self.v_proj(x).view(-1, c.num_kv_heads, c.head_dim)  # V 不做 Q/K Norm。
        q = self.rope(self.q_norm(q), positions)  # 先 Q Norm，再施加逻辑位置。
        k = self.rope(self.k_norm(k), positions)  # 缓存的是已经施加 RoPE 的 K。
        if past is not None:  # 后续调用只传来新 token，历史从缓存取得。
            k = torch.cat((past[0], k), dim=0)  # 教学版用 cat；它会重新分配并复制。
            v = torch.cat((past[1], v), dim=0)  # 同一层的历史 Value 也必须保留。
        output = reference_attention(q, k, v, positions)  # 新 Q 读取全部可见 KV。
        output = self.o_proj(output.reshape(x.size(0), -1))  # 合并头并返回残差宽度。
        return output, (k, v)  # 把本层更新后的缓存交给上层管理。


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))  # 门控分支先通过 SiLU 非线性。
        up = self.up_proj(x)  # 另一分支提供被门控的特征。
        return self.down_proj(gate * up)  # 逐元素乘，再压回 hidden_size。


class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)  # 每层都有独立的投影权重。
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)  # 每层也有独立的 MLP。

    def forward(self, x, positions, past=None):
        attn_output, present = self.self_attn(self.input_layernorm(x), positions, past)
        x = x + attn_output  # 第一条残差：保留输入，再加注意力提取的更新。
        x = x + self.mlp(self.post_attention_layernorm(x))  # 第二条残差。
        return x, present  # 只传回本层的 KV，不混入其他层。


class Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, input_ids, positions, caches=None):
        x = self.embed_tokens(input_ids)  # 整数 ID -> [T,hidden_size] 浮点向量。
        updated_caches = []  # 新列表按层号收集缓存。
        for layer_index, layer in enumerate(self.layers):
            past = None if caches is None else caches[layer_index]  # 取这一层的历史。
            x, present = layer(x, positions, past)  # 顺序经过每一层。
            updated_caches.append(present)  # 每层 K/V 都要保存。
        return self.norm(x), updated_caches  # 最后一层后再做一次 RMSNorm。


class TinyLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Backbone(config)  # 名称接近 Hugging Face 的 checkpoint 路径。
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight  # 共享同一个 Parameter。

    def forward(self, input_ids, positions=None, caches=None):
        if positions is None:  # 显式传 positions 时，调用者负责给出正确逻辑位置。
            start = 0 if caches is None else caches[0][0].size(0)  # 已缓存的 token 数。
            positions = torch.arange(start, start + input_ids.numel(), device=input_ids.device)
        hidden_states, updated_caches = self.model(input_ids, positions, caches)
        logits = self.lm_head(hidden_states)  # 每个输入位置预测它后面的 token。
        return logits, updated_caches  # 返回所有位置，便于逐位置比较正确性。


@torch.inference_mode()
def greedy_generate(model, prompt_ids, max_new_tokens, use_cache=True, eos_id=None):
    assert prompt_ids.ndim == 1 and prompt_ids.numel() > 0  # 此版只处理一个非空请求。
    assert max_new_tokens >= 0  # 生成长度用 token 数计。
    tokens = prompt_ids.clone()  # 不改变调用者提供的输入张量。
    current_input = tokens  # 第一轮把整个 prompt 输入模型。
    caches = None  # 第一次还没有任何层的历史 KV。
    for _ in range(max_new_tokens):
        logits, present = model(current_input, caches=caches)  # 推进一步模型计算。
        next_id = logits[-1].argmax().reshape(1)  # 最后一个位置预测下一个 token。
        tokens = torch.cat((tokens, next_id))  # 输出 token 已有 ID，但还没有自己的 KV。
        if eos_id is not None and next_id.item() == eos_id:
            break  # 这一版保留 EOS 的 ID；展示文字时可跳过特殊 token。
        if use_cache:
            caches = present  # 保留刚才算出的各层 KV。
            current_input = next_id  # 下一轮只为刚生成的 token 做前向。
        else:
            caches = None  # 无缓存基线每轮都从头算。
            current_input = tokens  # 将 prompt 和全部输出重新输入。
    return tokens[prompt_ids.numel():]  # 返回新生成部分，不包含 prompt。


def gather_paged(pool, block_table, context_length, block_size):
    positions = torch.arange(context_length, device=pool.device)  # 逻辑 token 位置。
    table = torch.tensor(block_table, device=pool.device, dtype=torch.long)  # 逻辑块 -> 物理块。
    physical_blocks = table[positions // block_size]  # 找到每个 token 所在的物理块。
    offsets = positions % block_size  # 找到块内的位置。
    return pool[physical_blocks, offsets]  # 按逻辑顺序取回 KV；教学版会产生拷贝。


@torch.inference_mode()
def run_experiments(device="cuda"):
    device = torch.device(device)  # 后面的模型与输入必须放在同一个设备。
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("请在 GPU 机器运行；本机数学检查可显式使用 --device cpu")
        torch.backends.cuda.matmul.allow_tf32 = False  # FP32 参考阶段避免 TF32 影响误差对照。
    torch.manual_seed(7)  # 随机权重固定，方便自己修改代码后比较。
    model = TinyLM(TinyConfig()).to(device).eval()  # eval 和 inference_mode 各自有不同职责。
    ids = torch.tensor([1, 5, 9, 2, 6, 3], dtype=torch.long, device=device)  # 玩具 prompt。
    full_logits, _ = model(ids)  # 整段无历史缓存的因果前向，作为参考答案。

    caches = None
    single_logits = []
    for token in ids:  # 同一串已知输入，每次只计算一个 token。
        logits, caches = model(token.reshape(1), caches=caches)
        single_logits.append(logits)
    incremental = torch.cat(single_logits, dim=0)  # 恢复原有的 token 顺序。
    torch.testing.assert_close(incremental, full_logits, rtol=1e-5, atol=1e-5)
    print("逐 token KV 与完整前向一致；最大误差", (incremental - full_logits).abs().max().item())

    caches = None
    chunk_logits = []
    offset = 0
    for chunk_size in (2, 3, 1):  # 测试非零位置起点，以及长短不同的 Prefill 分块。
        logits, caches = model(ids[offset:offset + chunk_size], caches=caches)
        chunk_logits.append(logits)
        offset += chunk_size
    torch.testing.assert_close(torch.cat(chunk_logits), full_logits, rtol=1e-5, atol=1e-5)
    print("分块 Prefill 与完整前向一致")

    generated_cached = greedy_generate(model, ids, 4, use_cache=True)
    generated_full = greedy_generate(model, ids, 4, use_cache=False)
    assert torch.equal(generated_cached, generated_full)  # greedy 让生成路径便于比较。
    print("有缓存和无缓存生成的 token 一致", generated_cached.tolist())

    block_size = 4  # 这是数学地址练习；项目的 FlashAttention 路径要求 256 的倍数。
    table = [3, 1]  # 故意让物理块顺序不同于逻辑顺序。
    values = torch.arange(6 * 2 * 3, dtype=torch.float32, device=device).view(6, 2, 3)
    pool = torch.full((5, block_size, 2, 3), -1.0, device=device)  # 分页池与输入在相同设备。
    for position, value in enumerate(values):
        pool[table[position // block_size], position % block_size] = value
    restored = gather_paged(pool, table, len(values), block_size)
    torch.testing.assert_close(restored, values, rtol=0, atol=0)
    slots = [table[p // block_size] * block_size + p % block_size for p in range(len(values))]
    print("分页 KV 按逻辑顺序读回一致；slot", slots)  # 物理地址可以乱序，逻辑结果保持有序。


if __name__ == "__main__":  # import 时只提供定义，直接运行时才启动实验。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda", help="默认 cuda，可指定 cuda:0")
    run_experiments(parser.parse_args().device)
