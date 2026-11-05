import torch
from torch import nn
import triton
import triton.language as tl

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.utils.context import get_context


@triton.jit # 表示这个函数会由 Triton 编译成 GPU 执行的代码。kernel 使用 tl.load()、`tl.store()` 直接访问数据
def store_kvcache_kernel( # [增量 KV Cache] 实际写入
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    """
    这个 Triton kernel 完成的是 KV 的散写
    现在要做的是：把“本轮刚计算出来的 K/V”，真正写入“后续生成还要使用的缓存池”。
    """

    # 1. 每个 Triton program 处理本轮打包输入中的一个 token
    idx = tl.program_id(0)
    # 2. 查出这个输入 token 对应的物理槽位
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    # 3. 读取这一行 K 的所有 KV heads、所有 head 维度。
    key_offsets = idx * key_stride + tl.arange(0, D) # 这里使用真实的 key_stride，因为从合并 QKV 投影中拆出来的张量，其相邻 token 行之间可能存在跨度。
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    # 4. 把一个物理 token 槽位展开成 D 个元素的位置，然后把整行 K/V 写进去
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    """新 K/V 的实际写入发生这里"""

    # N = 本轮新计算的 token 总数
    N, num_heads, head_dim = key.shape
    # D = 一个 token 的全部 KV heads 展平后的元素数
    D = num_heads * head_dim # 这里的 D 是 Hkv * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    """
    启动一次 kernel，使用一个包含 N 个 Triton program 的一维执行网格
    [(N,)] 是 Triton 的启动网格写法：启动 N 个 program 实例。这里每个实例负责一个 token。

    在这份实现中：
        program 0 → 负责输入第 0 个 token
        program 1 → 负责输入第 1 个 token
        program 2 → 负责输入第 2 个 token
    请注意：
        N 个 program，不等于 CPU 启动 N 次 kernel。CPU 这里提交的是一次 kernel 启动。GPU 根据硬件资源调度里面的这些 program，也不要求它们全部同时驻留执行
    """
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor): # [Prefill / Decode 分阶段执行] 完整 Prefill 结束时也会采样第一个输出 token，后续才进入逐 token Decode # [增量 KV Cache] 1. 将本轮新产生的 K、V 写入缓存。2. Decode 时把 k_cache、v_cache、有效上下文长度和块表传给 FlashAttention。3. 新 token 的 Query 可以访问此前保存的 KV。
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        """先写入本轮新产生的 KV"""
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        """
        再执行本轮 Attention
        (这样，新 Query 可以访问自己当前位置的 K/V，以及前面的历史 K/V)
        """
        """"IMPORTANT: Attention 的点积、Softmax、加权求和在 FlashAttention 调用中完成"""
        if context.is_prefill: # [Prefill / Decode 分阶段执行] Prefill 调用 flash_attn_varlen_func()，Decode 调用 flash_attn_with_kvcache()
            if context.block_tables is not None:    # prefix cache # [分块预填充] 通过块表读取此前已经计算出的 KV
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(q, k, v,                           # 本轮注意力需要处理的数据
                                       max_seqlen_q=context.max_seqlen_q, # 本批请求中最大的序列长度
                                       cu_seqlens_q=context.cu_seqlens_q, # 拼在一起的 token 分别属于哪些请求，例如 cu_seqlens_q = [0, 3, 8]，表示第一条请求有 3 个 Query token，第二条有 5 个，它们在一个张量中连续排列。
                                       max_seqlen_k=context.max_seqlen_k, # 本批请求中最大的序列长度
                                       cu_seqlens_k=context.cu_seqlens_k, # 拼在一起的 token 分别属于哪些请求
                                       softmax_scale=self.scale,          # 注意力分数的缩放系数
                                       causal=True,                       # 限制 token 只能使用允许的历史位置
                                       block_table=context.block_tables)  # 使用分页缓存时，去哪些物理块读取 K/V
        else:    # decode
            o = flash_attn_with_kvcache(q.unsqueeze(1),                     # `unsqueeze(1)` 补出长度为 1 的序列维度
                                        k_cache, v_cache,
                                        cache_seqlens=context.context_lens, # `context_lens` 告诉算子去读取多长的历史。
                                        block_table=context.block_tables,   # block_table 告诉算子历史在哪里。
                                        softmax_scale=self.scale,
                                        causal=True)
        return o
