"""第一课：亲手定义一个完整的因果 Decoder 模型，再调用它生成 token。

在 GPU 机器运行：python docs/learning/step01_model_and_generate.py
只依赖 PyTorch；本文件定义全部模型类，不导入其他教学模型或 nano-vllm。
本课使用随机参数，先学会构造和执行模型；真实权重加载接在指南第 7 章。
"""

# 1 导入工具并声明模型配置
from dataclasses import dataclass  # 根据字段自动生成配置对象的 __init__。
import torch  # 张量与 GPU 计算。
from torch import nn  # Module、Embedding、Linear、Parameter。
import torch.nn.functional as F  # 无独立参数的函数式算子，例如 SiLU。


@dataclass
class ModelConfig:
    vocab_size: int = 32  # 输入 ID 范围是 0 到 31，每行输出也有 32 个候选分数。
    hidden_size: int = 24  # 每个 token 在残差通路中的特征数。
    num_heads: int = 4  # Query 头数。
    num_kv_heads: int = 2  # 两个 Query head 共用一个 KV head。
    head_dim: int = 8  # 每个 head 的宽度；它不必等于 hidden_size / num_heads。
    intermediate_size: int = 48  # MLP 内部的特征数。
    num_layers: int = 2  # 叠加两个完整的 Decoder Layer。
    rms_norm_eps: float = 1e-6  # 归一化分母的保护项。
    rope_theta: float = 10000.0  # RoPE 频率参数。
    tie_word_embeddings: bool = True  # Embedding 和 LM Head 共用训练参数。


# 2 定义 RMSNorm
class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()  # 初始化 Module 对子模块和参数的管理。
        self.weight = nn.Parameter(torch.ones(width))  # 每个特征一个可加载的缩放参数。
        self.eps = eps  # 保存数值保护项。

    def forward(self, x):
        xf = x.float()  # 在 FP32 下计算均方根。
        mean_square = xf.square().mean(dim=-1, keepdim=True)  # 每个 token 独立统计。
        normalized = xf * torch.rsqrt(mean_square + self.eps)  # 除以均方根。
        return normalized.to(x.dtype) * self.weight  # 恢复 dtype，再施加缩放。


# 3 定义旋转位置编码
class RoPE(nn.Module):
    def __init__(self, head_dim, theta):
        super().__init__()
        assert head_dim % 2 == 0  # 每次旋转需要两个对应特征。
        index = torch.arange(0, head_dim, 2, dtype=torch.float32)  # d/2 个频率索引。
        inv_freq = theta ** (-index / head_dim)  # 每个特征对使用的旋转频率。
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # 随模型迁移到 GPU。

    def forward(self, x, positions):
        angles = positions.float()[:, None] * self.inv_freq[None, :]  # [T,d/2]。
        cosine = angles.cos()[:, None, :]  # 加入 head 广播维，形状 [T,1,d/2]。
        sine = angles.sin()[:, None, :]  # 同一位置的各个 head 使用相同角度。
        first, second = x.float().chunk(2, dim=-1)  # 使用前半区与后半区配对。
        rotated = torch.cat((first * cosine - second * sine,
                             second * cosine + first * sine), dim=-1)
        return rotated.to(x.dtype)  # 保持 Q/K 的 dtype。


# 4 定义注意力的数学计算
def reference_attention(q, k, v, positions):
    assert q.size(1) % k.size(1) == 0  # GQA 的头数比例必须是整数。
    repeats = q.size(1) // k.size(1)  # 每个 KV head 对应多少个 Query head。
    k = k.repeat_interleave(repeats, dim=1)  # 第一版显式展开 KV 头，便于理解。
    v = v.repeat_interleave(repeats, dim=1)  # 同样展开 V，保持头的对应关系。
    qh = q.transpose(0, 1).float()  # [Hq,T,d]，每个 head 单独进行矩阵运算。
    kh = k.transpose(0, 1).float()  # [Hq,T,d]。
    vh = v.transpose(0, 1).float()  # [Hq,T,d]。
    scores = (qh @ kh.transpose(-1, -2)) * q.size(-1) ** -0.5  # [Hq,T,T]。
    allowed = positions[None, :] <= positions[:, None]  # Key 位置不能晚于 Query。
    scores = scores.masked_fill(~allowed[None, :, :], float("-inf"))  # 屏蔽未来。
    probabilities = scores.softmax(dim=-1)  # 在可见 Key 之间分配概率权重。
    output = probabilities @ vh  # 对 Value 做加权和，得到 [Hq,T,d]。
    return output.transpose(0, 1).to(q.dtype)  # 恢复 [T,Hq,d]。


# 5 定义包含投影和位置编码的 Attention 层
class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # forward 拆头时还要使用这些维度。
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)  # 在每个 Q head 内归一化。
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)  # 在每个 K head 内归一化。
        self.rotary_emb = RoPE(config.head_dim, config.rope_theta)  # 现在 RoPE 类已经定义。
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_size, bias=False)

    def forward(self, x, positions):
        config = self.config  # 这里只是给已有配置取一个局部名字。
        q = self.q_proj(x).view(-1, config.num_heads, config.head_dim)  # [T,Hq,d]。
        k = self.k_proj(x).view(-1, config.num_kv_heads, config.head_dim)  # [T,Hkv,d]。
        v = self.v_proj(x).view(-1, config.num_kv_heads, config.head_dim)  # [T,Hkv,d]。
        q = self.rotary_emb(self.q_norm(q), positions)  # Q Norm 后加入位置信息。
        k = self.rotary_emb(self.k_norm(k), positions)  # K 同样处理；V 保持原投影。
        output = reference_attention(q, k, v, positions)  # 调用上一段亲手定义的函数。
        output = output.reshape(x.size(0), -1)  # 合并 head，形状 [T,Hq*d]。
        return self.o_proj(output)  # 投影回 [T,hidden_size]，方便残差相加。


# 6 定义 MLP
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        gate = F.silu(self.gate_proj(x))  # 门控分支提供非线性。
        up = self.up_proj(x)  # 特征分支映射到相同的中间宽度。
        return self.down_proj(gate * up)  # 逐元素门控，再映射回残差宽度。


# 7 将 Attention 和 MLP 组装为一层 Decoder
class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Attention(config)  # 使用第 5 段已经定义的 Attention。
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = MLP(config)  # 使用第 6 段已经定义的 MLP。

    def forward(self, x, positions):
        x = x + self.self_attn(self.input_layernorm(x), positions)  # 第一条残差。
        x = x + self.mlp(self.post_attention_layernorm(x))  # 第二条残差。
        return x  # 形状仍然是 [T,hidden_size]。


# 8 组装 Embedding 多层 Decoder 和最终归一化
class Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)  # ID 查表。
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)  # 模型末尾归一化。

    def forward(self, input_ids, positions):
        x = self.embed_tokens(input_ids)  # [T] 整数 -> [T,hidden_size] 浮点向量。
        for layer in self.layers:
            x = layer(x, positions)  # 顺序经过每一个 Decoder Layer。
        return self.norm(x)  # 返回隐藏状态，尚未转换成词表分数。


# 9 定义随后用于创建 model 对象的类
class CausalLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.model = Backbone(config)  # 完整的 Transformer 主体。
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight  # 共享同一个参数对象。

    def forward(self, input_ids):
        positions = torch.arange(input_ids.numel(), device=input_ids.device)  # [0,...,T-1]。
        hidden_states = self.model(input_ids, positions)  # [T,hidden_size]。
        logits = self.lm_head(hidden_states)  # [T,vocab_size]，每行预测下一 token。
        return logits  # 本课统一只返回一个 Tensor，尚未引入 KV Cache。


# 10 定义生成循环，model 明确作为参数传入
@torch.inference_mode()
def greedy_generate(model, prompt_ids, max_new_tokens, eos_id=None):
    assert prompt_ids.ndim == 1 and prompt_ids.numel() > 0  # 本课只处理一个非空请求。
    assert max_new_tokens >= 0  # 新增 token 数不能为负数。
    tokens = prompt_ids.clone()  # 保留调用者原来的 prompt，不原地修改。
    for _ in range(max_new_tokens):
        logits = model(tokens)  # 调用下面 main 创建的 CausalLM 对象。
        next_id = logits[-1].argmax().reshape(1)  # 取末位置分数最大的候选，变成 [1]。
        tokens = torch.cat((tokens, next_id), dim=0)  # 追加到输入，下一轮重算全部历史。
        if eos_id is not None and next_id.item() == eos_id:
            break  # 只有显式提供 EOS 时才应用这个停止条件。
    return tokens[prompt_ids.numel():]  # 只返回新增 token，不包含原 prompt。


# 11 创建所有对象，再执行第一次 model 调用
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("请在已经安装 CUDA PyTorch 的 GPU 机器运行第一课")
    device = torch.device("cuda:0")  # 本课明确使用第一张可见 GPU。
    torch.manual_seed(7)  # 固定随机参数，方便你修改代码后对照。
    config = ModelConfig()  # 创建第 1 段定义的配置对象。
    model = CausalLM(config)  # model 在这一行出现：调用第 9 段定义的类，创建实例。
    model = model.to(device)  # 将参数和 buffer 迁移到 GPU，仍然是同一个模型对象。
    model.eval()  # 使用推理时的模块行为。
    prompt_ids = torch.tensor([1, 5, 9], dtype=torch.long, device=device)  # 输入与模型同设备。
    with torch.inference_mode():
        logits = model(prompt_ids)  # nn.Module 的调用机制进入 CausalLM.forward。
    print("模型类型:", type(model).__name__)  # 应为 CausalLM。
    print("输入形状:", tuple(prompt_ids.shape))  # 应为 (3,)。
    print("logits 形状:", tuple(logits.shape))  # 应为 (3, 32)。
    assert logits.shape == (3, config.vocab_size)
    assert torch.isfinite(logits).all().item()  # 当前分数中不应出现 NaN 或无穷。
    completion = greedy_generate(model, prompt_ids, max_new_tokens=4)  # 传入同一个 model。
    print("新增 token:", completion.tolist())  # 随机权重的整数输出没有语言意义。


if __name__ == "__main__":
    main()  # 只有直接运行文件时才执行，import 时只建立上面的定义。
