import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model

"""
ModelRunner 负责：
    加载模型权重。
    管理 GPU。
    执行 Prefill 和 Decode。
    管理 KV Cache。
    执行采样。
"""
class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        """
        [张量并行] IMPORTANT: 初始化 NCCL 进程组，并通过 torch.cuda.set_device(rank) 为每个 rank 选择 GPU。

        `2333` 是 PyTorch 用来让各个 rank 会合、交换初始化信息的端口；GPU 张量的传输路径另由 NCCL 建立
        """
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        # 模型接入点
        self.model = Qwen3ForCausalLM(hf_config) # [自动估算缓存容量] 初始化顺序
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph() # [CUDA Graph]
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
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

    """
    TP>1 时，`call("exit")` 会:
        rank 0 把 "exit" 写入共享内存
        → 设置所有 worker 的 Event
        → rank 0 自己执行 ModelRunner.exit()
        → worker 收到 Event 后读取 "exit"
        → worker 也执行自己的 ModelRunner.exit()

    TP=1 时没有 worker，rank 0 直接执行自己的 ModelRunner.exit()

    （这里的 worker 指的是引擎初始化时，为张量并行额外启动的 Python 子进程。它不是线程，也不是一条请求）
    """
    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    """
    模型预热：先清理空闲缓存、重置显存峰值统计，再根据 max_num_batched_tokens、`max_model_len` 和 max_num_seqs 创建全零 token 的模拟序列，设置 num_scheduled_tokens

    预热的作用：1. 测量执行时的额外显存需求，给 KV Cache 留出合理预算。2. 把首次执行的部分开销提前到启动阶段。
    """
    def warmup_model(self): # [自动估算缓存容量] 重置峰值统计，构造预热输入并执行一次 Prefill
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    """[Paged Attention] 真正的 GPU KV 池 由这个方法分配"""
    def allocate_kv_cache(self): # [增量 KV Cache] 分配缓存  # [自动估算缓存容量] 随后读取
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        # block_bytes = 一个块覆盖全部层的 K/V 所需字节数
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
        assert config.num_kvcache_blocks > 0
        """
        [Paged Attention] 核心语句

        令：
        L   = Transformer 层数
        N   = 物理块总数
        B   = 每块 token 数
        Hkv = 当前 GPU 保存的 KV head 数
        d   = 每个 head 的维度

        整个张量布局就是：
        [2, L, N, B, Hkv, d]
        """
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim) # [分页 KV Cache] 物理缓存布局
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id] # [增量 KV Cache] 把每层对应的缓存挂到该层 Attention
                module.v_cache = self.kv_cache[1, layer_id] # [增量 KV Cache] 把每层对应的缓存挂到该层 Attention
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]): # [分页 KV Cache] 读取时，这里将各请求的块表补齐为 GPU Tensor，再传给注意力接口的 block_table 参数。
        """准备块表"""
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]): # [Prefill / Decode 分阶段执行] 处理从 num_cached_tokens 开始、本轮安排计算的 token # [变长请求批处理]
        """
        准备 prefill
        """
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        """每个请求先确定本轮计算范围"""
        for seq in seqs:
            """
            比如：
            start = 512
            seqlen_q = 88
            end = 600
            seqlen_k = 600

            那么模型本轮只为位置 512～599 计算 Q/K/V，但这些 Query 需要访问从位置 0 开始的历史 K/V
            """
            start = seq.num_cached_tokens # [分块预填充] 下一轮从新的 num_cached_tokens 位置继续计算。
            """本轮 Query 长度 = 本轮新计算的 token 数"""
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            """本轮可读取的 Key 长度 = 已有缓存长度 + 本轮新计算长度"""
            seqlen_k = end
            input_ids.extend(seq[start:end])
            """
            positions 传的是逻辑位置：[512, 513, ..., 599]
            它与物理块编号无关。token 的位置编码使用逻辑位置；物理块编号只决定缓存放到哪里。
            """
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            """
            接下来构造新 KV 的写入地址
            实际上就是在执行公式：slot(t) = block_table[t // B] * B + t % B # 其中 B 就是 kvcache_block_size
            """
            start_block = start // self.block_size # [分页 KV Cache] 写入地址映射，核心是：物理 token 槽位 = 物理块编号 × block_size + 块内偏移
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                """
                假设 block_table = [7, 2, 9]

                第一次计算完整 600 个 token 时，写入槽位依次为：
                位置   0～255 → 物理块 7 → slot 1792～2047
                位置 256～511 → 物理块 2 → slot  512～767
                位置 512～599 → 物理块 9 → slot 2304～2391

                所以：
                slot_mapping = [
                    1792, ..., 2047,
                    512, ...,  767,
                    2304, ..., 2391,
                ]
                顺序与本轮打包后的输入 token 顺序严格对应。
                """
                slot_mapping.extend(range(slot_start, slot_end))
        """
        触发读取分页缓存的判断
        (本质上是在判断：本批次的总有效 K 长度，是否大于总新 Query 长度。只要有请求需要读取以前的 KV，就需要走这条路径。)
        """
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        """
        Prefill 把多条请求的新 token 打包到一起，通过：cu_seqlens_q、cu_seqlens_k 描述每条请求的逻辑长度。
        例如：
            请求 A：已有 512 个缓存 token，本轮计算 88 个
            请求 B：没有缓存，本轮计算 50 个
        那么：
            cu_seqlens_q = [0, 88, 138]
            cu_seqlens_k = [0, 600, 650]
        含义是：
            A 的 Query 长度 = 88
            B 的 Query 长度 = 50

            A 的有效 Key 长度 = 600
            B 的有效 Key 长度 = 50
        """
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]): # [Prefill / Decode 分阶段执行] 每条序列只输入 last_token，位置为 len(seq) - 1
        """准备 decode"""
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            """每个请求只输入最后一个 token"""
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1) # [分页 KV Cache] Decode 的单 token 槽位在这里计算
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    """
    `inference_mode()` 的作用，是让 PyTorch 省掉“为以后求梯度做准备”的记录工作。
    """
    @torch.inference_mode() # 普通模型执行、CUDA Graph 重放时的 logits 计算，以及图捕获阶段的前向，都在这些`@torch.inference_mode()`函数的推理模式作用范围内，不构建反向传播计算图
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            """普通执行路径"""
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]: # [Prefill / Decode 分阶段执行] 执行端在这里分流
        """
        ModelRunner.run(): 准备本轮输入，执行模型并采样
            准备 token、位置、块表、写入地址
            执行各层模型
            写入 KV
            读取 KV 并计算 Attention
            采样输出 token
        """

        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        """主进程采样"""
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    @torch.inference_mode() # 普通模型执行、CUDA Graph 重放时的 logits 计算，以及图捕获阶段的前向，都在这些`@torch.inference_mode()`函数的推理模式作用范围内，不构建反向传播计算图
    def capture_cudagraph(self):
        """捕获过程包括分配固定输入/输出缓冲区、建立 batch size 档位、逐档预热，再进入 torch.cuda.graph(...) 捕获"""
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
