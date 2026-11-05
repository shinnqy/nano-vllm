from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


"""
Scheduler 负责：
    waiting 队列。
    running 队列。
    KV 块管理器。
    最大 batch 大小。
    EOS ID。
"""
class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        """
        保存结束符 token ID
        模型采样出 token 后，调度器需要判断是否结束
        """
        self.eos = config.eos
        self.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque() # [动态批处理] 等待执行或继续 Prefill 的请求
        self.running: deque[Sequence] = deque() # [动态批处理] 已进入可继续 Decode 状态、尚未完成的请求

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence): # 新请求的进入路径是: LLMEngine.add_request() → 创建 Sequence → Scheduler.add() → waiting.append(seq)
        self.waiting.append(seq)

    """选择这一轮要交给 ModelRunner 的请求"""
    def schedule(self) -> tuple[list[Sequence], bool]: # [动态批处理] 核心逻辑
        scheduled_seqs = [] # [动态批处理] 每次 schedule() 都新建 scheduled_seqs，重新从队列中选择本轮参与计算的请求。因此，一个 batch 的成员可以随着请求加入、完成、抢占而变化。
        num_batched_tokens = 0

        """
        [continuous batching] 先尝试安排 prefill

        这里同时考虑：
            - 等待队列中有没有请求。
            - 本轮已经选了多少条请求。
            - 本轮还剩多少新 token 的计算预算。
        """
        # prefill
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs: # [Prefill / Decode 分阶段执行] 进行“阶段选择” # [每轮序列数限制]
            seq = self.waiting[0] # [continuous batching] 只查看而暂时不移除，是因为它可能还需要多轮 Prefill，或者当前资源不足，暂时不能接纳
            remaining = self.max_num_batched_tokens - num_batched_tokens # [Prefill token 预算] 核心逻辑
            if remaining == 0:
                break
            # 调度器判断还要计算多少 token，以及能否分配 KV 块
            if not seq.block_table:
                num_cached_blocks = self.block_manager.can_allocate(seq)
                if num_cached_blocks == -1:
                    break
                num_tokens = seq.num_tokens - num_cached_blocks * self.block_size
            else:
                num_tokens = seq.num_tokens - seq.num_cached_tokens
            """计算预算不足时，它允许把一个长 prompt 分块处理。"""
            if remaining < num_tokens and scheduled_seqs:  # only allow chunked prefill for the first seq # [分块预填充] 入口条件
                break
            if not seq.block_table:
                self.block_manager.allocate(seq, num_cached_blocks)
            """min(num_tokens, remaining) 决定本轮究竟计算多少 token"""
            seq.num_scheduled_tokens = min(num_tokens, remaining) # [Prefill token 预算] 实际计算。这里的 num_tokens 会扣除已缓存部分，因此预算统计的是本轮新计算的 token 数，命中的前缀缓存不重复占用计算预算
            num_batched_tokens += seq.num_scheduled_tokens
            if seq.num_cached_tokens + seq.num_scheduled_tokens == seq.num_tokens: # [请求状态管理] 最后一段 Prefill 被安排执行时，改为 RUNNING，从等待队列移到运行队列
                # WAITING → RUNNING
                seq.status = SequenceStatus.RUNNING
                self.waiting.popleft()
                self.running.append(seq)
            scheduled_seqs.append(seq)

        if scheduled_seqs: # [Prefill / Decode 分阶段执行] 提前返回的位置。所以同一轮只执行 Prefill 或 Decode，当前没有把两者混入同一 batch。持续有可执行的 Prefill 时，会推迟已有请求的 Decode
            return scheduled_seqs, True

        """没有选出 Prefill 时，才安排 Decode"""
        # decode
        while self.running and len(scheduled_seqs) < self.max_num_seqs: # [每轮序列数限制]
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq): # [缓存不足时抢占] 触发点
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                seq.num_scheduled_tokens = 1
                seq.is_prefill = False
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence): # [请求状态管理] 抢占后改回 WAITING。 # [缓存不足时抢占] 执行
        seq.status = SequenceStatus.WAITING
        seq.is_prefill = True
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(
        self,
        seqs: list[Sequence], # 接收本轮执行的请求
        token_ids: list[int], # 接收本轮执行的每条请求采样出的 token
        is_prefill: bool
    ):
        """更新请求进度、处理完成、回收块引用"""

        for seq, token_id in zip(seqs, token_ids):
            self.block_manager.hash_blocks(seq)
            seq.num_cached_tokens += seq.num_scheduled_tokens # [分块预填充] 增加 num_cached_tokens；若 Prefill 尚未完成，跳过追加生成 token。
            seq.num_scheduled_tokens = 0
            if is_prefill and seq.num_cached_tokens < seq.num_tokens: # [请求状态管理] num_cached_tokens 和 num_tokens 有区别：刚采样出来的 token 已经加入结果，但它的 KV 通常要下一轮才计算
                continue
            """把采样出的 token 追加到 Sequence"""
            seq.append_token(token_id)
            """
            检查是否应该结束请求
                1. 除非请求明确要求忽略 EOS，否则采样出 EOS 就停止 => 请求结束
                2. 生成数量达到用户设置的最大值 => 请求结束
            """
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens: # [请求状态管理] 生成 EOS 或达到输出长度限制后，改为 FINISHED
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                # 从 running 队列中删除已完成的请求
                self.running.remove(seq)
