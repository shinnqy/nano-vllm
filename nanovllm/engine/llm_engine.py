import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        """获取 Config 定义的属性"""
        config_fields = {field.name for field in fields(Config)}
        """是在从 kwargs 中筛选出 Config 支持的参数，构造一个新的字典"""
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size): # [张量并行] 根据 tensor_parallel_size 创建其他 rank 的进程，主进程运行 rank 0
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        """
        use_fast：要求优先使用 Rust 实现的快速 tokenizer
        """
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        """调度器要靠这个判断结束，所以先于调度器设置"""
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        """向 Python 注册一个退出时要调用的清理函数。"""
        atexit.register(self.exit)

    def exit(self):
        """
        self.model_runner 是 rank 0 的 ModelRunner。
        调用 call("exit") 不是普通地直接执行 exit()，因为多卡时需要先通知其他 worker
        """
        """
        TP>1 时，`call("exit")` 会:
            rank 0 把 "exit" 写入共享内存
            → 设置所有 worker 的 Event
            → rank 0 自己执行 ModelRunner.exit()
            → worker 收到 Event 后读取 "exit"
            → worker 也执行自己的 ModelRunner.exit()

        TP=1 时没有 worker，rank 0 直接执行自己的 ModelRunner.exit()
        """
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    """[continuous batching] 这个方法用来请求入队：创建请求，放入调度器的 waiting 队列"""
    def add_request(
        self,
        prompt: str | list[int], # 原始文本 | 已编码的 token IDs
        sampling_params: SamplingParams
    ): # 新请求的进入路径是: LLMEngine.add_request() → 创建 Sequence → Scheduler.add() → waiting.append(seq)
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        """创建请求对象 seq"""
        seq = Sequence(prompt, sampling_params)
        """
        请求进入 waiting 队列。
        请求提交之后，还需要与其他请求一起接受调度。
        将提交与执行分开，调度器才能综合考虑当前有哪些等待请求、哪些运行请求，以及还有多少资源
        """
        """
        IMPORTANT: 这里不立即运行模型，因为引擎需要把多个请求放在一起考虑：
            - 本轮 token 预算是否足够。
            - 是否可以分配 KV Cache。
            - 当前 batch 是否已达到请求数上限。
            - 应先执行 Prefill 还是 Decode。
        """
        self.scheduler.add(seq)

    """
    step() 只推进一轮，控制权随后回到引擎，调度器就有机会重新选择下一轮请求
    推进引擎一轮。它不会让一条请求一直生成到结束，而是只执行一个调度步骤
    """
    def step(self):
        """
        Scheduler.schedule(): 选择请求，决定 Prefill / Decode
            选择本轮请求
            分配本轮需要的块
            设置 num_scheduled_tokens

        => ”这一轮谁来算？“
        """
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        """
        ModelRunner.run(): 准备本轮输入，执行模型并采样
            准备 token、位置、块表、写入地址
            执行各层模型
            写入 KV
            读取 KV 并计算 Attention
            采样输出 token

        => 为这些请求计算这一轮
        """
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        """
        Scheduler.postprocess(): 更新缓存进度、追加输出、处理结束和回收

            登记新完成的完整缓存块
            更新 num_cached_tokens
            追加输出 token
            对结束请求释放块引用

        => 算完以后，各请求到了什么进度？谁结束了？
        """
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        """把每个 prompt 变成一个请求，放进调度器"""
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            """把 sampling_params 复制为一列，列里的个数是 prompts 的数量"""
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            """把全部 prompt 转成 Sequence，并放入 waiting 队列。"""
            self.add_request(prompt, sp)
        """
        用来建立 seq_id -> completion_token_ids
        IMPORTANT: 完成顺序可能不同，所以不能简单地 append。例如 B 可能比 A 先结束，但最终仍要按输入顺序返回 A、B。
        """
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        """只要调度器中还有请求，就继续执行下一轮"""
        while not self.is_finished():
            """
            IMPORTANT: while 这个循环会不断重新组成 batch，例如：
                第 1 轮：A、B 一起 Prefill
                第 2 轮：A、B 一起 Decode
                第 3 轮：B 已完成，只剩 A

            这就是 continuous batching 的核心。
            """
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            """
            每一步可能完成零条或多条请求
            完成一条就：
                - 保存它的输出 token IDs。
                - 把进度增加 1
            """
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        """按 seq_id 排序，使完成顺序恢复为提交顺序"""
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        """把每条生成的 token IDs 解码成文本。"""
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
