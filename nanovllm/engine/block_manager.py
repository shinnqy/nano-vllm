from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block: # [缓存块分配与回收] 每个块的 ref_count、hash、token_ids 定义这里

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        # 物理块的 CPU 元数据
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        # 可复用前缀块的索引
        self.hash_to_block_id: dict[int, int] = dict()
        # 当前没有请求引用的块
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        # 当前至少被一个请求引用的块
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1): # [前缀缓存] 哈希计算
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self) -> int: # [缓存块分配与回收] 分配新物理块: 从空闲队列取块，清理旧映射，重置元数据，加入已用集合
        # 1. 从空闲队列取一个物理块编号。
        block_id = self.free_block_ids.popleft()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        # 2. 如果它之前保留着可复用缓存，移除指向它的旧索引。
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block_id: # [请求结束后保留缓存] 当块被当作新块重新分配时，这里才删除指向它的旧哈希映射，并调用 block.reset() 重置元数据；GPU 内容随后由新的计算覆盖。
            del self.hash_to_block_id[block.hash]
        # 3. 把元数据重置为新块
        block.reset()
        # 把它标记为已占用
        self.used_block_ids.add(block_id)
        return block_id

    def _deallocate_block(self, block_id: int): # [请求结束后保留缓存]
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> int: # [前缀缓存] 查找
        h = -1
        num_cached_blocks = 0
        num_new_blocks = seq.num_blocks
        for i in range(seq.num_blocks - 1): # [缓存粒度控制] 最后一块不复用，则由这里明确控制
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            num_cached_blocks += 1
            if block_id in self.used_block_ids:
                num_new_blocks -= 1
        if len(self.free_block_ids) < num_new_blocks:
            return -1
        return num_cached_blocks

    def allocate(self, seq: Sequence, num_cached_blocks: int): # [缓存块分配与回收] 给请求建立块表: 引用命中块，分配剩余块，填充 seq.block_table
        assert not seq.block_table
        h = -1
        for i in range(num_cached_blocks): # [跨请求共享前缀]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block_id = self.hash_to_block_id[h]
            block = self.blocks[block_id]
            if block_id in self.used_block_ids:
                block.ref_count += 1
            else:
                block.ref_count = 1
                self.free_block_ids.remove(block_id)
                self.used_block_ids.add(block_id)
            seq.block_table.append(block_id)
        for i in range(num_cached_blocks, seq.num_blocks): # [分块预填充] 初次分配时会给整条当前序列分配所需块。因此，分块控制的是每轮计算量，缓存块仍会提前分配。
            seq.block_table.append(self._allocate_block())
        seq.num_cached_tokens = num_cached_blocks * self.block_size

    def deallocate(self, seq: Sequence): # [缓存块分配与回收] 释放请求引用: 逐块减少引用计数，计数归零后回到空闲队列 # [跨请求共享前缀] 释放时，只有在 ref_count == 0 时才回收物理块。因此，一条请求结束不会回收其他请求仍在引用的块。
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        """
        生成超过当前块容量时，调度器会增加一个物理块
        （这里的判断条件为什么是 len(seq) % B == 1，需要结合“先采样 token，下一轮才计算它的 KV”来理解。）
        """
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence): # [缓存块分配与回收] 跨块时追加: 需要时新增一个块
        """生成超过当前块容量时，调度器会增加一个物理块"""
        if len(seq) % self.block_size == 1:
            seq.block_table.append(self._allocate_block())

    def hash_blocks(self, seq: Sequence): # [前缀缓存] 缓存登记，由模型执行后的 Scheduler.postprocess() 调用。因此，新块完成计算后才进入可复用索引。
        start = seq.num_cached_tokens // self.block_size # [缓存粒度控制] 只登记完整块的逻辑在这里
        end = (seq.num_cached_tokens + seq.num_scheduled_tokens) // self.block_size
        if start == end: return
        h = self.blocks[seq.block_table[start - 1]].hash if start > 0 else -1
        for i in range(start, end):
            block = self.blocks[seq.block_table[i]]
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h)
            block.update(h, token_ids)
            self.hash_to_block_id[h] = block.block_id
