import time
import torch
import asyncio

from typing import List
from typing import AsyncGenerator

from model import Model
from tokenizer import Tokenizer
from scheduler import Scheduler
from request import Request
from request import SamplingParams
from dataclasses import dataclass

from torch.nn.utils.rnn import pad_sequence

@dataclass
class GenerationState:
    ''' 生成统计信息，用于跟踪生成过程 这是学习的关键！'''
    return_generation_state: bool = False # 是否返回生成状态统计信息
    prefill_time: float = 0.0
    decode_time: float = 0.0
    total_time: float = 0.0
    prefill_memory: float = 0.0
    decode_memory: float = 0.0
    total_memory: float = 0.0
    num_tokens_generated: int = 0
    tokens_per_second: float = 0.0    
    prefill_kv_cache_size_mb: float = 0.0 # prefill阶段的KV Cache大小
    kv_cache_size_mb: float = 0.0 # 总KV Cache的大小，单位MB    
    prompt_tokens: int = 0 # prompt中的token数
    time_to_first_token: float = 0.0 # 生成第一个token的时间
    eos_info: bool = False # 是否由于生成了eos而停止


class MiniVLLM:

    model: Model
    tokenizer: Tokenizer
    scheduler: Scheduler
    engine_running: bool

    def __init__(self, model_id: str, bnb_config=None, device_map="cuda"):
        self.model = Model(model_id, bnb_config, device_map=device_map)
        self.tokenizer = Tokenizer(model_id)
        self.scheduler = Scheduler()
        self.engine_running = False
        self.engine_task = None
        pass


    async def generate(self, prompt: str, max_new_tokens: int = 100, 
                 temperature: float = 1.0, 
                 top_p: float = 1.0, 
                 return_generation_state: bool = False) -> AsyncGenerator[str, None]:
        
        # 首先要把prompt保存到request中，并添加到shceduler管理器中
        req = self._make_request(prompt, max_new_tokens, temperature, top_p)
        self.scheduler.add_request(req)

        if self.engine_task is None or self.engine_task.done():
            self.engine_task = asyncio.create_task(self._engine_loop())        

        # 接收已经生成的token，并以stream方式返回
        emitted = 0
        while not req.finished or emitted < req.gen_token_ids.size(1):
            cur = req.gen_token_ids.size(1)
            if cur > emitted:
                token_id = int(req.gen_token_ids[0, emitted].item())
                yield self.tokenizer.decode([token_id])
                emitted += 1
                continue
            await asyncio.sleep(0)
        
    def _make_request(self, prompt: str, max_new_tokens: int = 100, 
                 temperature: float = 1.0, 
                 top_p: float = 1.0)->Request:
        
        input_ids = self.tokenizer.encode(prompt)
        req = Request(prefill_token_ids = input_ids, max_gen_tokens = max_new_tokens, sampling_params = SamplingParams(temperature, top_p))
        req.eos_token_id = self.tokenizer.eos_token_id
        req.finished = False
        req.gen_token_ids = torch.empty((1, 0), dtype=torch.long)
        return req


    async def _engine_loop(self):
        if self.engine_running:
            return
        
        self.engine_running = True
        try:
            while self.scheduler.has_pending():
                await self._step()
                await asyncio.sleep(0)
        finally:
            self.engine_running = False

    async def _step(self):
        if self.scheduler.waiting_list:
            waiting = self.scheduler.waiting_list[:]           
            prompted_ids =  await self._batch_prefill(waiting)
            for rid in prompted_ids:
                self.scheduler.promote_to_running(rid)

        if self.scheduler.running_list:
            running = self.scheduler.running_list[:]
            finished_ids = await self._handle_decode(running)
            for rid in finished_ids:
                self.scheduler.remove_request(rid)

    async def _batch_prefill(self, request_list: List[Request])->List[int]:
        

        if not request_list:
            return []
        device = self.model.model.device
        pad_id = self.tokenizer.pad_token_id #获取分词器中的填充token_id
        
        # 从每个request取出token tensor组成list，同时这些tensor 拷贝到目标设备上，如GPU
        # 当 device 是 GPU 时，.to(device) 通常会触发对应 tensor 的显存分配与数据拷贝，第一次用 CUDA 还可能有上下文初始化开销。
        seqs = [req.prefill_token_ids.squeeze(0).to(device) for req in request_list]

        # 把 list[int] 变成 torch.Tensor，这样后面才能做向量化张量运算（如 unsqueeze、广播比较）。
        # 放到同一 device（如 GPU），避免后续和 GPU tensor 运算时报设备不一致错误。
        lengths = torch.tensor([s.size(0) for s in seqs], device=device)

        # pad_sequence 函数把 list[int] 变成 torch.Tensor，并填充成相同长度，返回一个二维张量
        input_ids = pad_sequence(seqs, batch_first=True, padding_value=pad_id)

        # input_ids.size = [batch_size, max_len],直接说，参数0，行数，参数1，列数
        max_len = input_ids.size(1)
        # 获得attention_mask 张量，shape = [batch_size, max_len]，与input_ids张量一样，
        # 但是attention_mask张量中，1表示有效，0表示无效
        attention_mask = (
            torch.arange(max_len, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        ).long()

        outputs = self.model.forward(input_ids=input_ids, attention_mask=attention_mask)

        # 取出每条请求的最后一个token的 logits
        batch_idx = torch.arange(len(request_list), device=device)
        last_pos = lengths - 1
        next_logits = outputs.logits[batch_idx, last_pos, :]   # 每条样本最后一个真实 token 位置        

        promoted = []
        for i, req in enumerate(request_list):
            req.past_key_values = tuple(
                (k[i:i+1].contiguous(), v[i:i+1].contiguous())
                for (k, v) in outputs.past_key_values
            )

            tok = int(self._sample(next_logits[i:i+1,:], 
                                   req.sampling_params.temperature, 
                                   req.sampling_params.top_p
                                   ).item())
            req.next_token = tok
            req.gen_token_ids = torch.tensor([[tok]], device=device, dtype=torch.long)
            req.gen_text += self.tokenizer.decode([tok])
            req.finished = (tok == req.eos_token_id) or (req.max_gen_tokens <= 1)
            if req.finished:
                self.scheduler.remove_request(req.request_id)
            else:
                promoted.append(req.request_id)

        print("------ prefill done ------- \n")
        return promoted
        
    async def _batch_decode(self, request_list: List[Request]):
        # decode batch 几步走：
        # 1. 所有request 的最新的next_token ids组合成一个batch，组合成input_ids
        # 2. 所有request的kv cache 组合成一个batch
        
        device = self.model.model.device
        # 1.先搞定所有 next_token_ids
        input_ids = torch.tensor([req.next_token for req in request_list], device=device)

        # 2.开始搞定batch kv cache
        # 核心先记住：HF 的 past_key_values 结构是
        # tuple(layer_idx -> (k, v))，每层里通常是：

        # k: [B, num_heads, seq_len, head_dim]
        # v: [B, num_heads, seq_len, head_dim]
        # 每个 request 里存的是 B=1，所以 batch 堆叠就是按 batch 维 dim=0 拼。
        
        past_key_values = _merge_past_kv(request_list=request_list)

        outputs = self.model.forward(input_ids=input_ids)
        
        
        pass

    def _merge_past_kv(self, request_list):
        # 每个 req.past_key_values: tuple[(k,v)]，其中 k/v shape [1, H, S, D]
        num_layers = len(request_list[0].past_key_values)
        merged = []

        for l in range(num_layers):
            k_list = [req.past_key_values[l][0] for req in request_list]  # [1,H,S,D] x N
            v_list = [req.past_key_values[l][1] for req in request_list]
            k = torch.cat(k_list, dim=0)  # [N,H,S,D]
            v = torch.cat(v_list, dim=0)  # [N,H,S,D]
            merged.append((k, v))

        return tuple(merged)


    def _split_past_kv_back(self, merged_past_kv, request_list):
        for i, req in enumerate(request_list):
            req.past_key_values = tuple(
                (k[i:i+1].contiguous(), v[i:i+1].contiguous())
                for (k, v) in merged_past_kv
            )


    async def _handle_decode(self, request_list: List[Request]):

        finished_ids = []

        for req in request_list:
            if req.finished:
                finished_ids.append(req.request_id)
                continue

            input_ids = torch.tensor([[req.next_token]], dtype=torch.long, device=self.model.model.device)
            outputs = self.model.forward(input_ids=input_ids, past_key_values=req.past_key_values)

            # next_token = int(torch.argmax(outputs.logits[:, -1, :], dim=-1).item())
            next_token = int(self._sample(outputs.logits[ :, -1, :], 
                                   req.sampling_params.temperature, 
                                   req.sampling_params.top_p
                                   ).item())
            req.past_key_values = outputs.past_key_values
            req.next_token = next_token
            req.gen_token_ids = torch.cat(
                [req.gen_token_ids, torch.tensor([[next_token]], device=req.gen_token_ids.device)],
                dim=-1,
            )
            req.gen_text += self.tokenizer.decode([next_token])

            gen_len = req.gen_token_ids.size(1)
            req.finished = (next_token == req.eos_token_id) or (gen_len >= req.max_gen_tokens)
            if req.finished:
                finished_ids.append(req.request_id)

        return finished_ids
    
    def _sample(self, logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
        """
        logits: [B, V]
        return: [B, 1]
        """
        if logits.dim() != 2:
            raise ValueError(f"logits must be 2D [B, V], got shape={tuple(logits.shape)}")
        if top_p <= 0.0 or top_p > 1.0:
            raise ValueError(f"top_p must be in (0, 1], got {top_p}")

        # greedy
        if temperature <= 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        # temperature scaling
        logits = logits / temperature

        # top-p (batch-safe)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)   # [B, V], [B, V]
            sorted_probs = torch.softmax(sorted_logits, dim=-1)                            # [B, V]
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)                          # [B, V]

            # remove tokens with cumulative prob > top_p, but keep first token above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            # scatter mask back to original vocab order
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)                  # [B, V]
            indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        probs = torch.softmax(logits, dim=-1)                                               # [B, V]
        next_token_id = torch.multinomial(probs, num_samples=1)                            # [B, 1]
        return next_token_id


    def _sample_old(self, logits: torch.Tensor, temperature: float, top_p: float)->torch.Tensor:

            if temperature <= 0.0:
                # 直接取最大值
                return logits.argmax(dim=-1, keepdim=True)
            
            # 应用温度
            logits = logits / temperature

            # Top-p(nucleus) sampling
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

                # 找到累计概率超过top_p的索引
                sorted_indices_to_remove = cumulative_probs > top_p
                # 保留第一个超过top_p的token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # 将这些token的logits设为负无穷
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[0, indices_to_remove] = -float('Inf')

            # 计算概率分布
            probs = torch.softmax(logits, dim=-1)
            # 从分布中采样
            next_token_id = torch.multinomial(probs, num_samples=1)
            return next_token_id
