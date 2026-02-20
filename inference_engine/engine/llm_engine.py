import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from inference_engine.config import Config
from inference_engine.sampling_params import SamplingParams
from inference_engine.engine.sequence import Sequence
from inference_engine.engine.scheduler import Scheduler
from inference_engine.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, tokenizer_path=None, **kwargs):
        print(f"[LLMEngine] [INIT] START: Initializing LLMEngine with model={model}", flush=True)
        
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        print(f"[LLMEngine] [INIT] Config created: tensor_parallel_size={config.tensor_parallel_size}", flush=True)
        
        # Use tokenizer_path if provided, otherwise use model path
        tokenizer_path = tokenizer_path or model
        
        self.ps = []
        self.events = []
        self._exited = False  # Guard against double cleanup
        ctx = mp.get_context("spawn")
        
        print(f"[LLMEngine] [INIT] Spawning {config.tensor_parallel_size - 1} worker processes...", flush=True)
        for i in range(1, config.tensor_parallel_size):
            print(f"[LLMEngine] [INIT] Starting worker process {i}...", flush=True)
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            print(f"[LLMEngine] [INIT] Worker process {i} started (PID: {process.pid})", flush=True)
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        print(f"[LLMEngine] [INIT] Rank 0 ModelRunner initialized", flush=True)
        print(f"[LLMEngine] [INIT] Loading tokenizer from {tokenizer_path}...", flush=True)
        if tokenizer_path is None:
            raise ValueError("tokenizer_path is None when trying to load tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        print(f"[LLMEngine] [INIT] Tokenizer loaded successfully", flush=True)
        print(f"[LLMEngine] [INIT] Tokenizer properties: vocab_size={self.tokenizer.vocab_size}, eos_token_id={self.tokenizer.eos_token_id}, pad_token_id={self.tokenizer.pad_token_id}", flush=True)
        config.eos = self.tokenizer.eos_token_id
        config.pad = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        print(f"[LLMEngine] [INIT] Config updated: config.eos={config.eos}, config.pad={config.pad}, vocab_size={config.hf_config.vocab_size}", flush=True)
        self.scheduler = Scheduler(config)
        self.model_runner.block_manager = self.scheduler.block_manager
        self.scheduler.set_kv_cache(self.model_runner.kv_cache)
        atexit.register(self.exit)
        print(f"[LLMEngine] [INIT] COMPLETE: LLMEngine initialization finished", flush=True)

    def exit(self):
        if getattr(self, '_exited', False):
            return  # Already cleaned up
        self._exited = True
        
        if hasattr(self, 'model_runner') and self.model_runner is not None:
            self.model_runner.call("exit")
            self.model_runner = None
        
        if hasattr(self, 'ps'):
            for i, p in enumerate(self.ps):
                p.join(timeout=10)
                if p.is_alive():
                    print(f"[LLMEngine] Worker {i+1} didn't exit, terminating...", flush=True)
                    p.terminate()
                    p.join(timeout=5)
                    if p.is_alive():
                        print(f"[LLMEngine] Worker {i+1} didn't terminate, killing...", flush=True)
                        p.kill()
            self.ps = []

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        result = self.model_runner.call("run", seqs, is_prefill)
        
        if result is None or (is_prefill and len(result) > 0 and result[0] == []):
            outputs = []
            num_tokens = sum(len(seq) for seq in seqs)
            return outputs, num_tokens
        
        is_on_policy = False
        if len(result) > 0 and isinstance(result[0], dict):
            is_on_policy = True
        
        if is_on_policy:
            rollout_records = result
            
            num_new_tokens = 0
            if not is_prefill:
                num_new_tokens = sum(seq.num_completion_tokens for seq in seqs)
                
            for seq in seqs:
                seq.status = seq.status.__class__.FINISHED
                seq._rollout_records = rollout_records
                if seq in self.scheduler.running:
                    self.scheduler.running.remove(seq)
            
            outputs = [(seq.seq_id, seq._rollout_records) for seq in seqs if seq.is_finished]
            
            if is_prefill:
                num_tokens = sum(len(seq) for seq in seqs)
            
            num_tokens = -num_new_tokens if num_new_tokens > 0 else 0
            
            return outputs, num_tokens
        
        token_ids = result
        
        is_jacobi = False
        if len(token_ids) > 0:
            first_elem = token_ids[0]
            is_jacobi = isinstance(first_elem, (list, tuple))
        
        if is_jacobi:
            self.scheduler.postprocess_jacobi(seqs, token_ids)
            num_new_tokens = sum(len(toks) for toks in token_ids)
        else:
            self.scheduler.postprocess(seqs, token_ids)
            num_new_tokens = len(seqs)
        
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -num_new_tokens
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        #print(f"Invoking Generation function...", flush=True)
        #print(f"Generating {len(prompts)} prompts...", flush=True)
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for seq_id, result_data in output:
                outputs[seq_id] = result_data
                if use_tqdm:
                    pbar.update(1)
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        
        if len(outputs) > 0 and isinstance(outputs[0], list) and len(outputs[0]) > 0 and isinstance(outputs[0][0], dict):
            rollout_records = []
            for seq_output in outputs:
                if isinstance(seq_output, list):
                    rollout_records.extend(seq_output)
                else:
                    rollout_records.append(seq_output)
            if use_tqdm:
                pbar.close()
            return rollout_records
        
        def flatten_token_ids(token_ids):
            """Ensure token_ids is a flat list of integers."""
            if all(isinstance(t, int) for t in token_ids):
                return token_ids
            flattened = []
            for item in token_ids:
                if isinstance(item, (list, tuple)):
                    flattened.extend(item)
                else:
                    flattened.append(int(item))
            return flattened
        
        outputs = [{"text": self.tokenizer.decode(flatten_token_ids(token_ids)), "token_ids": flatten_token_ids(token_ids)} for token_ids in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
