import pickle
import os
import time
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from inference_engine.config import Config
from inference_engine.engine.sequence import Sequence
from inference_engine.models.qwen3 import Qwen3ForCausalLM
from inference_engine.layers.sampler import Sampler
from inference_engine.utils.context import set_context, get_context, reset_context
from inference_engine.utils.loader import load_model

from typing import Optional, Union
from inference_engine.engine.jacobi_decoding import JacobiDecoder
from inference_engine.engine.jacobi_decoding_nongreedy import JacobiDecoderNonGreedy
from inference_engine.engine.jacobi_decoding_nongreedy_on_policy import JacobiDecoderNonGreedyOnPolicy

from datetime import timedelta

# ============================================================================
# Lightweight Profiler (activated with PROFILE=1 environment variable)
# ============================================================================
_PROFILE = os.environ.get("PROFILE", "0") == "1"


class ProfileTimer:
    """Lightweight profiler for TPF→TPS bottleneck analysis.
    
    Activated with: PROFILE=1 python your_script.py
    """
    
    def __init__(self):
        self.enabled = _PROFILE
        self.timings = {}      # name -> total time (ms)
        self.counts = {}       # name -> call count
        self._start_times = {} # name -> start timestamp
        self._tokens_generated = 0
        self._iterations = 0
    
    def start(self, name: str):
        if self.enabled:
            torch.cuda.synchronize()
            self._start_times[name] = time.perf_counter()
    
    def stop(self, name: str):
        if self.enabled and name in self._start_times:
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - self._start_times[name]) * 1000  # ms
            if name not in self.timings:
                self.timings[name] = 0.0
                self.counts[name] = 0
            self.timings[name] += elapsed
            self.counts[name] += 1
            del self._start_times[name]
    
    def add_tokens(self, n: int):
        if self.enabled:
            self._tokens_generated += n
    
    def add_iteration(self):
        if self.enabled:
            self._iterations += 1
    
    def report(self):
        if not self.enabled or not self.timings:
            return
        
        print("\n" + "="*80)
        print("PROFILE REPORT: TPF → TPS Bottleneck Analysis")
        print("="*80)
        
        total = sum(self.timings.values())
        
        # Group by category
        categories = {
            'Forward Pass': ['jacobi.forward', 'jacobi.lm_head', 'ar.forward'],
            'Draft Building': ['jacobi.draft_build', 'jacobi.next_draft', 'jacobi.prefill_draft'],
            'Verification': ['jacobi.verify', 'jacobi.commit'],
            'KV/Block Ops': ['jacobi.trim', 'jacobi.block_alloc', 'block.trim', 'block.alloc', 'block.dealloc'],
            'Context Setup': ['jacobi.context_setup', 'jacobi.buffer_fill'],
            'Other': []
        }
        
        categorized = set()
        for cat_names in categories.values():
            categorized.update(cat_names)
        
        # Add uncategorized to Other
        for name in self.timings:
            if name not in categorized:
                categories['Other'].append(name)
        
        print(f"\n{'Component':<45} {'Time (ms)':>10} {'Calls':>8} {'Avg (ms)':>10} {'%':>7}")
        print("-"*80)
        
        for category, names in categories.items():
            cat_total = sum(self.timings.get(n, 0) for n in names)
            if cat_total > 0:
                cat_pct = cat_total / total * 100 if total > 0 else 0
                print(f"\n[{category}] ({cat_pct:.1f}%)")
                for name in names:
                    if name in self.timings:
                        t = self.timings[name]
                        c = self.counts[name]
                        avg = t / c if c > 0 else 0
                        pct = t / total * 100 if total > 0 else 0
                        print(f"  {name:<43} {t:>10.2f} {c:>8} {avg:>10.3f} {pct:>6.1f}%")
        
        print("-"*80)
        print(f"{'TOTAL':<45} {total:>10.2f}ms")
        
        # Summary stats
        if self._tokens_generated > 0 and total > 0:
            tps = self._tokens_generated / (total / 1000)
            print(f"\n📊 Summary:")
            print(f"   Tokens generated: {self._tokens_generated}")
            print(f"   Jacobi iterations: {self._iterations}")
            print(f"   Effective TPS: {tps:.1f}")
            if self._iterations > 0:
                tpf = self._tokens_generated / self._iterations
                print(f"   TPF (tokens/forward): {tpf:.2f}")
                
                # Calculate theoretical max TPS
                forward_time = self.timings.get('jacobi.forward', 0) + self.timings.get('jacobi.lm_head', 0)
                if forward_time > 0:
                    theoretical_tps = self._tokens_generated / (forward_time / 1000)
                    efficiency = tps / theoretical_tps * 100 if theoretical_tps > 0 else 0
                    print(f"   Theoretical max TPS (forward only): {theoretical_tps:.1f}")
                    print(f"   Efficiency: {efficiency:.1f}%")
                    overhead_pct = 100 - efficiency
                    print(f"   ⚠️  Overhead cost: {overhead_pct:.1f}% of potential throughput")
        
        print("="*80 + "\n")
    
    def reset(self):
        self.timings.clear()
        self.counts.clear()
        self._start_times.clear()
        self._tokens_generated = 0
        self._iterations = 0


# Global profiler instance
_profiler = ProfileTimer()


def get_profiler() -> ProfileTimer:
    """Get the global profiler instance."""
    return _profiler


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        print(f"[RANK {rank}] [INIT] START: ModelRunner initialization beginning", flush=True)
        print(f"[RANK {rank}] [INIT] Config: tensor_parallel_size={config.tensor_parallel_size}, model={config.model}", flush=True)
        
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        
        self.block_manager = None
        self.jacobi_decoder: Optional[Union[JacobiDecoder, JacobiDecoderNonGreedy, JacobiDecoderNonGreedyOnPolicy]] = None
        self._jacobi_config = config if config.jacobi_enabled else None
        
        self.cuda_graph_hits = 0
        self.cuda_graph_misses = 0
        self.cuda_graph_debug = os.environ.get('CUDA_GRAPH_DEBUG', '0') == '1'
        
        self.slot_mapping_cache = {}

        print(f"[RANK {rank}] [INIT] About to initialize process group (this blocks until all ranks join)", flush=True)
        timeout_minutes = int(os.environ.get('TORCH_DISTRIBUTED_TIMEOUT_MINUTES', '3'))
        master_port = int(os.environ.get('TORCH_DISTRIBUTED_PORT', '2333'))
        master_addr = f"tcp://localhost:{master_port}"
        print(f"[RANK {rank}] [INIT] Using distributed address: {master_addr}", flush=True)
        dist.init_process_group(
            "nccl", 
            master_addr, 
            world_size=self.world_size, 
            rank=rank,
            timeout=timedelta(minutes=timeout_minutes)
        )
        print(f"[RANK {rank}] [INIT] Process group initialized successfully", flush=True)
        
        print(f"[RANK {rank}] [INIT] Setting CUDA device to {rank}", flush=True)
        torch.cuda.set_device(rank)
        
        print(f"[RANK {rank}] [INIT] Loading model...", flush=True)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        print(f"[RANK {rank}] [INIT] Model loaded successfully", flush=True)
        
        print(f"[RANK {rank}] [INIT] Creating sampler and warming up model...", flush=True)
        self.sampler = Sampler()
        self.warmup_model()
        print(f"[RANK {rank}] [INIT] Model warmup complete", flush=True)
        
        # Synchronize all ranks before KV cache allocation to ensure consistent memory state
        if self.world_size > 1:
            print(f"[RANK {rank}] [INIT] Entering barrier before KV cache allocation", flush=True)
            dist.barrier()
            print(f"[RANK {rank}] [INIT] Barrier passed, proceeding with KV cache allocation", flush=True)
            
            # Have rank 0 calculate allocation and broadcast to all ranks for consistency
            if rank == 0:
                print(f"[RANK {rank}] [INIT] Rank 0 calculating KV cache allocation...", flush=True)
                self.allocate_kv_cache()
                num_blocks = self.config.num_kvcache_blocks
                # Calculate block_bytes for summary
                num_kv_heads = hf_config.num_key_value_heads // self.world_size
                head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
                block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
                # Broadcast the allocation size to all ranks
                num_blocks_tensor = torch.tensor([num_blocks], dtype=torch.int32, device='cuda')
                dist.broadcast(num_blocks_tensor, src=0)
                print(f"[RANK {rank}] [INIT] Rank 0 broadcasting num_kvcache_blocks={num_blocks} to all ranks", flush=True)
                print(f"[KV_CACHE_SUMMARY] All ranks allocated KV cache:", flush=True)
                print(f"[KV_CACHE_SUMMARY]   Rank 0: {num_blocks} blocks ({num_blocks * block_bytes / 1e9:.2f} GB)", flush=True)
            else:
                # Other ranks receive the allocation size from rank 0
                num_blocks_tensor = torch.tensor([0], dtype=torch.int32, device='cuda')
                dist.broadcast(num_blocks_tensor, src=0)
                self.config.num_kvcache_blocks = num_blocks_tensor.item()
                print(f"[RANK {rank}] [INIT] Rank {rank} received num_kvcache_blocks={self.config.num_kvcache_blocks} from rank 0", flush=True)
                # Allocate KV cache with the same size as rank 0
                print(f"[RANK {rank}] [INIT] Allocating KV cache...", flush=True)
                self._allocate_kv_cache_with_size()
            print(f"[RANK {rank}] [INIT] KV cache allocated", flush=True)
        else:
            print(f"[RANK {rank}] [INIT] Allocating KV cache...", flush=True)
            self.allocate_kv_cache()
            print(f"[RANK {rank}] [INIT] KV cache allocated", flush=True)
        
        if not self.enforce_eager:
            print(f"[RANK {rank}] [INIT] Capturing CUDA graphs...", flush=True)
            self.capture_cudagraph()
            if config.jacobi_enabled:
                self.capture_cudagraph_jacobi()
            print(f"[RANK {rank}] [INIT] CUDA graphs captured", flush=True)
        
        self._all_logits_buffer = None
        if self.world_size > 1 and rank == 0:
            # This is enough for typical batches, fallback allocation for larger ones
            typical_batch_size = min(2048, self.config.max_num_batched_tokens)
            vocab_size_per_rank = self.config.hf_config.vocab_size // self.world_size
            print(f"[RANK {rank}] [INIT] Pre-allocating all_logits buffer (batch_size={typical_batch_size}, vocab_per_rank={vocab_size_per_rank})...", flush=True)
            torch.cuda.empty_cache()
            self._all_logits_buffer = [
                torch.empty(typical_batch_size, vocab_size_per_rank, dtype=self.config.hf_config.torch_dtype, device='cuda')
                for _ in range(self.world_size)
            ]
            print(f"[RANK {rank}] [INIT] all_logits buffer pre-allocated successfully", flush=True)
        
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            print(f"[RANK {rank}] [INIT] Setting up shared memory and barrier...", flush=True)
            if rank == 0:
                try:
                    existing_shm = SharedMemory(name="inference_engine", create=False)
                    existing_shm.close()
                    existing_shm.unlink()
                    print(f"[RANK {rank}] [INIT] Rank 0: Cleaned up existing shared memory", flush=True)
                except FileNotFoundError:
                    pass
                
                self.shm = SharedMemory(name="inference_engine", create=True, size=2**20)
                print(f"[RANK {rank}] [INIT] Rank 0: Shared memory created, entering barrier", flush=True)
                dist.barrier()
                print(f"[RANK {rank}] [INIT] Rank 0: Barrier passed", flush=True)
            else:
                print(f"[RANK {rank}] [INIT] Rank {rank}: Entering barrier", flush=True)
                dist.barrier()
                print(f"[RANK {rank}] [INIT] Rank {rank}: Barrier passed, connecting to shared memory", flush=True)
                self.shm = SharedMemory(name="inference_engine")
                print(f"[RANK {rank}] [INIT] Rank {rank}: Shared memory connected, entering worker loop", flush=True)
                self.loop()
        
        print(f"[RANK {rank}] [INIT] COMPLETE: ModelRunner initialization finished", flush=True)
    
    def _ensure_jacobi_decoder_initialized(self, seqs: Optional[list[Sequence]] = None):
        """Initialize jacobi_decoder if needed and block_manager is available.
        
        Args:
            seqs: Optional list of sequences to check temperature and on-policy flags from.
                  If None, defaults to greedy decoder.
                  
        Routing logic for decode_strategy == 'jacobi':
            1. If any seq has jacobi_on_policy=True -> use JacobiDecoderNonGreedyOnPolicy
               (temp > 0 already validated at this step)
            2. Else if all jacobi seqs have temp == 0 -> use JacobiDecoder (greedy)
            3. Else if all jacobi seqs have temp > 0 -> use JacobiDecoderNonGreedy (non-greedy)
            4. Else (mixed temp) -> raise NotImplementedError
            
        Note: Only rank 0 has block_manager. Workers use _worker_jacobi_forward_loop
        to participate in forward passes without needing the decoder.
        """
        if self._jacobi_config is not None and self.block_manager is not None and self.jacobi_decoder is None:
            if seqs:
                # Filter to only jacobi sequences
                jacobi_seqs = [s for s in seqs if s.decode_strategy == "jacobi"]
                
                if jacobi_seqs:
                    # Check on-policy flag (only check jacobi sequences)
                    use_on_policy = any(getattr(s, 'jacobi_on_policy', False) for s in jacobi_seqs)
                    
                    # Collect temperatures for jacobi sequences
                    temps = []
                    for seq in jacobi_seqs:
                        temp = getattr(seq, 'temperature', 0.0)
                        if temp <= 0.0:
                            sp = getattr(seq, 'sampling_params', None)
                            if sp is not None:
                                temp = getattr(sp, 'temperature', 0.0)
                        temps.append(temp)
                    
                    # Check temperature consistency
                    all_greedy = all(t == 0.0 for t in temps)
                    all_nongreedy = all(t > 0.0 for t in temps)
                    mixed_temp = not (all_greedy or all_nongreedy)
                    
                    if use_on_policy:
                        self.jacobi_decoder = JacobiDecoderNonGreedyOnPolicy(
                            block_manager=self.block_manager,
                            forward_step=self._jacobi_forward_step,
                            forward_step_batch=self._jacobi_forward_step_batch,
                            eos_token_id=self._jacobi_config.eos,
                            pad_token_id=self._jacobi_config.pad,
                            vocab_size=self.config.hf_config.vocab_size,
                        )
                        print(f"[RANK {self.rank}] [INIT] JacobiDecoderNonGreedyOnPolicy initialized: eos={self._jacobi_config.eos}, pad={self._jacobi_config.pad}, vocab_size={self.config.hf_config.vocab_size}", flush=True)
                    elif all_greedy:
                        # All jacobi sequences have temp == 0 → use greedy decoder
                        self.jacobi_decoder = JacobiDecoder(
                            block_manager=self.block_manager,
                            forward_step=self._jacobi_forward_step,
                            forward_step_batch=self._jacobi_forward_step_batch,
                            eos_token_id=self._jacobi_config.eos,
                            pad_token_id=self._jacobi_config.pad,
                            vocab_size=self.config.hf_config.vocab_size,
                        )
                        print(f"[RANK {self.rank}] [INIT] JacobiDecoder (greedy) initialized: eos={self._jacobi_config.eos}, pad={self._jacobi_config.pad}, vocab_size={self.config.hf_config.vocab_size}", flush=True)
                    elif all_nongreedy:
                        # All jacobi sequences have temp > 0 → use non-greedy decoder
                        from inference_engine.engine.jacobi_decoding_nongreedy import JacobiDecoderNonGreedy
                        self.jacobi_decoder = JacobiDecoderNonGreedy(
                            block_manager=self.block_manager,
                            forward_step=self._jacobi_forward_step,
                            forward_step_batch=self._jacobi_forward_step_batch,
                            eos_token_id=self._jacobi_config.eos,
                            pad_token_id=self._jacobi_config.pad,
                            vocab_size=self.config.hf_config.vocab_size,
                        )
                        print(f"[RANK {self.rank}] [INIT] JacobiDecoderNonGreedy initialized: eos={self._jacobi_config.eos}, pad={self._jacobi_config.pad}, vocab_size={self.config.hf_config.vocab_size}", flush=True)
                    else:
                        raise NotImplementedError(
                            f"Mixed temperature modes in Jacobi batch not supported. "
                            f"Got some sequences with temperature=0 (greedy) and some with temperature>0 (non-greedy). "
                            f"All sequences in a batch must use the same temperature mode. "
                            f"Temperatures: {temps}"
                        )

    
    def report_cuda_graph_usage(self):
        """Print CUDA graph usage statistics."""
        total = self.cuda_graph_hits + self.cuda_graph_misses
        if total == 0:
            return
        
        hit_rate = self.cuda_graph_hits / total * 100
        print("\n" + "="*70)
        print("CUDA GRAPH USAGE REPORT (NOT INCLUDING JACOBI PREFILL)")
        print("="*70)
        print(f"Hits:        {self.cuda_graph_hits:>6} ({hit_rate:>5.1f}%)")
        print(f"Misses:      {self.cuda_graph_misses:>6} ({100-hit_rate:>5.1f}%)")
        print(f"Total calls: {total:>6}")
        
        if hit_rate < 50:
            print(f"\n⚠️  WARNING: Low CUDA graph hit rate ({hit_rate:.1f}%)")
            print("   Most forwards are using eager mode (slow!)")
        elif hit_rate < 80:
            print(f"\n⚠️  Note: Moderate CUDA graph hit rate ({hit_rate:.1f}%)")
            print("   Some forwards are falling back to eager mode")
        else:
            print(f"\n✓ Good CUDA graph hit rate ({hit_rate:.1f}%)")
        print("="*70)

    def exit(self):
        if self.rank == 0:
            profiler = get_profiler()
            profiler.report()
        
        if self.rank == 0 and (self.cuda_graph_debug or (self.cuda_graph_hits + self.cuda_graph_misses) > 0):
            self.report_cuda_graph_usage()
        
        from inference_engine.utils.context import clear_jacobi_context_cache
        clear_jacobi_context_cache()
        
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
            if hasattr(self, 'jacobi_graphs'):
                del self.jacobi_graphs, self.jacobi_graph_pool, self.jacobi_graph_vars
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        print(f"[RANK {self.rank}] [LOOP] START: Worker entering main loop, waiting for commands via shared memory", flush=True)
        loop_iteration = 0
        while True:
            loop_iteration += 1
            print(f"[RANK {self.rank}] [LOOP] Iteration {loop_iteration}: Waiting for command from shared memory...", flush=True)
            method_name, args = self.read_shm()
            print(f"[RANK {self.rank}] [LOOP] Received command: {method_name} with {len(args)} args", flush=True)
            self.call(method_name, *args)
            if method_name == "exit":
                print(f"[RANK {self.rank}] [LOOP] Received exit command, breaking loop", flush=True)
                break
        print(f"[RANK {self.rank}] [LOOP] END: Worker exiting main loop", flush=True)

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

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def _estimate_jacobi_buffer_size(self):
        """Estimate memory needed for Jacobi buffers."""
        config = self.config
        
        if not config.jacobi_enabled:
            return 0
        
        max_jacobi_batch = config.max_num_seqs
        max_block_len = getattr(config, 'jacobi_max_block_len', 64)
        jacobi_buffer = 1024
        max_blocks_per_seq = (config.max_model_len + jacobi_buffer + self.block_size - 1) // self.block_size
        max_total_tokens = max_jacobi_batch * max_block_len
        
        # Estimate buffer sizes
        # Input tensors: input_ids, positions, slot_mapping
        input_buffers = max_total_tokens * (8 + 8 + 4)  # int64 + int64 + int32
        
        # Sequence metadata: cu_seqlens_q, cu_seqlens_k, cache_seqlens
        seq_metadata = (max_jacobi_batch + 1) * 4 * 2 + max_jacobi_batch * 4  # int32
        
        # Block tables: max_batch * max_blocks_per_seq
        block_tables = max_jacobi_batch * max_blocks_per_seq * 4  # int32
        
        # Position/block indices and offsets
        pos_buffers = max_block_len * 8 * 3  # int64 * 3
        
        # Block table scratch
        block_scratch = max_blocks_per_seq * 4  # int32
        
        # Base range
        base_range = max_block_len * 8  # int64
        
        # Lookup tables: common_block_lens * block_size * (block_indices + offsets)
        # Each LUT entry: (block_indices, offsets) where each is max_block_len * 8 bytes
        common_block_lens = [32, 48, 64, 96, 128]
        lut_size = 0
        for L in common_block_lens:
            lut_size += self.block_size * (L * 8 + L * 4)  # block_indices (int64) + offsets (int32)
        
        total_estimate = input_buffers + seq_metadata + block_tables + pos_buffers + block_scratch + base_range + lut_size
        
        # Add 50% safety margin (increased from 20% to be more conservative)
        # This accounts for:
        # - Dynamic allocations during forward passes
        # - Temporary tensors in attention mechanisms
        # - Memory fragmentation
        # - Peak memory usage during batched operations
        total_estimate = int(total_estimate * 1.5)
        
        # Ensure minimum reserve of 1GB (increased from 500MB)
        # This provides a safety buffer for unexpected allocations
        min_reserve = 1024 * 1024 * 1024
        return max(total_estimate, min_reserve)
    
    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        
        # Calculate available memory for KV cache
        # Use current allocated bytes (not peak) since peak is historical
        # Available = total * utilization - current_allocated
        max_memory_for_kv = total * config.gpu_memory_utilization
        available_memory = max_memory_for_kv - current
        
        # Reserve memory for Jacobi buffers if Jacobi is enabled
        jacobi_reserve = self._estimate_jacobi_buffer_size()
        if jacobi_reserve > 0:
            # Ensure we have enough memory for Jacobi buffers
            if free < jacobi_reserve:
                raise RuntimeError(
                    f"Insufficient free GPU memory for Jacobi buffers. "
                    f"Free: {free / 1e9:.2f} GB, "
                    f"Required: {jacobi_reserve / 1e9:.2f} GB. "
                    f"Try freeing GPU memory or reducing max_num_seqs/jacobi_max_block_len."
                )
            available_memory = available_memory - jacobi_reserve
        
        # Reserve memory for CUDA graph capture (can be substantial)
        # CUDA graphs need memory to store the captured graph and temporary buffers
        cuda_graph_reserve = 2 * 1024 * 1024 * 1024  # 2GB reserve for CUDA graphs
        available_memory = available_memory - cuda_graph_reserve
        
        # Also ensure we don't exceed free memory (accounting for all reserves)
        available_memory = min(available_memory, free - jacobi_reserve - cuda_graph_reserve)
        
        # Ensure available_memory is non-negative
        available_memory = max(available_memory, 0)
        
        # Calculate KV cache blocks
        calculated_blocks = int(available_memory) // block_bytes
        
        # Reserve at least 80% of total memory for:
        # - CUDA graph capture (already reserved above)
        # - Forward pass activations
        # - Temporary tensors
        # - Memory fragmentation buffer
        max_kv_memory = total * 0.8  # Use at most 80% of GPU for KV cache
        max_kv_blocks = int(max_kv_memory) // block_bytes
        config.num_kvcache_blocks = min(calculated_blocks, max_kv_blocks)
        
        # safety: Hard cap at 40000 blocks regardless of calculations
        # This prevents any single rank from allocating excessive KV cache
        hard_max_blocks = 40000
        if config.num_kvcache_blocks > hard_max_blocks:
            print(f"[RANK {self.rank}] [KV_CACHE] ⚠️  WARNING: Calculated blocks ({config.num_kvcache_blocks}) exceeds hard maximum ({hard_max_blocks}). Capping to {hard_max_blocks}.", flush=True)
            config.num_kvcache_blocks = hard_max_blocks
        
        # Add debug logging
        print(f"[RANK {self.rank}] [KV_CACHE] Memory stats:", flush=True)
        print(f"[RANK {self.rank}] [KV_CACHE]   Total GPU memory: {total / 1e9:.2f} GB", flush=True)
        print(f"[RANK {self.rank}] [KV_CACHE]   Free memory: {free / 1e9:.2f} GB", flush=True)
        print(f"[RANK {self.rank}] [KV_CACHE]   Used memory: {used / 1e9:.2f} GB", flush=True)
        print(f"[RANK {self.rank}] [KV_CACHE]   Current allocated: {current / 1e9:.2f} GB", flush=True)
        print(f"[RANK {self.rank}] [KV_CACHE]   GPU memory utilization: {config.gpu_memory_utilization}", flush=True)
        print(f"[RANK {self.rank}] [KV_CACHE]   Max memory for KV: {max_memory_for_kv / 1e9:.2f} GB", flush=True)
        if jacobi_reserve > 0:
            print(f"[RANK {self.rank}] [KV_CACHE]   Reserved for Jacobi buffers: {jacobi_reserve / 1e9:.2f} GB", flush=True)
        print(f"[RANK {self.rank}] [KV_CACHE]   Reserved for CUDA graphs: {cuda_graph_reserve / 1e9:.2f} GB", flush=True)
        print(f"[RANK {self.rank}] [KV_CACHE]   Available for KV cache: {available_memory / 1e9:.2f} GB", flush=True)
        print(f"[RANK {self.rank}] [KV_CACHE]   Block bytes: {block_bytes / 1e6:.2f} MB", flush=True)
        print(f"[RANK {self.rank}] [KV_CACHE]   Calculated num_kvcache_blocks (before cap): {calculated_blocks}", flush=True)
        print(f"[RANK {self.rank}] [KV_CACHE]   Max KV blocks (50% cap): {max_kv_blocks}", flush=True)
        print(f"[RANK {self.rank}] [KV_CACHE]   Final num_kvcache_blocks: {config.num_kvcache_blocks}", flush=True)
        print(f"[RANK {self.rank}] [KV_CACHE] ✓ Allocated {config.num_kvcache_blocks} KV cache blocks ({config.num_kvcache_blocks * block_bytes / 1e9:.2f} GB)", flush=True)
        
        if config.num_kvcache_blocks <= 0:
            raise RuntimeError(
                f"Insufficient GPU memory for KV cache allocation. "
                f"Available: {available_memory / 1e9:.2f} GB, "
                f"Required per block: {block_bytes / 1e6:.2f} MB, "
                f"Would need at least {block_bytes / 1e9:.2f} GB for 1 block. "
                f"Try reducing gpu_memory_utilization (current: {config.gpu_memory_utilization}) "
                f"or freeing GPU memory."
            )
        
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        
        if config.jacobi_enabled:
            self._allocate_jacobi_buffers()
        else:
            self.jacobi_buffers = None
    
    def _allocate_kv_cache_with_size(self):
        """Allocate KV cache with a pre-determined size (used by non-rank-0 workers)."""
        config = self.config
        hf_config = config.hf_config
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        print(f"[RANK {self.rank}] [KV_CACHE] Allocating with pre-determined size: {config.num_kvcache_blocks} blocks", flush=True)
        print(f"[RANK {self.rank}] [KV_CACHE] ✓ Allocated {config.num_kvcache_blocks} KV cache blocks ({config.num_kvcache_blocks * block_bytes / 1e9:.2f} GB)", flush=True)
        print(f"[KV_CACHE_SUMMARY]   Rank {self.rank}: {config.num_kvcache_blocks} blocks ({config.num_kvcache_blocks * block_bytes / 1e9:.2f} GB)", flush=True)
        assert config.num_kvcache_blocks > 0, f"num_kvcache_blocks must be > 0, got {config.num_kvcache_blocks}"
        
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        
        if config.jacobi_enabled:
            self._allocate_jacobi_buffers()
        else:
            self.jacobi_buffers = None
    
    def _allocate_jacobi_buffers(self):
        """Pre-allocate GPU buffers for Jacobi decoding."""
        config = self.config
        
        max_jacobi_batch = config.max_num_seqs
        max_block_len = getattr(config, 'jacobi_max_block_len', 64)
        jacobi_buffer = 1024

        max_blocks_per_seq = (config.max_model_len + jacobi_buffer + self.block_size - 1) // self.block_size
        
        print(f"[JACOBI] max model len: {config.max_model_len}")
        print(f"[JACOBI] block size: {self.block_size}")
        print(f"[JACOBI] jacobi buffer: {jacobi_buffer}")
        
        print(f"[JACOBI] max blocks per seq: {max_blocks_per_seq}")
        
        max_total_tokens = max_jacobi_batch * max_block_len
        
        self.jacobi_buffers = {
            # Input tensors
            'input_ids': torch.zeros(max_total_tokens, dtype=torch.int64, device='cuda'),
            'positions': torch.zeros(max_total_tokens, dtype=torch.int64, device='cuda'),
            'slot_mapping': torch.zeros(max_total_tokens, dtype=torch.int32, device='cuda'),
            
            'cu_seqlens_q': torch.zeros(max_jacobi_batch + 1, dtype=torch.int32, device='cuda'),
            'cu_seqlens_k': torch.zeros(max_jacobi_batch + 1, dtype=torch.int32, device='cuda'),
            
            'cache_seqlens': torch.zeros(max_jacobi_batch, dtype=torch.int32, device='cuda'),
            'block_tables': torch.full((max_jacobi_batch, max_blocks_per_seq), -1, dtype=torch.int32, device='cuda'),
            
            'pos_indices': torch.zeros(max_block_len, dtype=torch.int64, device='cuda'),
            'block_indices': torch.zeros(max_block_len, dtype=torch.int64, device='cuda'),
            'offsets_in_block': torch.zeros(max_block_len, dtype=torch.int64, device='cuda'),
            'block_table_scratch': torch.zeros(max_blocks_per_seq, dtype=torch.int32, device='cuda'),
            'base_range': torch.arange(max_block_len, dtype=torch.int64, device='cuda'),
            
            'max_batch': max_jacobi_batch,
            'max_block_len': max_block_len,
            'max_blocks_per_seq': max_blocks_per_seq,
        }
        
        print(f"[JACOBI] Pre-allocated GPU buffers: batch={max_jacobi_batch}, block_len={max_block_len}, "
              f"blocks_per_seq={max_blocks_per_seq}")
        
        common_block_lens = [32, 48, 64, 96, 128]
        self.slot_mapping_lut = {}
        
        for L in common_block_lens:
            for start_offset in range(self.block_size):
                pos_local = torch.arange(L, dtype=torch.int64, device='cuda') + start_offset
                
                block_indices = torch.div(pos_local, self.block_size, rounding_mode='floor').int()
                offsets = (pos_local - block_indices.long() * self.block_size).int()
                
                self.slot_mapping_lut[(L, start_offset)] = (block_indices, offsets)
        
        print(f"[JACOBI] Pre-computed {len(self.slot_mapping_lut)} slot mapping LUT patterns")

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures
    
    def _jacobi_prefill_with_drafting(self, seqs: list[Sequence]) -> list[list[int]]:
        """Prefill with drafting: forward prompt + draft, extract predictions, trim KV."""
        import random
        import torch.nn.functional as F
        import os
        
        debug = os.environ.get("JACOBI_DEBUG", "0") == "1"
        
        B = len(seqs)
        
        block_lens = []
        for seq in seqs:
            sp = getattr(seq, "sampling_params", None)
            block_len = getattr(sp, "jacobi_block_len", 64) if sp else 64
            block_lens.append(block_len)
        
        prompt_lens = [len(seq) for seq in seqs]
        
        draft_tokens_batch = []
        for i, seq in enumerate(seqs):
            draft = [random.choice(seq.token_ids) for _ in range(block_lens[i])]
            draft_tokens_batch.append(draft)
            for tok in draft:
                seq.token_ids.append(tok)
                seq.num_tokens += 1
        
        if debug:
            print(f"  [PREFILL_DRAFT] Forward prompt + draft: B={B}")
            for i, seq in enumerate(seqs):
                print(f"    seq_{i}: prompt_len={prompt_lens[i]}, block_len={block_lens[i]}, "
                      f"total_len={len(seq)}, num_cached={seq.num_cached_tokens}, draft={draft_tokens_batch[i][:5]}...")
        
        # Only rank 0 manages block allocation (workers don't have block_manager)
        if self.rank == 0 or self.world_size == 1:
            for seq in seqs:
                blocks_needed = seq.num_blocks
                blocks_have = len(seq.block_table)
                while blocks_have < blocks_needed:
                    if not self.block_manager.free_block_ids:
                        raise RuntimeError(f"Not enough free blocks for Jacobi prefill draft: need {blocks_needed}, have {blocks_have}")
                    block_id = self.block_manager.free_block_ids[0]
                    self.block_manager._allocate_block(block_id)
                    seq.block_table.append(block_id)
                    blocks_have += 1
                    if debug:
                        print(f"    [PREFILL_DRAFT] Allocated block {block_id} for seq, now have {blocks_have}/{blocks_needed} blocks")
        
        # Broadcast block tables from rank 0 to all workers
        if self.world_size > 1:
            import torch.distributed as dist

            # Calculate max_len on rank 0 after block allocation
            if self.rank == 0:
                max_len = max(len(seq.block_table) for seq in seqs)
            else:
                max_len = 0  # Will be updated by broadcast

            # Broadcast max_len first
            max_len_tensor = torch.tensor([max_len], dtype=torch.int32, device='cuda')
            dist.broadcast(max_len_tensor, src=0)
            max_len = max_len_tensor.item()

            # Prepare block tables as tensor for broadcast
            if self.rank == 0:
                block_tables_data = []
                for seq in seqs:
                    padded = seq.block_table + [-1] * (max_len - len(seq.block_table))
                    block_tables_data.extend(padded)
                block_tables_tensor = torch.tensor(block_tables_data, dtype=torch.int32, device='cuda')
            else:
                block_tables_tensor = torch.zeros(len(seqs) * max_len, dtype=torch.int32, device='cuda')

            # Broadcast block tables
            dist.broadcast(block_tables_tensor, src=0)

            # Workers update their sequences with received block tables
            if self.rank > 0:
                for i, seq in enumerate(seqs):
                    start_idx = i * max_len
                    end_idx = (i + 1) * max_len
                    block_table = block_tables_tensor[start_idx:end_idx].cpu().tolist()
                    # Remove padding
                    seq.block_table = [b for b in block_table if b != -1]
        
        input_ids, positions = self.prepare_prefill(seqs)
        
        if debug:
            print(f"  [PREFILL_DRAFT] input_ids.shape={input_ids.shape}, positions.shape={positions.shape}")
        
        import torch.nn.functional as F
        with torch.inference_mode():
            hidden_states = self.model(input_ids, positions)
            logits = F.linear(hidden_states, self.model.lm_head.weight)
            # Free hidden_states immediately - we don't need it anymore
            del hidden_states
            torch.cuda.empty_cache()
        
        if self.world_size > 1:
            lm_head = self.model.lm_head
            if lm_head.tp_rank == 0:
                if self._all_logits_buffer is not None and self._all_logits_buffer[0].shape[0] >= logits.shape[0]:
                    all_logits = [buf[:logits.shape[0], :logits.shape[1]] for buf in self._all_logits_buffer]
                else:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    all_logits = []
                    for i in range(self.world_size):
                        torch.cuda.empty_cache()
                        all_logits.append(torch.empty_like(logits))
                    self._all_logits_buffer = all_logits
            else:
                all_logits = None
            torch.distributed.gather(logits, all_logits, 0)
            if lm_head.tp_rank == 0:
                logits = torch.cat(all_logits, -1)
        
        if debug:
            print(f"  [PREFILL_DRAFT] logits.shape={logits.shape}")
        
        reset_context()
        
        prefill_drafts = []
        offset = 0
        for i, seq in enumerate(seqs):
            prompt_len = prompt_lens[i]
            block_len = block_lens[i]
            total_len = prompt_len + block_len
            
            start_idx = offset + prompt_len - 1
            end_idx = offset + prompt_len + block_len - 1
            
            if debug:
                print(f"  [PREFILL_DRAFT] Extracting seq_{i}: offset={offset}, start_idx={start_idx}, end_idx={end_idx}, "
                      f"logits.shape[0]={logits.shape[0]}")
            
            if self.rank == 0 or self.world_size == 1:
                draft_logits = logits[start_idx:end_idx]
                if debug:
                    print(f"  [PREFILL_DRAFT] draft_logits.shape={draft_logits.shape}")
                greedy_draft = torch.argmax(draft_logits, dim=-1).tolist()
                prefill_drafts.append(greedy_draft)
            else:
                prefill_drafts.append([])
            
            offset += total_len
        
        if debug and (self.rank == 0 or self.world_size == 1):
            print(f"  [PREFILL_DRAFT] Extracted greedy drafts:")
            for i, draft in enumerate(prefill_drafts):
                print(f"    seq_{i}: draft={draft[:10]}{'...' if len(draft) > 10 else ''}")
        
        for i, seq in enumerate(seqs):
            block_len = block_lens[i]
            prompt_len = prompt_lens[i]
            
            seq.token_ids = seq.token_ids[:prompt_len]
            seq.num_tokens = prompt_len
            seq.last_token = seq.token_ids[-1]
            
            # Only rank 0 manages block trimming (workers don't have block_manager)
            if self.rank == 0 or self.world_size == 1:
                self.block_manager.trim_kv_only_fast(seq, block_len)
            
            seq.num_cached_tokens = prompt_len
            seq.draft_tokens = None
            
            if self.rank == 0 or self.world_size == 1:
                seq._prefill_draft = prefill_drafts[i]
            else:
                seq._prefill_draft = None
        
        if debug:
            print(f"  [PREFILL_DRAFT] After trim: seq_lens={[len(s) for s in seqs]}, num_cached={[s.num_cached_tokens for s in seqs]}")
            print(f"  [PREFILL_DRAFT] Prefill complete, drafts stored for first iteration")
        
        # Clean up temporary tensors to prevent memory accumulation
        if 'hidden_states' in locals():
            del hidden_states
        if 'logits' in locals() and (self.rank == 0 or self.world_size == 1):
            del logits
        # Don't delete all_logits if we stored it in self._all_logits_buffer for reuse
        if 'all_logits' in locals() and (self._all_logits_buffer is None or all_logits is not self._all_logits_buffer):
            del all_logits
        torch.cuda.empty_cache()
        
        return [[] for _ in seqs]

    def _get_slot_mapping_pattern(self, seq_len: int, draft_len: int) -> tuple:
        """Get slot mapping pattern from pre-computed LUT when possible."""
        start_offset = (seq_len - 1) % self.block_size
        lut_key = (draft_len, start_offset)
        
        if lut_key in self.slot_mapping_lut:
            relative_block_indices, offsets = self.slot_mapping_lut[lut_key]
            base_block = (seq_len - 1) // self.block_size
            block_indices = relative_block_indices + base_block
            return (block_indices, offsets)
        
        cache_key = (seq_len, draft_len)
        
        if cache_key not in self.slot_mapping_cache:
            pos_indices = torch.arange(draft_len, dtype=torch.int32, device='cuda') + (seq_len - 1)
            
            block_indices = torch.div(pos_indices, self.block_size, rounding_mode='floor').int()
            offsets = torch.remainder(pos_indices, self.block_size).int()
            
            self.slot_mapping_cache[cache_key] = (block_indices, offsets)
        
        return self.slot_mapping_cache[cache_key]
    
    def _signal_jacobi_loop_done(self):
        """Signal workers that the Jacobi decoder loop is done."""
        if self.world_size <= 1:
            return
        CMD_EXIT = -1
        cmd = torch.full((1,), CMD_EXIT, dtype=torch.int32, device='cuda')
        torch.cuda.synchronize()
        dist.broadcast(cmd, src=0)
        torch.cuda.synchronize()
    
    def _sync_jacobi_ranks(self, mode: str = ""):
        """Synchronize ranks before Jacobi decode. Rank 0 sends ready signal that workers consume in their loop."""
        if self.world_size <= 1:
            return
        
        dist.barrier()
        
        # Rank 0 sends ready handshake, but workers will consume it as their FIRST command
        # in the worker loop, not here. This prevents double broadcast consumption.
        if self.rank == 0:
            READY_SIGNAL = 42
            ready_cmd = torch.full((1,), READY_SIGNAL, dtype=torch.int32, device='cuda')
            torch.cuda.synchronize()
            dist.broadcast(ready_cmd, src=0)
            torch.cuda.synchronize()
        # Workers do NOT call broadcast here - they'll receive the ready signal
        # as the first command in _worker_jacobi_forward_loop()
    
    def _worker_jacobi_forward_loop(self):
        """Worker loop: wait for forward pass signals from rank 0."""
        import torch.nn.functional as F
        import threading
        import time as time_module
        
        print(f"[RANK {self.rank}] [WORKER_LOOP] Entering worker loop, waiting for commands", flush=True)
        
        loop_iteration = 0
        CMD_READY = 42
        CMD_FORWARD = 1
        CMD_EXIT = -1
        
        while True:
            loop_iteration += 1
            cmd = torch.zeros(1, dtype=torch.int32, device='cuda')
            
            # Add timeout watchdog for debugging
            broadcast_completed = [False]
            def timeout_watchdog():
                for wait_time in [30, 60, 120, 300, 600]:
                    time_module.sleep(wait_time)
                    if not broadcast_completed[0]:
                        # print(f"[RANK {self.rank}] [WORKER_LOOP] (iteration {loop_iteration})", flush=True)
                        continue
                    else:
                        return
            
            watchdog_thread = threading.Thread(target=timeout_watchdog, daemon=True)
            watchdog_thread.start()
            
            start_time = time_module.time()
            dist.broadcast(cmd, src=0)
            torch.cuda.synchronize()
            broadcast_completed[0] = True
            elapsed = time_module.time() - start_time
            
            cmd_val = cmd.item()
            
            # First iteration: expect ready signal from _sync_jacobi_ranks()
            if loop_iteration == 1:
                if cmd_val != CMD_READY:
                    raise RuntimeError(
                        f"Rank {self.rank} expected ready signal {CMD_READY} on first iteration, got {cmd_val}. "
                        f"This indicates a broadcast synchronization issue."
                    )
                print(f"[RANK {self.rank}] [WORKER_LOOP] Ready signal verified, waiting for forward passes", flush=True)
                continue
            
            # Treat 0 as stale/no-op (default from torch.zeros), not as exit
            if cmd_val == 0:
                print(f"[RANK {self.rank}] [WORKER_LOOP] Received stale command (0), ignoring and continuing", flush=True)
                continue
            
            if cmd_val == CMD_EXIT:
                print(f"[RANK {self.rank}] [WORKER_LOOP] Received exit command after {loop_iteration} iterations", flush=True)
                break
            
            if cmd_val != CMD_FORWARD:
                raise RuntimeError(
                    f"Rank {self.rank} received unexpected command {cmd_val}. "
                    f"Expected {CMD_FORWARD} (forward) or {CMD_EXIT} (exit)."
                )
            
            # Receive forward pass inputs
            meta = torch.zeros(2, dtype=torch.int32, device='cuda')
            dist.broadcast(meta, src=0)
            B, L = meta[0].item(), meta[1].item()
            
            total_tokens = B * L
            input_ids = torch.zeros(total_tokens, dtype=torch.int64, device='cuda')
            positions = torch.zeros(total_tokens, dtype=torch.int64, device='cuda')
            slot_mapping = torch.zeros(total_tokens, dtype=torch.int32, device='cuda')
            dist.broadcast(input_ids, src=0)
            dist.broadcast(positions, src=0)
            
            # Receive slot_mapping
            dist.broadcast(slot_mapping, src=0)
            
            # Receive block tables
            max_blocks = torch.zeros(1, dtype=torch.int32, device='cuda')
            dist.broadcast(max_blocks, src=0)
            max_blocks_val = max_blocks.item()
            block_tables = torch.zeros(B, max_blocks_val, dtype=torch.int32, device='cuda')
            dist.broadcast(block_tables, src=0)
            
            # Receive cache_seqlens
            cache_seqlens = torch.zeros(B, dtype=torch.int32, device='cuda')
            dist.broadcast(cache_seqlens, src=0)
            
            # Set context for the forward pass
            from inference_engine.utils.context import get_or_create_jacobi_context, set_jacobi_context_active
            ctx = get_or_create_jacobi_context(B, L, device='cuda')
            ctx.slot_mapping = slot_mapping
            ctx.block_tables = block_tables
            ctx.cache_seqlens = cache_seqlens
            ctx.seqlen_q = L
            ctx.is_jacobi_graphed = False
            set_jacobi_context_active(ctx)
            
            # Do the forward pass (NCCL collectives sync automatically)
            # Wrap in inference_mode to avoid gradient tracking and torch.compile issues
            with torch.inference_mode():
                hidden_states = self.model(input_ids, positions)
                
                # Participate in the logits gather operation
                logits = F.linear(hidden_states, self.model.lm_head.weight)
            
            lm_head = self.model.lm_head
            # Workers participate in gather but don't receive the result
            torch.distributed.gather(logits, None, 0)
            
            reset_context()
    
    def _jacobi_forward_step(self, seq: Sequence, draft_tokens: torch.Tensor) -> torch.Tensor:
        """Forward step callback for JacobiDecoder (single sequence)."""
        return self._jacobi_forward_step_batch([seq], draft_tokens)
    
    def _jacobi_forward_step_batch(self, seqs: list[Sequence], draft_tokens_batch: torch.Tensor) -> torch.Tensor:
        """Batched forward pass for Jacobi decoding with seed token."""
        import os
        debug = os.environ.get("JACOBI_DEBUG", "0") == "1"
        
        B = len(seqs)
        L = draft_tokens_batch.size(1)
        
        # NCCL DEBUG: Entry point
        
        if L < 2:
            raise ValueError("Draft must have at least 2 tokens (seed + 1 speculative)")
        
        for i, seq in enumerate(seqs):
            seq.draft_tokens_gpu = draft_tokens_batch[i]
        
        original_seq_lens = [len(seq) for seq in seqs]
        original_num_cached = [seq.num_cached_tokens for seq in seqs]
        
        if debug:
            print(f"    [FORWARD_STEP_BATCH] B={B}, L={L}, seq_lens={original_seq_lens}, num_cached={original_num_cached}")
            print(f"    [FORWARD_STEP_BATCH] Draft (seed + spec): {draft_tokens_batch[0, :min(10, L)].tolist()}...")
        
        # Verify seed matches last committed token
        for i, seq in enumerate(seqs):
            expected_seed = seq.token_ids[-1]
            actual_seed = draft_tokens_batch[i, 0].item()
            if expected_seed != actual_seed:
                raise ValueError(f"Seed mismatch: seq[-1]={expected_seed}, draft[0]={actual_seed}")
        
        profiler = get_profiler()
        profiler.start("jacobi.block_alloc")
        # Only rank 0 manages block allocation (workers don't have block_manager)
        if self.rank == 0 or self.world_size == 1:
            for i, seq in enumerate(seqs):
                S = original_seq_lens[i]
                current_blocks = len(seq.block_table)
                total_tokens = S + (L - 1)
                blocks_needed = (total_tokens + self.block_size - 1) // self.block_size
                
                committed_blocks = (S + self.block_size - 1) // self.block_size
                spec_blocks_needed = blocks_needed - committed_blocks
                
                if current_blocks >= blocks_needed:
                    if debug:
                        print(f"    [BLOCK_REUSE] seq_{i}: Reusing {blocks_needed} blocks (had {current_blocks})")
                    if current_blocks > blocks_needed:
                        seq.block_table = seq.block_table[:blocks_needed]
                        seq.block_table_version += 1
                else:
                    blocks_to_allocate = blocks_needed - current_blocks
                    
                    if debug:
                        print(f"    [BLOCK_ALLOC] seq_{i}: Allocating {blocks_to_allocate} new blocks (no-clear)")
                    
                    for _ in range(blocks_to_allocate):
                        if not self.block_manager.can_append(seq):
                            raise RuntimeError("Cannot allocate blocks for draft tokens")
                        block_id = self.block_manager.free_block_ids[0]
                        
                        block = self.block_manager._allocate_block_no_clear(block_id)
                        seq.block_table.append(block_id)
                        seq.block_table_version += 1
                
                seq.num_permanent_spec_blocks = max(seq.num_permanent_spec_blocks, spec_blocks_needed)
        profiler.stop("jacobi.block_alloc")
        
        # Note: Block tables are broadcast later in the forward pass sequence (after command=1),
        # not here. Workers receive them in _worker_jacobi_forward_loop() at line ~1131.
        
        profiler.start("jacobi.buffer_fill")
        total_tokens = B * L
        buffers = self.jacobi_buffers
        
        input_ids = buffers['input_ids'][:total_tokens]
        positions = buffers['positions'][:total_tokens]
        slot_mapping = buffers['slot_mapping'][:total_tokens]
        cu_seqlens_q = buffers['cu_seqlens_q'][:B+1]
        cu_seqlens_k = buffers['cu_seqlens_k'][:B+1]
        cache_seqlens = buffers['cache_seqlens'][:B]
        block_tables = buffers['block_tables'][:B]
        
        cu_seqlens_q[0] = 0
        cu_seqlens_k[0] = 0
        
        base_range = buffers['base_range'][:L]
        
        for i, seq in enumerate(seqs):
            S = original_seq_lens[i]
            if S < 1:
                raise ValueError(f"Sequence {i} has invalid length S={S}. Must be >= 1.")
            
            offset = i * L
            
            input_ids[offset:offset+L] = draft_tokens_batch[i]
            
            positions[offset:offset+L] = base_range + (S - 1)
            
            num_blocks = len(seq.block_table)
            if (seq.block_table_gpu is None or 
                seq.block_table_version != getattr(seq, '_cached_bt_version', -1) or
                len(seq.block_table) != seq.block_table_gpu.shape[0]):
                seq.block_table_gpu = torch.tensor(seq.block_table, device='cuda', dtype=torch.int32)
                seq._cached_bt_version = seq.block_table_version
                if debug:
                    print(f"    [BT_CACHE] seq_{i}: Updated block_table_gpu cache ({num_blocks} blocks)")
            
            max_block_cols = block_tables.shape[1]
            copy_blocks = min(num_blocks, max_block_cols)
            if num_blocks > max_block_cols:
                raise RuntimeError(
                    f"Sequence {i} needs {num_blocks} blocks but buffer only has {max_block_cols}. "
                    f"Increase max_model_len or jacobi_buffer. Current max_model_len={self.config.max_model_len}, "
                    f"sequence length S={S}, block_len L={L}, "
                    f"Note prompt length is part of sequence length S and its size is {original_seq_lens[i]}"
                )
            block_tables[i, :copy_blocks] = seq.block_table_gpu[:copy_blocks]
            block_tables[i, copy_blocks:] = -1
            
            block_indices_pattern, offsets_pattern = self._get_slot_mapping_pattern(S, L)
            slot_mapping[offset:offset+L] = (
                block_tables[i, block_indices_pattern] * self.block_size + offsets_pattern
            )
            
            cu_seqlens_q[i+1] = cu_seqlens_q[i] + L
            cu_seqlens_k[i+1] = cu_seqlens_k[i] + (S - 1) + L
            
            # Ensure cache_seqlens value is non-negative and within reasonable bounds
            cache_seqlen_val = S - 1
            if cache_seqlen_val < 0:
                raise ValueError(f"Sequence {i} has invalid cache_seqlen: S={S}, S-1={cache_seqlen_val}")
            cache_seqlens[i] = cache_seqlen_val
        
        if debug:
            print(f"    [FORWARD_STEP_BATCH] input_ids.shape={input_ids.shape}, positions.shape={positions.shape}")
            print(f"    [FORWARD_STEP_BATCH] cu_seqlens_q={cu_seqlens_q.tolist()}, cu_seqlens_k={cu_seqlens_k.tolist()}")
            print(f"    [FORWARD_STEP_BATCH] Forwarding {L} tokens (seed + {L-1} speculative)")
            print(f"    [FA_PARAMS] positions: {positions.tolist()[:10]}{'...' if len(positions) > 10 else ''}")
            print(f"    [FA_PARAMS] slot_mapping: {slot_mapping.tolist()[:10]}{'...' if len(slot_mapping) > 10 else ''}")
            print(f"    [FA_PARAMS] cache_seqlens: {cache_seqlens.tolist()}")
        
        max_seqlen_k = max((original_seq_lens[i] - 1) + L for i in range(B))
        profiler.stop("jacobi.buffer_fill")
        
        # Signal workers to do forward pass (tensor parallelism sync)
        if self.world_size > 1 and self.rank == 0:
            # Command=1 means "do forward pass"
            cmd = torch.ones(1, dtype=torch.int32, device='cuda')
            torch.cuda.synchronize()
            dist.broadcast(cmd, src=0)
            torch.cuda.synchronize()
            
            # Broadcast metadata
            meta = torch.tensor([B, L], dtype=torch.int32, device='cuda')
            dist.broadcast(meta, src=0)
            
            # Broadcast inputs
            dist.broadcast(input_ids, src=0)
            dist.broadcast(positions, src=0)
            
            # Broadcast slot_mapping
            dist.broadcast(slot_mapping, src=0)
            
            # Broadcast block tables
            max_blocks = torch.tensor([block_tables.size(1)], dtype=torch.int32, device='cuda')
            dist.broadcast(max_blocks, src=0)
            dist.broadcast(block_tables, src=0)
            
            # Broadcast cache_seqlens
            dist.broadcast(cache_seqlens, src=0)
        
        profiler.start("jacobi.context_setup")
        use_jacobi_graph = (
            not self.enforce_eager and 
            hasattr(self, 'jacobi_graphs') and
            (B, L) in self.jacobi_graphs and
            B <= 512 and
            block_tables.size(1) <= self.jacobi_graph_vars["block_tables"].size(1)
        )
        
        if use_jacobi_graph:
            graph = self.jacobi_graphs[(B, L)]
            graph_vars = self.jacobi_graph_vars
            total_tokens = B * L
            
            graph_vars["input_ids"][:total_tokens] = input_ids
            graph_vars["positions"][:total_tokens] = positions
            graph_vars["slot_mapping"][:total_tokens] = slot_mapping
            graph_vars["cache_seqlens"][:B] = cache_seqlens
            graph_vars["block_tables"][:B, :block_tables.size(1)] = block_tables
            
            set_context(
                is_prefill=True,
                slot_mapping=graph_vars["slot_mapping"][:total_tokens],
                block_tables=graph_vars["block_tables"][:B],
                cache_seqlens=graph_vars["cache_seqlens"][:B],
                seqlen_q=L,
                is_jacobi_graphed=True
            )
            
            profiler.stop("jacobi.context_setup")
            
            profiler.start("jacobi.forward")
            graph.replay()
            profiler.stop("jacobi.forward")
            
            hidden_states = graph_vars["outputs"][:total_tokens]
            
            if self.cuda_graph_debug or debug:
                print(f"[CUDA_GRAPH] ✓ HIT - Jacobi graph: B={B}, L={L}")
            self.cuda_graph_hits += 1
            
        else:
            from inference_engine.utils.context import get_or_create_jacobi_context, set_jacobi_context_active
            
            ctx = get_or_create_jacobi_context(B, L, device='cuda')
            
            ctx.cu_seqlens_q.copy_(cu_seqlens_q)
            ctx.cu_seqlens_k.copy_(cu_seqlens_k)
            ctx.max_seqlen_q = L
            ctx.max_seqlen_k = max_seqlen_k
            ctx.slot_mapping = slot_mapping
            ctx.block_tables = block_tables
            ctx.cache_seqlens = cache_seqlens
            ctx.seqlen_q = L
            ctx.is_jacobi_graphed = False
            
            set_jacobi_context_active(ctx)
            profiler.stop("jacobi.context_setup")
            
            profiler.start("jacobi.forward")
            hidden_states = self.model(input_ids, positions)
            profiler.stop("jacobi.forward")
            
            if self.cuda_graph_debug or debug:
                if not hasattr(self, 'jacobi_graphs'):
                    reason = "graphs not captured"
                elif (B, L) not in self.jacobi_graphs:
                    reason = f"(B={B},L={L}) not captured"
                elif block_tables.size(1) > self.jacobi_graph_vars["block_tables"].size(1):
                    reason = f"block_table too large ({block_tables.size(1)} > {self.jacobi_graph_vars['block_tables'].size(1)})"
                else:
                    reason = "enforce_eager or other condition"
                print(f"[CUDA_GRAPH] ✗ MISS - Jacobi eager (cached ctx): {reason}")
            self.cuda_graph_misses += 1
        
        import torch.nn.functional as F
        profiler.start("jacobi.lm_head")
        logits = F.linear(hidden_states, self.model.lm_head.weight)
        
        if self.world_size > 1:
            lm_head = self.model.lm_head
            if lm_head.tp_rank == 0:
                if self._all_logits_buffer is not None and self._all_logits_buffer[0].shape[0] >= logits.shape[0]:
                    all_logits = [buf[:logits.shape[0], :logits.shape[1]] for buf in self._all_logits_buffer]
                else:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                    all_logits = []
                    for i in range(self.world_size):
                        torch.cuda.empty_cache()
                        all_logits.append(torch.empty_like(logits))
                    self._all_logits_buffer = all_logits
            else:
                all_logits = None
            torch.distributed.gather(logits, all_logits, 0)
            if lm_head.tp_rank == 0:
                logits = torch.cat(all_logits, -1)
        profiler.stop("jacobi.lm_head")
        
        reset_context()
        
        for seq in seqs:
            seq.num_cached_tokens = (len(seq) - 1) + L
        
        if debug:
            print(f"    [FORWARD_STEP_BATCH] After forward: num_cached={[s.num_cached_tokens for s in seqs]}")
        
        if self.rank == 0 or self.world_size == 1:
            vocab_size = logits.size(-1)
            logits = logits.view(B, L, vocab_size)
            return logits[:, :-1, :]
        else:
            return None

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            if self.cuda_graph_debug:
                reason = "prefill" if is_prefill else ("enforce_eager" if self.enforce_eager else "batch_size>512")
                print(f"[CUDA_GRAPH] ✗ MISS - Eager mode ({reason}): bs={input_ids.size(0)}, tokens={input_ids.size(0)}")
            self.cuda_graph_misses += 1
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            
            if context.context_lens is None:
                if self.cuda_graph_debug:
                    print(f"[CUDA_GRAPH] ✗ MISS - Varlen mode (Jacobi): bs={bs}, tokens={input_ids.numel()}")
                self.cuda_graph_misses += 1
                return self.model.compute_logits(self.model(input_ids, positions))
            
            max_captured_blocks = self.graph_vars["block_tables"].size(1)
            actual_blocks = context.block_tables.size(1)
            
            if actual_blocks > max_captured_blocks:
                if self.cuda_graph_debug:
                    print(f"[CUDA_GRAPH] ✗ MISS - Block table too large: {actual_blocks} > {max_captured_blocks} (bs={bs})")
                self.cuda_graph_misses += 1
                return self.model.compute_logits(self.model(input_ids, positions))
            
            if self.cuda_graph_debug:
                print(f"[CUDA_GRAPH] ✓ HIT - Using graph: bs={bs}, blocks={actual_blocks}/{max_captured_blocks}")
            self.cuda_graph_hits += 1
            
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

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int] | list[list[int]] | list[dict]:

        self._ensure_jacobi_decoder_initialized(seqs)
        
        jacobi_multiblock_seqs = [s for s in seqs if s.decode_strategy == "jacobi_multiblock_rejection_recycling"]
        if jacobi_multiblock_seqs:
            raise NotImplementedError(
                f"Decode strategy 'jacobi_multiblock_rejection_recycling' is not implemented. "
                f"Supported strategies: 'autoregressive', 'jacobi'"
            )
        
        jacobi_seqs = [s for s in seqs if s.decode_strategy == "jacobi"]
        autoregressive_seqs = [s for s in seqs if s.decode_strategy == "autoregressive"]
        
        if jacobi_seqs and len(jacobi_seqs) == len(seqs):
            # all Jacobi decoding requests
            if is_prefill:
                return self._jacobi_prefill_with_drafting(seqs)
            else:
                use_on_policy = any(getattr(s, 'jacobi_on_policy', False) for s in seqs)
                
                if use_on_policy:
                    self._sync_jacobi_ranks("on-policy")
                    if self.rank == 0 or self.world_size == 1:
                        # Rank 0 runs decoder, workers wait for forward pass signals
                        if not isinstance(self.jacobi_decoder, JacobiDecoderNonGreedyOnPolicy):
                            raise RuntimeError(
                                f"On-policy decoder not initialized correctly. "
                                f"Expected JacobiDecoderNonGreedyOnPolicy but got {type(self.jacobi_decoder)}"
                            )
                        

                        n_token_seq_len = getattr(seqs[0], 'jacobi_block_len', 64)
                        print(f"[RANK {self.rank}] [RUN] BEFORE generate_rollout_records_batch, n_token_seq_len={n_token_seq_len}, num_seqs={len(seqs)}", flush=True)
                        print(f"[RANK {self.rank}] [RUN] First seq details: max_tokens={getattr(seqs[0], 'max_tokens', 'N/A')}, temp={getattr(seqs[0], 'temperature', 'N/A')}", flush=True)
                        
                        rollout_records = self.jacobi_decoder.generate_rollout_records_batch(
                            seqs, 
                            n_token_seq_len=n_token_seq_len,
                            return_metrics=False
                        )
                        print(f"[RANK {self.rank}] [RUN] AFTER generate_rollout_records_batch, got {len(rollout_records) if rollout_records else 0} records", flush=True)
                        
                        if self.world_size > 1:
                            print(f"[RANK {self.rank}] [RUN] Sending loop done signal (normal path)", flush=True)
                            self._signal_jacobi_loop_done()
                        return rollout_records
                    else:
                        # Workers wait for forward pass signals from rank 0
                        print(f"[RANK {self.rank}] [RUN] Worker entering jacobi forward loop", flush=True)
                        self._worker_jacobi_forward_loop()
                        print(f"[RANK {self.rank}] [RUN] Worker exited jacobi forward loop", flush=True)
                        return None
                else:
                    self._sync_jacobi_ranks("regular")
                    if self.rank == 0 or self.world_size == 1:
                        # Rank 0 runs decoder
                        #print(f"[RANK {self.rank}] [RUN] Starting regular jacobi generation", flush=True)
                        
                        #print(f"[RANK {self.rank}] [RUN] Calling jacobi_decoder.generate_chunk_batch", flush=True)
                        
                        token_ids_batch = self.jacobi_decoder.generate_chunk_batch(seqs)
                        #print(f"[RANK {self.rank}] [RUN] Returned from jacobi_decoder.generate_chunk_batch", flush=True)
                        
                        # Signal workers that decoder loop is done
                        if self.world_size > 1:
                            self._signal_jacobi_loop_done()
                        return token_ids_batch
                    else:
                        # Workers wait for forward pass signals from rank 0
                        print(f"[RANK {self.rank}] [RUN] Worker entering jacobi forward loop", flush=True)
                        self._worker_jacobi_forward_loop()
                        print(f"[RANK {self.rank}] [RUN] Worker exited jacobi forward loop", flush=True)
                        return None
        elif jacobi_seqs:
            raise NotImplementedError(
                f"Mixed decode strategies in same batch not supported. "
                f"Got {len(jacobi_seqs)} Jacobi and {len(autoregressive_seqs)} autoregressive sequences."
            )
        else:
            # all autoregressive decoding requests
            input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
            temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
            logits = self.run_model(input_ids, positions, is_prefill)
            token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
            reset_context()
            return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        
        if config.jacobi_enabled:
            jacobi_buffer = 1024
            max_num_blocks_with_buffer = (config.max_model_len + jacobi_buffer + self.block_size - 1) // self.block_size
            max_num_blocks = max(max_num_blocks, max_num_blocks_with_buffer)
            print(f"[CUDAGRAPH] Jacobi enabled: capturing with {max_num_blocks} blocks (includes buffer for speculative tokens)")
        
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 3, 4, 5, 6, 16] + list(range(32, min(max_bs + 1, 64 + 1), 16))
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
    
    @torch.inference_mode()
    def capture_cudagraph_jacobi(self):
        """Capture CUDA graphs for Jacobi decoding mode."""
        config = self.config
        hf_config = config.hf_config
        
        max_bs = min(config.max_num_seqs, 256)
        max_block_len = getattr(config, 'jacobi_max_block_len', 64)
        jacobi_buffer = 1024
        max_num_blocks_with_buffer = (config.max_model_len + jacobi_buffer + self.block_size - 1) // self.block_size
        
        graph_bs = [1, 2, 3, 4, 5, 6, 16] + list(range(32, min(max_bs + 1, 64 + 1), 16))
        
        graph_block_lens = list(range(4, max_block_len + 1))
        
        print(f"[CUDAGRAPH_JACOBI] Capturing graphs for batch_sizes={graph_bs}, block_lens={graph_block_lens}")
        
        max_total_tokens = max_bs * max_block_len
        input_ids = torch.zeros(max_total_tokens, dtype=torch.int64)
        positions = torch.zeros(max_total_tokens, dtype=torch.int64)
        slot_mapping = torch.zeros(max_total_tokens, dtype=torch.int32)
        cache_seqlens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks_with_buffer, dtype=torch.int32)
        outputs = torch.zeros(max_total_tokens, hf_config.hidden_size)
        
        self.jacobi_graphs = {}
        self.jacobi_graph_pool = None
        
        for bs in reversed(graph_bs):
            for L in graph_block_lens:
                total_tokens = bs * L
                
                graph = torch.cuda.CUDAGraph()
                
                set_context(
                    is_prefill=True,
                    slot_mapping=slot_mapping[:total_tokens],
                    block_tables=block_tables[:bs],
                    cache_seqlens=cache_seqlens[:bs],
                    seqlen_q=L,
                    is_jacobi_graphed=True
                )
                
                outputs[:total_tokens] = self.model(input_ids[:total_tokens], positions[:total_tokens])
                
                with torch.cuda.graph(graph, self.jacobi_graph_pool):
                    outputs[:total_tokens] = self.model(input_ids[:total_tokens], positions[:total_tokens])
                
                if self.jacobi_graph_pool is None:
                    self.jacobi_graph_pool = graph.pool()
                
                self.jacobi_graphs[(bs, L)] = graph
                torch.cuda.synchronize()
                reset_context()
        
        self.jacobi_graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            cache_seqlens=cache_seqlens,
            block_tables=block_tables,
            outputs=outputs,
        )
        
        print(f"[CUDAGRAPH_JACOBI] Captured {len(self.jacobi_graphs)} Jacobi graphs")