from collections import deque
import os
import xxhash
import numpy as np

from inference_engine.engine.sequence import Sequence


# ============================================================================
# Profiler helper (import from model_runner when available)
# ============================================================================
def _get_profiler():
    """Get profiler instance if available and PROFILE=1."""
    if os.environ.get("PROFILE", "0") != "1":
        return None
    try:
        from inference_engine.engine.model_runner import get_profiler
        return get_profiler()
    except ImportError:
        return None


class Block:

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

    def __init__(self, num_blocks: int, block_size: int, kv_cache=None):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        self.kv_cache = kv_cache
        
        self._trim_buffers_initialized = False
        self._trim_base_range = None
        self._max_trim_size = 256  # Maximum tokens to trim at once

    def _init_trim_buffers(self):
        """Initialize pre-allocated buffers for fast KV trimming."""
        if self._trim_buffers_initialized:
            return
        
        import torch
        device = self.kv_cache.device if self.kv_cache is not None else 'cuda'
        
        self._trim_base_range = torch.arange(self._max_trim_size, dtype=torch.int64, device=device)
        self._trim_buffers_initialized = True

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        try:
            import sys
            for idx, t in enumerate(token_ids):
                if not isinstance(t, (int, np.integer)):
                    print(f"ERROR: token_ids[{idx}] has type {type(t)}, value: {t}", file=sys.stderr, flush=True)
                    if hasattr(t, 'shape'):
                        print(f"  It has shape: {t.shape}", file=sys.stderr, flush=True)
                    if hasattr(t, 'item'):
                        token_ids[idx] = int(t.item())
                    elif isinstance(t, (list, tuple)) and len(t) > 0:
                        raise ValueError(f"token_ids[{idx}] is a sequence with {len(t)} elements, not a single integer!")
            
            token_ids_int = [int(t) if not isinstance(t, int) else t for t in token_ids]
            h.update(np.array(token_ids_int, dtype=np.int64).tobytes())
        except (ValueError, TypeError) as e:
            import sys
            print(f"ERROR in compute_hash: len(token_ids)={len(token_ids)}", file=sys.stderr, flush=True)
            print(f"First few elements: {token_ids[:min(5, len(token_ids))]}", file=sys.stderr, flush=True)
            print(f"Types: {[type(t) for t in token_ids[:min(5, len(token_ids))]]}", file=sys.stderr, flush=True)
            raise ValueError(f"token_ids contains non-integer elements, error: {e}")
        return h.intdigest()

    def _clear_kv_cache_block(self, block_id: int):
        """Clear KV cache for a specific block."""
        if self.kv_cache is not None:
            import torch
            import os
            if os.environ.get("BLOCK_DEBUG", "0") == "1":
                print(f"  [KV_CLEAR] Clearing KV cache for block {block_id}")
            self.kv_cache[:, :, block_id, :, :, :] = 0
    
    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        profiler = _get_profiler()
        if profiler: profiler.start("block.alloc")
        self._clear_kv_cache_block(block_id)  # Clear stale KV cache data
        if profiler: profiler.stop("block.alloc")
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]
    
    def _allocate_block_no_clear(self, block_id: int) -> Block:
        """Allocate block WITHOUT clearing KV cache (used for Jacobi speculative blocks)."""
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        profiler = _get_profiler()
        if profiler: profiler.start("block.dealloc")
        self._clear_kv_cache_block(block_id)
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)
        if profiler: profiler.stop("block.dealloc")

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        import os
        debug = os.environ.get("JACOBI_DEBUG", "0") == "1"
        
        if debug and hasattr(seq, 'decode_strategy') and seq.decode_strategy == "jacobi":
            print(f"  [ALLOCATE] seq_id={seq.seq_id}, seq_len={len(seq)}, free_blocks={len(self.free_block_ids)}")
        
        assert not seq.block_table
        h = -1
        cache_miss = False
        
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)
        
        if debug and hasattr(seq, 'decode_strategy') and seq.decode_strategy == "jacobi":
            print(f"  [ALLOCATE] Allocated blocks: {seq.block_table}, free_blocks_remaining={len(self.free_block_ids)}")

    def deallocate(self, seq: Sequence):
        import os
        debug = os.environ.get("JACOBI_DEBUG", "0") == "1"
        
        if debug and hasattr(seq, 'decode_strategy') and seq.decode_strategy == "jacobi":
            print(f"  [DEALLOCATE] seq_id={seq.seq_id}, block_table={seq.block_table}, free_blocks_before={len(self.free_block_ids)}")
        
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()
        
        # Clear permanent speculative blocks counter
        if hasattr(seq, 'num_permanent_spec_blocks'):
            seq.num_permanent_spec_blocks = 0
        
        if debug and hasattr(seq, 'decode_strategy') and seq.decode_strategy == "jacobi":
            print(f"  [DEALLOCATE] Freed blocks, free_blocks_after={len(self.free_block_ids)}")

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        import os
        debug = os.environ.get("JACOBI_DEBUG", "0") == "1"
        
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        
        if debug and hasattr(seq, 'decode_strategy') and seq.decode_strategy == "jacobi":
            print(f"    [MAY_APPEND] seq_len={len(seq)}, pos_in_block={len(seq) % self.block_size}, "
                  f"block_size={self.block_size}, block_table={block_table[:3]}{'...' if len(block_table) > 3 else ''}")
        
        if len(seq) % self.block_size == 1:
            # Need to allocate a new block
            # For normal decoding: previous block should be finalized (hash != -1)
            # For Jacobi decoding: previous block may not be finalized due to trim/reappend
            is_jacobi = hasattr(seq, 'decode_strategy') and seq.decode_strategy == "jacobi"
            if not is_jacobi:
                assert last_block.hash != -1
            elif last_block.hash == -1:
                # Jacobi case: block was trimmed and not yet finalized
                # Finalize it now before allocating new block
                token_ids = seq.block(seq.num_blocks-1)
                prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
                h = self.compute_hash(token_ids, prefix)
                last_block.update(h, token_ids)
                self.hash_to_block_id[h] = last_block.block_id
                if debug:
                    print(f"    [MAY_APPEND] Finalized previous block: hash={h}, block_id={last_block.block_id}")
            
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
            if debug and hasattr(seq, 'decode_strategy') and seq.decode_strategy == "jacobi":
                print(f"    [MAY_APPEND] New block allocated: {block_id}")
        elif len(seq) % self.block_size == 0:
            # Block is now complete - finalize it if not already finalized
            # For Jacobi: block may already be finalized from the allocation step above
            is_jacobi = hasattr(seq, 'decode_strategy') and seq.decode_strategy == "jacobi"
            if last_block.hash == -1:
                # Block is not yet finalized - finalize it now
                token_ids = seq.block(seq.num_blocks-1)
                prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
                h = self.compute_hash(token_ids, prefix)
                last_block.update(h, token_ids)
                self.hash_to_block_id[h] = last_block.block_id
                if debug and hasattr(seq, 'decode_strategy') and seq.decode_strategy == "jacobi":
                    print(f"    [MAY_APPEND] Block finalized: hash={h}, block_id={last_block.block_id}")
            elif is_jacobi:
                # Jacobi: block was already finalized (e.g., during allocation)
                # This is okay - skip finalization
                if debug:
                    print(f"    [MAY_APPEND] Block already finalized: hash={last_block.hash}, block_id={last_block.block_id}")
            else:
                # Non-Jacobi: this shouldn't happen
                assert False, f"Block already finalized in non-Jacobi mode: {last_block.hash}"
        else:
            # Normal case: token being added to a partially filled block
            # For Jacobi: after KV trimming, block may be finalized but then modified
            # In this case, we need to reset the hash to allow modifications
            is_jacobi = hasattr(seq, 'decode_strategy') and seq.decode_strategy == "jacobi"
            if is_jacobi and last_block.hash != -1:
                # Jacobi case: block was finalized but now being modified
                # Reset hash to allow modifications
                if last_block.hash in self.hash_to_block_id:
                    del self.hash_to_block_id[last_block.hash]
                last_block.hash = -1
                if debug:
                    print(f"    [MAY_APPEND] Reset finalized block hash for modification: block_id={last_block.block_id}")
            else:
                # Normal case or Jacobi with unfinal ized block
                assert last_block.hash == -1, f"Block hash should be -1 for appending, got {last_block.hash}"

    def may_append_batch(self, seq: "Sequence", num_tokens: int):
        """Batch version of may_append for multiple tokens at once."""
        if num_tokens <= 0 or not seq.block_table:
            return
        
        last_block = self.blocks[seq.block_table[-1]]
        if last_block.hash != -1:
            if last_block.hash in self.hash_to_block_id:
                del self.hash_to_block_id[last_block.hash]
            last_block.hash = -1

    def allocate_temporary_blocks(self, seq: "Sequence", num_tokens: int) -> list[int]:
        """
        Allocate temporary blocks for draft tokens WITHOUT modifying the sequence.
        Used for Jacobi decoding to compute KV for draft without committing.
        
        Args:
            seq: The sequence (for calculating positions)
            num_tokens: Number of draft tokens to allocate blocks for
            
        Returns:
            List of temporary block IDs
        """
        if num_tokens <= 0:
            return []
        
        import os
        debug = os.environ.get("JACOBI_DEBUG", "0") == "1"
        
        # Calculate how many blocks we need for the draft
        seq_len = len(seq)
        start_pos = seq_len
        end_pos = seq_len + num_tokens
        
        start_block = start_pos // self.block_size
        end_block = (end_pos - 1) // self.block_size
        num_blocks_needed = end_block - start_block + 1
        
        # Check if we already have some blocks allocated
        existing_blocks = len(seq.block_table)
        blocks_to_allocate = num_blocks_needed - (existing_blocks - start_block)
        
        if blocks_to_allocate <= 0:
            # All needed blocks already exist
            if debug:
                print(f"    [TEMP_BLOCKS] No new blocks needed, using existing {seq.block_table[start_block:]}")
            return []
        
        if len(self.free_block_ids) < blocks_to_allocate:
            raise RuntimeError(f"Not enough free blocks for draft: need {blocks_to_allocate}, have {len(self.free_block_ids)}")
        
        # Allocate temporary blocks
        temp_blocks = []
        for _ in range(blocks_to_allocate):
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
            temp_blocks.append(block_id)
        
        if debug:
            print(f"    [TEMP_BLOCKS] Allocated {len(temp_blocks)} temporary blocks: {temp_blocks}")
        
        return temp_blocks
    
    def free_temporary_blocks(self, block_ids: list[int]) -> None:
        """
        Free temporary blocks that were allocated for draft tokens.
        
        Args:
            block_ids: List of temporary block IDs to free
        """
        if not block_ids:
            return
        
        import os
        debug = os.environ.get("JACOBI_DEBUG", "0") == "1"
        
        if debug:
            print(f"    [TEMP_BLOCKS] Freeing {len(block_ids)} temporary blocks: {block_ids}")
        
        for block_id in block_ids:
            block = self.blocks[block_id]
            if block.ref_count == 0:
                # Already free
                continue
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
    
    def commit_temporary_blocks(self, seq: "Sequence", temp_block_ids: list[int], num_tokens_to_commit: int) -> None:
        """
        Commit temporary blocks to the sequence's block_table for accepted tokens.
        
        Args:
            seq: The sequence to commit blocks to
            temp_block_ids: List of temporary block IDs
            num_tokens_to_commit: Number of tokens being committed (may be less than full draft)
        """
        if not temp_block_ids or num_tokens_to_commit <= 0:
            return
        
        import os
        debug = os.environ.get("JACOBI_DEBUG", "0") == "1"
        
        # Calculate which blocks are actually needed for the committed tokens
        seq_len = len(seq)
        end_pos = seq_len + num_tokens_to_commit
        
        start_block_idx = seq_len // self.block_size
        end_block_idx = (end_pos - 1) // self.block_size
        
        blocks_to_commit = end_block_idx - start_block_idx + 1 - (len(seq.block_table) - start_block_idx)
        
        if blocks_to_commit > 0:
            # Add the needed temporary blocks to the sequence's block_table
            seq.block_table.extend(temp_block_ids[:blocks_to_commit])
            
            if debug:
                print(f"    [TEMP_BLOCKS] Committed {blocks_to_commit} blocks to sequence: {temp_block_ids[:blocks_to_commit]}")
        
        # Free any remaining temporary blocks that weren't needed
        unused_blocks = temp_block_ids[blocks_to_commit:]
        if unused_blocks:
            self.free_temporary_blocks(unused_blocks)

    def trim_seq_tail(self, seq: "Sequence", num_tokens: int) -> None:
        """Logically deletes the last num_tokens from seq.token_ids and updates block_table."""
        if num_tokens <= 0:
            return
        
        import os
        debug = os.environ.get("JACOBI_DEBUG", "0") == "1"
        
        if debug and hasattr(seq, 'decode_strategy') and seq.decode_strategy == "jacobi":
            print(f"    [TRIM_START] Trimming {num_tokens} tokens from seq_len={len(seq.token_ids)}")
            print(f"    [TRIM_START] block_table before={seq.block_table}, num_cached={seq.num_cached_tokens}")
        
        old_len = len(seq.token_ids)
        new_len = max(0, old_len - num_tokens)
        old_num_cached = seq.num_cached_tokens
        
        if self.kv_cache is not None and new_len < old_num_cached:
            block_positions = {}
            
            for pos in range(new_len, old_num_cached):
                block_idx = pos // self.block_size
                offset_in_block = pos % self.block_size
                if block_idx < len(seq.block_table):
                    block_id = seq.block_table[block_idx]
                    if block_id not in block_positions:
                        block_positions[block_id] = []
                    block_positions[block_id].append(offset_in_block)
            
            import torch
            for block_id, offsets in block_positions.items():
                if len(offsets) == self.block_size:
                    self.kv_cache[:, :, block_id, :, :, :] = 0
                else:
                    offsets_t = torch.tensor(offsets, dtype=torch.long, device=self.kv_cache.device)
                    self.kv_cache[:, :, block_id, offsets_t, :, :] = 0
            
            if debug and hasattr(seq, 'decode_strategy') and seq.decode_strategy == "jacobi":
                print(f"    [TRIM_KV_CLEAR] Cleared KV cache positions {new_len} to {old_num_cached} ({len(block_positions)} blocks)")
        
        flattened = []
        for t in seq.token_ids[:new_len]:
            if isinstance(t, (list, tuple)):
                import sys
                print(f"WARNING: Found nested token at position, flattening: {type(t)} with {len(t)} elements", 
                      file=sys.stderr, flush=True)
                flattened.extend([int(x) for x in t])
            else:
                flattened.append(int(t))
        
        seq.token_ids = flattened
        
        seq.num_tokens = len(seq.token_ids)
        
        if seq.token_ids:
            seq.last_token = seq.token_ids[-1]
        
        seq.num_cached_tokens = min(old_num_cached, new_len)

        new_num_blocks = (new_len + self.block_size - 1) // self.block_size if new_len > 0 else 0
        
        while len(seq.block_table) > new_num_blocks:
            block_id = seq.block_table.pop()
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        
        if seq.block_table and new_len > 0:
            tokens_in_last_block = new_len % self.block_size
            last_block_id = seq.block_table[-1]
            last_block = self.blocks[last_block_id]
            
            if last_block.hash != -1 and last_block.hash in self.hash_to_block_id:
                del self.hash_to_block_id[last_block.hash]
            last_block.hash = -1
        
        if debug and hasattr(seq, 'decode_strategy') and seq.decode_strategy == "jacobi":
            print(f"    [TRIM_END] new_len={new_len}, block_table after={seq.block_table}, num_cached={seq.num_cached_tokens}")
            if seq.block_table:
                print(f"    [TRIM_END] Last block: id={seq.block_table[-1]}, hash={self.blocks[seq.block_table[-1]].hash}")
    
    def trim_kv_only(self, seq: "Sequence", num_tokens: int) -> None:
        """Trim KV cache for the last num_tokens WITHOUT modifying token_ids."""
        if num_tokens <= 0:
            return
        
        import os
        debug = os.environ.get("JACOBI_DEBUG", "0") == "1"
        
        if debug:
            print(f"    [TRIM_KV_ONLY] Trimming {num_tokens} tokens from KV, seq_len={len(seq)}, num_cached={seq.num_cached_tokens}")
        
        old_num_cached = seq.num_cached_tokens
        new_num_cached = max(len(seq), old_num_cached - num_tokens)
        
        if self.kv_cache is not None and new_num_cached < old_num_cached:
            import torch
            
            num_to_clear = old_num_cached - new_num_cached
            positions = torch.arange(new_num_cached, old_num_cached, dtype=torch.int32, device='cuda')
            block_indices = torch.div(positions, self.block_size, rounding_mode='floor')
            offsets = torch.remainder(positions, self.block_size)
            
            unique_blocks = torch.unique(block_indices)
            
            for block_idx in unique_blocks:
                if block_idx.item() >= len(seq.block_table):
                    continue
                block_id = seq.block_table[block_idx.item()]
                
                mask = block_indices == block_idx
                block_offsets = offsets[mask]
                
                if len(block_offsets) == self.block_size:
                    self.kv_cache[:, :, block_id, :, :, :] = 0
                else:
                    self.kv_cache[:, :, block_id, block_offsets, :, :] = 0
            
            if debug:
                print(f"    [TRIM_KV_ONLY] Cleared KV positions {new_num_cached} to {old_num_cached} ({len(unique_blocks)} blocks, vectorized)")
        
        seq.num_cached_tokens = new_num_cached
        
        blocks_needed = (new_num_cached + self.block_size - 1) // self.block_size if new_num_cached > 0 else 0
        
        permanent_spec_blocks = getattr(seq, 'num_permanent_spec_blocks', 0)
        min_blocks_to_keep = blocks_needed + permanent_spec_blocks
        
        while len(seq.block_table) > min_blocks_to_keep:
            block_id = seq.block_table.pop()
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
                if debug:
                    print(f"    [TRIM_KV_ONLY] Deallocated block {block_id}")
        
        if debug and permanent_spec_blocks > 0:
            print(f"    [TRIM_KV_ONLY] Keeping {permanent_spec_blocks} permanent speculative blocks")
        
        if debug:
            print(f"    [TRIM_KV_ONLY] After trim: num_cached={seq.num_cached_tokens}, blocks={len(seq.block_table)}")
    
    def trim_kv_only_fast(self, seq: "Sequence", num_tokens: int) -> None:
        """Fast KV cache trimming - NO KV CLEARING (flash attention respects cache_seqlens)."""
        if num_tokens <= 0:
            return
        
        import os
        debug = os.environ.get("JACOBI_DEBUG", "0") == "1"
        
        old_num_cached = seq.num_cached_tokens
        new_num_cached = max(len(seq), old_num_cached - num_tokens)
        
        if debug:
            num_trimmed = old_num_cached - new_num_cached
            print(f"    [TRIM_KV_FAST] Trimmed {num_trimmed} positions (NO CLEAR - FA respects cache_seqlens)")
        
        seq.num_cached_tokens = new_num_cached
        
        blocks_needed = (new_num_cached + self.block_size - 1) // self.block_size if new_num_cached > 0 else 0
        permanent_spec_blocks = getattr(seq, 'num_permanent_spec_blocks', 0)
        min_blocks_to_keep = blocks_needed + permanent_spec_blocks
        
        while len(seq.block_table) > min_blocks_to_keep:
            block_id = seq.block_table.pop()
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
                if debug:
                    print(f"    [TRIM_KV_FAST] Deallocated block {block_id}")
        
        if debug:
            print(f"    [TRIM_KV_FAST] After trim: num_cached={seq.num_cached_tokens}, blocks={len(seq.block_table)}")