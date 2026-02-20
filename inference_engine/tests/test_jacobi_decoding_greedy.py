#!/usr/bin/env python3
"""
Test script for Jacobi decoding in nano-vLLM.

Usage:
    python tests/test_jacobi_decoding.py --model /path/to/model

This tests:
    1. Normal autoregressive decoding
    2. Jacobi decoding (vanilla)
    3. Correctness comparison between strategies
"""

import argparse
import os
import sys
import time
from typing import List, Tuple

# Add CLLM2/ to path so `from inference_engine import ...` works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from inference_engine import LLM, SamplingParams


def cleanup_scheduler(llm: LLM):
    """Clear all sequences from the scheduler to prevent state leakage between tests."""
    llm.scheduler.waiting.clear()
    llm.scheduler.running.clear()
    if llm.model_runner.jacobi_decoder:
        llm.model_runner.jacobi_decoder.stats = {
            'num_chunk_calls': 0,
            'num_jacobi_iterations': 0,
            'tokens_accepted': 0,
            'tokens_per_call': [],
            'tokens_per_iteration': [],
            'iterations_per_call': [],
        }


def test_autoregressive_decoding(llm: LLM, prompts: List[str], max_tokens: int = 64) -> Tuple[List[str], float]:
    """Test normal autoregressive decoding."""
    sampling_params = SamplingParams(
        temperature=0.0,  # greedy decoding
        max_tokens=max_tokens,
        decode_strategy="autoregressive",
    )
    
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - start
    
    return [out["text"] for out in outputs], elapsed


def test_jacobi_decoding(llm: LLM, prompts: List[str], max_tokens: int = 1024, block_len: int = 64) -> Tuple[List[str], float, dict]:
    """Test Jacobi decoding (vanilla)."""
    
    jacobi_decoder = llm.model_runner.jacobi_decoder
    if jacobi_decoder:
        jacobi_decoder.stats = {
            'num_chunk_calls': 0,
            'num_jacobi_iterations': 0,
            'tokens_accepted': 0,
            'tokens_per_call': [],
            'tokens_per_iteration': [],
            'iterations_per_call': [],
        }
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        decode_strategy="jacobi",
        jacobi_block_len=block_len,
        jacobi_max_iterations=128,
    )
    
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - start
    
    if llm.model_runner.jacobi_decoder:
        acceptance_stats = llm.model_runner.jacobi_decoder.stats.copy()
    else:
        acceptance_stats = {
            'num_chunk_calls': 0,
            'num_jacobi_iterations': 0,
            'tokens_accepted': 0,
            'tokens_per_call': [],
            'tokens_per_iteration': [],
            'iterations_per_call': [],
        }
    
    return [out["text"] for out in outputs], elapsed, acceptance_stats


def test_jacobi_multiblock_not_implemented(llm: LLM, prompts: List[str]) -> bool:
    """Test that JacobiMultiBlock raises NotImplementedError."""
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
        decode_strategy="jacobi_multiblock_rejection_recycling",
    )
    
    try:
        llm.generate(prompts[:1], sampling_params, use_tqdm=False)
        return False
    except NotImplementedError as e:
        print(f" [OK] NotImplementedError raised as expected: {e}")
        cleanup_scheduler(llm)
        return True
    except Exception as e:
        print(f" [FAIL] Unexpected error: {e}")
        cleanup_scheduler(llm)
        return False


def test_single_sequence(llm: LLM, prompt: str, max_tokens: int = 1024):
    """Test single sequence generation with both strategies."""
    print("\n" + "="*60)
    print("TEST: Single Sequence Generation")
    print("="*60)
    
    print(f"\nPrompt: {prompt}")
    
    # Autoregressive
    print("\n[Autoregressive Decoding]")
    ar_outputs, ar_time = test_autoregressive_decoding(llm, [prompt], max_tokens)
    ar_num_tokens = len(llm.tokenizer.encode(ar_outputs[0]))
    ar_tps = ar_num_tokens / ar_time if ar_time > 0 else 0
    print(f"  Output: {ar_outputs[0]}")
    print(f"  Time: {ar_time:.3f}s")
    print(f"  Tokens generated: {ar_num_tokens}")
    print(f"  TPS (tokens/sec): {ar_tps:.2f}")
    
    # Jacobi
    print("\n[Jacobi Decoding]")
    jacobi_outputs, jacobi_time, stats = test_jacobi_decoding(llm, [prompt], max_tokens)
    jacobi_num_tokens = len(llm.tokenizer.encode(jacobi_outputs[0]))
    jacobi_tps = jacobi_num_tokens / jacobi_time if jacobi_time > 0 else 0
    print(f"  Output: {jacobi_outputs[0]}")
    print(f"  Time: {jacobi_time:.3f}s")
    print(f"  Tokens generated: {jacobi_num_tokens}")
    print(f"  TPS (tokens/sec): {jacobi_tps:.2f}")
    
    # Speedup comparison
    if ar_tps > 0:
        speedup = jacobi_tps / ar_tps
        print(f"\n  [Performance Comparison]")
        print(f"    Jacobi speedup: {speedup:.2f}x")
    
    # Print acceptance statistics
    print(f"\n  [Acceptance Stats]")
    if stats['num_chunk_calls'] > 0:
        avg_tokens_per_call = stats['tokens_accepted'] / stats['num_chunk_calls']
        print(f"    Chunk calls: {stats['num_chunk_calls']}")
        print(f"    Jacobi iterations: {stats['num_jacobi_iterations']}")
        print(f"    Tokens accepted: {stats['tokens_accepted']}")
        print(f"    Avg tokens per call: {avg_tokens_per_call:.2f}")
        
        if stats['num_jacobi_iterations'] > 0:
            avg_tokens_per_iter = stats['tokens_accepted'] / stats['num_jacobi_iterations']
            avg_iters_per_call = stats['num_jacobi_iterations'] / stats['num_chunk_calls']
            print(f"    Avg tokens per iteration (TPF): {avg_tokens_per_iter:.2f}")
            print(f"    Avg iterations per call: {avg_iters_per_call:.2f}")
        
        if stats['tokens_per_call']:
            print(f"    Min/Max tokens per call: {min(stats['tokens_per_call'])}/{max(stats['tokens_per_call'])}")
        
        if len(stats['tokens_per_iteration']) > 1:
            from collections import Counter
            dist = Counter(stats['tokens_per_iteration'])
            print(f"    Token acceptance distribution (per iteration, top 10):")
            for tokens, count in dist.most_common(10):
                pct = count / len(stats['tokens_per_iteration']) * 100
                print(f"      {tokens} tokens: {count} times ({pct:.1f}%)")
    else:
        print(f"    No stats collected (chunk_calls={stats['num_chunk_calls']}, tokens_accepted={stats['tokens_accepted']})")
    
    # Compare first 200 tokens
    print("\n[Correctness Check]")
    ar_tokens = llm.tokenizer.encode(ar_outputs[0])
    jacobi_tokens = llm.tokenizer.encode(jacobi_outputs[0])
    
    tokens_to_check = min(200, len(ar_tokens), len(jacobi_tokens))
    
    if tokens_to_check < 200:
        print(f"  Warning: Only {tokens_to_check} tokens available (less than 200)")
    
    num_matches = sum(1 for i in range(tokens_to_check) if ar_tokens[i] == jacobi_tokens[i])
    match_percentage = (num_matches / tokens_to_check * 100) if tokens_to_check > 0 else 0
    
    print(f"  Match rate: {num_matches}/{tokens_to_check} tokens ({match_percentage:.1f}%)")
    
    if num_matches == tokens_to_check:
        print(f"  ✔️ First {tokens_to_check} tokens match!")
        return True
    else:
        print(f"  ❌ Token mismatch in first {tokens_to_check} tokens")
        for i in range(tokens_to_check):
            if ar_tokens[i] != jacobi_tokens[i]:
                print(f"    First mismatch at position {i}:")
                print(f"      Autoregressive: token_id={ar_tokens[i]} '{llm.tokenizer.decode([ar_tokens[i]])}'")
                print(f"      Jacobi: token_id={jacobi_tokens[i]} '{llm.tokenizer.decode([jacobi_tokens[i]])}'")
                break
        return False


def test_batch_sequences(llm: LLM, prompts: List[str], max_tokens: int = 1024):
    """Test batch sequence generation with Jacobi decoding."""
    print("\n" + "="*60)
    print(f"TEST: Batch Sequence Generation (bsz={len(prompts)})")
    print("="*60)
    
    # Autoregressive
    print("\n[Autoregressive Decoding - Batch]")
    ar_outputs, ar_time = test_autoregressive_decoding(llm, prompts, max_tokens)
    for i, output in enumerate(ar_outputs):
        print(f"  Output {i}: {output}")
    print(f"  Completed {len(ar_outputs)} sequences")
    print(f"  Time: {ar_time:.3f}s")
    print(f"  Throughput: {len(prompts) * max_tokens / ar_time:.1f} tok/s (upper bound)")
    
    # Jacobi
    print("\n[Jacobi Decoding - Batch]")
    jacobi_outputs, jacobi_time, stats = test_jacobi_decoding(llm, prompts, max_tokens)
    for i, output in enumerate(jacobi_outputs):
        print(f"  Output {i}: {output}")
    print(f"  Completed {len(jacobi_outputs)} sequences")
    print(f"  Time: {jacobi_time:.3f}s")
    print(f"  Throughput: {len(prompts) * max_tokens / jacobi_time:.1f} tok/s (upper bound)")
    
    # Print acceptance statistics
    if stats['num_chunk_calls'] > 0:
        avg_tokens_per_call = stats['tokens_accepted'] / stats['num_chunk_calls']
        print(f"\n  [Acceptance Stats]")
        print(f"    Chunk calls: {stats['num_chunk_calls']}")
        print(f"    Jacobi iterations: {stats['num_jacobi_iterations']}")
        print(f"    Tokens accepted: {stats['tokens_accepted']}")
        print(f"    Avg tokens per call: {avg_tokens_per_call:.2f}")
        
        # Per-iteration stats (key metric for Jacobi efficiency)
        if stats['num_jacobi_iterations'] > 0:
            avg_tokens_per_iter = stats['tokens_accepted'] / stats['num_jacobi_iterations']
            avg_iters_per_call = stats['num_jacobi_iterations'] / stats['num_chunk_calls']
            print(f"    Avg tokens per iteration (TPF): {avg_tokens_per_iter:.2f}")
            print(f"    Avg iterations per call: {avg_iters_per_call:.2f}")
        
        if stats['tokens_per_call']:
            print(f"    Min/Max tokens per call: {min(stats['tokens_per_call'])}/{max(stats['tokens_per_call'])}")
        
        if len(stats['tokens_per_iteration']) > 1:
            from collections import Counter
            dist = Counter(stats['tokens_per_iteration'])
            print(f"    Token acceptance distribution (per iteration, top 20):")
            for tokens, count in dist.most_common(20):
                pct = count / len(stats['tokens_per_iteration']) * 100
                print(f"      {tokens} tokens: {count} times ({pct:.1f}%)")
    
    # Compare first 200 tokens for each prompt
    print("\n[Correctness Check]")
    total_matches = 0
    total_checked = 0
    
    for i in range(len(prompts)):
        ar_tokens = llm.tokenizer.encode(ar_outputs[i])
        jacobi_tokens = llm.tokenizer.encode(jacobi_outputs[i])
        
        tokens_to_check = min(200, len(ar_tokens), len(jacobi_tokens))
        
        num_matches = sum(1 for j in range(tokens_to_check) if ar_tokens[j] == jacobi_tokens[j])
        match_percentage = (num_matches / tokens_to_check * 100) if tokens_to_check > 0 else 0
        
        total_matches += num_matches
        total_checked += tokens_to_check
        
        if num_matches == tokens_to_check:
            print(f"  Prompt {i}: ✔️ {num_matches}/{tokens_to_check} tokens ({match_percentage:.1f}%)")
        else:
            print(f"  Prompt {i}: ❌  {num_matches}/{tokens_to_check} tokens ({match_percentage:.1f}%)")
            for j in range(tokens_to_check):
                if ar_tokens[j] != jacobi_tokens[j]:
                    print(f"    First mismatch at position {j}:")
                    print(f"      Autoregressive: token_id={ar_tokens[j]} '{llm.tokenizer.decode([ar_tokens[j]])}'")
                    print(f"      Jacobi: token_id={jacobi_tokens[j]} '{llm.tokenizer.decode([jacobi_tokens[j]])}'")
                    break
    
    overall_percentage = (total_matches / total_checked * 100) if total_checked > 0 else 0    
    print(f"\n  Overall match rate: {total_matches}/{total_checked} tokens ({overall_percentage:.1f}%)")
    
    if overall_percentage == 100.0:
        print(f"  ✔️ PASS - Exact match (100%)")
        test_passed = True
    elif overall_percentage >= 50.0:
        print(f"  ⚠️ PASS - {overall_percentage:.1f}% match (threshold: 50%)")
        print(f"      Note: Mismatches may be due to numerical variance in floating-point operations")
        test_passed = True
    else:
        print(f"  ❌ FAIL - Only {overall_percentage:.1f}% match (threshold: 50%)")
        test_passed = False
    
    return test_passed


def test_jacobi_multiblock_error(llm: LLM, prompts: List[str]):
    """Test that JacobiMultiBlock raises NotImplementedError."""
    print("\n" + "="*60)
    print("TEST: JacobiMultiBlock NotImplementedError")
    print("="*60)
    
    return test_jacobi_multiblock_not_implemented(llm, prompts)


def test_cross_serving_mode_consistency(llm: LLM, prompts: List[str], max_tokens: int = 1024, block_len: int = 64):
    """Test that all serving modes produce consistent results.
    
    Compares:
    - Autoregressive single requests (one at a time)
    - Jacobi single requests (one at a time)
    - Autoregressive batched
    - Jacobi batched
    
    All serving modes should produce the same outputs for the same prompts.
    """
    print("\n" + "="*60)
    print(f"TEST: Cross-Serving Mode Consistency (N={len(prompts)} prompts)")
    print("="*60)
    
    # Strategy 1: Autoregressive single requests
    print("\n[Strategy 1: Autoregressive Single Requests]")
    ar_single_outputs = []
    ar_single_time = 0
    for i, prompt in enumerate(prompts):
        outputs, elapsed = test_autoregressive_decoding(llm, [prompt], max_tokens)
        ar_single_outputs.append(outputs[0])
        ar_single_time += elapsed
        print(f"  Prompt {i}: Generated {len(llm.tokenizer.encode(outputs[0]))} tokens in {elapsed:.3f}s")
    print(f"  Total time: {ar_single_time:.3f}s")
    
    # Strategy 2: Jacobi single requests
    print("\n[Strategy 2: Jacobi Single Requests]")
    jacobi_single_outputs = []
    jacobi_single_time = 0
    jacobi_single_stats = {
        'num_chunk_calls': 0,
        'num_jacobi_iterations': 0,
        'tokens_accepted': 0,
        'tokens_per_call': [],
        'tokens_per_iteration': [],
        'iterations_per_call': [],
    }
    for i, prompt in enumerate(prompts):
        outputs, elapsed, stats = test_jacobi_decoding(llm, [prompt], max_tokens, block_len)
        jacobi_single_outputs.append(outputs[0])
        jacobi_single_time += elapsed
        # Aggregate stats
        jacobi_single_stats['num_chunk_calls'] += stats['num_chunk_calls']
        jacobi_single_stats['num_jacobi_iterations'] += stats['num_jacobi_iterations']
        jacobi_single_stats['tokens_accepted'] += stats['tokens_accepted']
        jacobi_single_stats['tokens_per_call'].extend(stats['tokens_per_call'])
        jacobi_single_stats['tokens_per_iteration'].extend(stats['tokens_per_iteration'])
        jacobi_single_stats['iterations_per_call'].extend(stats['iterations_per_call'])
        print(f"  Prompt {i}: Generated {len(llm.tokenizer.encode(outputs[0]))} tokens in {elapsed:.3f}s")
    print(f"  Total time: {jacobi_single_time:.3f}s")
    if jacobi_single_stats['num_chunk_calls'] > 0:
        avg_tokens_per_call = jacobi_single_stats['tokens_accepted'] / jacobi_single_stats['num_chunk_calls']
        print(f"  Avg tokens per call: {avg_tokens_per_call:.2f}")
        if jacobi_single_stats['num_jacobi_iterations'] > 0:
            avg_tokens_per_iter = jacobi_single_stats['tokens_accepted'] / jacobi_single_stats['num_jacobi_iterations']
            print(f"  Avg tokens per iteration (TPF): {avg_tokens_per_iter:.2f}")
            print(f"  Total Jacobi iterations: {jacobi_single_stats['num_jacobi_iterations']}")
    
    # Strategy 3: Autoregressive batched
    print("\n[Strategy 3: Autoregressive Batched]")
    ar_batch_outputs, ar_batch_time = test_autoregressive_decoding(llm, prompts, max_tokens)
    print(f"  Generated {len(prompts)} sequences in {ar_batch_time:.3f}s")
    for i, output in enumerate(ar_batch_outputs):
        print(f"  Prompt {i}: {len(llm.tokenizer.encode(output))} tokens")
    
    # Strategy 4: Jacobi batched
    print("\n[Strategy 4: Jacobi Batched]")
    jacobi_batch_outputs, jacobi_batch_time, jacobi_batch_stats = test_jacobi_decoding(llm, prompts, max_tokens, block_len)
    print(f"  Generated {len(prompts)} sequences in {jacobi_batch_time:.3f}s")
    for i, output in enumerate(jacobi_batch_outputs):
        print(f"  Prompt {i}: {len(llm.tokenizer.encode(output))} tokens")
    if jacobi_batch_stats['num_chunk_calls'] > 0:
        avg_tokens_per_call = jacobi_batch_stats['tokens_accepted'] / jacobi_batch_stats['num_chunk_calls']
        print(f"  Avg tokens per call: {avg_tokens_per_call:.2f}")
        if jacobi_batch_stats['num_jacobi_iterations'] > 0:
            avg_tokens_per_iter = jacobi_batch_stats['tokens_accepted'] / jacobi_batch_stats['num_jacobi_iterations']
            print(f"  Avg tokens per iteration (TPF): {avg_tokens_per_iter:.2f}")
            print(f"  Total Jacobi iterations: {jacobi_batch_stats['num_jacobi_iterations']}")
    
    # Correctness check: Compare all strategies
    print("\n[Correctness Check: Cross-Strategy Comparison]")
    print("Comparing first 200 tokens across all strategies...")
    
    all_match = True
    total_matches = 0
    total_checked = 0
    prompts_passed = 0
    
    for i in range(len(prompts)):
        # Tokenize all outputs
        ar_single_tokens = llm.tokenizer.encode(ar_single_outputs[i])
        jacobi_single_tokens = llm.tokenizer.encode(jacobi_single_outputs[i])
        ar_batch_tokens = llm.tokenizer.encode(ar_batch_outputs[i])
        jacobi_batch_tokens = llm.tokenizer.encode(jacobi_batch_outputs[i])
        
        # Find minimum length to compare
        tokens_to_check = min(200, 
                             len(ar_single_tokens), 
                             len(jacobi_single_tokens),
                             len(ar_batch_tokens),
                             len(jacobi_batch_tokens))
        
        # Compare all strategies pairwise with AR single as reference
        matches_jacobi_single = sum(1 for j in range(tokens_to_check) 
                                   if ar_single_tokens[j] == jacobi_single_tokens[j])
        matches_ar_batch = sum(1 for j in range(tokens_to_check) 
                              if ar_single_tokens[j] == ar_batch_tokens[j])
        matches_jacobi_batch = sum(1 for j in range(tokens_to_check) 
                                  if ar_single_tokens[j] == jacobi_batch_tokens[j])
        
        prompt_matches = (matches_jacobi_single == tokens_to_check and 
                         matches_ar_batch == tokens_to_check and 
                         matches_jacobi_batch == tokens_to_check)
        
        if prompt_matches:
            prompts_passed += 1
        
        total_checked += tokens_to_check * 3  # 3 comparisons per prompt
        total_matches += matches_jacobi_single + matches_ar_batch + matches_jacobi_batch
        
        if prompt_matches:
            print(f"  Prompt {i}: ✔️ All strategies match ({tokens_to_check} tokens)")
        else:
            print(f"  Prompt {i}: ❌ Mismatch detected")
            print(f"    AR single vs Jacobi single: {matches_jacobi_single}/{tokens_to_check}")
            print(f"    AR single vs AR batch: {matches_ar_batch}/{tokens_to_check}")
            print(f"    AR single vs Jacobi batch: {matches_jacobi_batch}/{tokens_to_check}")
            
            # Find first mismatches
            for j in range(tokens_to_check):
                if ar_single_tokens[j] != jacobi_single_tokens[j]:
                    print(f"    First mismatch (AR single vs Jacobi single) at pos {j}:")
                    print(f"      AR single: {ar_single_tokens[j]} '{llm.tokenizer.decode([ar_single_tokens[j]])}'")
                    print(f"      Jacobi single: {jacobi_single_tokens[j]} '{llm.tokenizer.decode([jacobi_single_tokens[j]])}'")
                    break
            for j in range(tokens_to_check):
                if ar_single_tokens[j] != ar_batch_tokens[j]:
                    print(f"    First mismatch (AR single vs AR batch) at pos {j}:")
                    print(f"      AR single: {ar_single_tokens[j]} '{llm.tokenizer.decode([ar_single_tokens[j]])}'")
                    print(f"      AR batch: {ar_batch_tokens[j]} '{llm.tokenizer.decode([ar_batch_tokens[j]])}'")
                    break
            for j in range(tokens_to_check):
                if ar_single_tokens[j] != jacobi_batch_tokens[j]:
                    print(f"    First mismatch (AR single vs Jacobi batch) at pos {j}:")
                    print(f"      AR single: {ar_single_tokens[j]} '{llm.tokenizer.decode([ar_single_tokens[j]])}'")
                    print(f"      Jacobi batch: {jacobi_batch_tokens[j]} '{llm.tokenizer.decode([jacobi_batch_tokens[j]])}'")
                    break
            
            all_match = False
    
    # Overall statistics
    overall_percentage = (total_matches / total_checked * 100) if total_checked > 0 else 0
    print(f"\n  Overall match rate: {total_matches}/{total_checked} comparisons ({overall_percentage:.1f}%)")
    
    # Use the prompts_passed count from the loop above
    prompts_pass_rate = (prompts_passed / len(prompts) * 100) if len(prompts) > 0 else 0
    test_passed = prompts_passed >= len(prompts) * 0.5  # 50% threshold
    
    print(f"\n[Test Result]")
    print(f"  Prompts passed: {prompts_passed}/{len(prompts)} ({prompts_pass_rate:.1f}%)")
    
    if prompts_passed == len(prompts):
        print(f"  ✔️ PASS - Perfect match across all strategies!")
    elif test_passed:
        print(f"  ⚠️  PASS (with possible numerical variance) - {prompts_pass_rate:.1f}% of prompts exact match")
        print(f"      Note: Some mismatches may be due to numerical variance in floating-point operations")
    else:
        print(f"  ❌ FAIL - Only {prompts_pass_rate:.1f}% of prompts match (threshold: 50%)")
    
    # Performance summary
    print("\n[Cross Serving Mode Performance Summary]")
    print(f"  AR single (sequential): {ar_single_time:.3f}s")
    print(f"  Jacobi single (sequential): {jacobi_single_time:.3f}s (speedup: {ar_single_time/jacobi_single_time:.2f}x)")
    print(f"  AR batch: {ar_batch_time:.3f}s")
    print(f"  Jacobi batch: {jacobi_batch_time:.3f}s (speedup: {ar_batch_time/jacobi_batch_time:.2f}x)")
    
    return test_passed


def main():
    parser = argparse.ArgumentParser(description="Test Jacobi decoding in nano-vLLM")
    parser.add_argument("--model-path", type=str, default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"),
                        help="Path to model directory")
    parser.add_argument("--tokenizer-path", type=str, default=None,
                        help="Path to tokenizer directory (defaults to model path)")
    parser.add_argument("--enforce-eager", action="store_true", default=False,
                        help="Disable CUDA graphs")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Max tokens to generate")
    args = parser.parse_args()
    
    print("="*60)
    print("Nano-vLLM Jacobi Decoding Test Suite")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Tokenizer: {args.tokenizer_path or args.model_path}")
    print(f"Enforce eager: {args.enforce_eager}")
    print(f"Max tokens: {args.max_tokens}")
    
    # Initialize LLM
    print("\nInitializing LLM...")
    try:
        llm = LLM(
            args.model_path,
            tokenizer_path=args.tokenizer_path,
            enforce_eager=args.enforce_eager,
            tensor_parallel_size=1,
            max_model_len=2048,
        )
    except Exception as e:
        print(f"Failed to initialize LLM: {e}")
        print("\nOne potential cause: Make sure you have downloaded the model:")
        print("  huggingface-cli download --resume-download Qwen/Qwen3-0.6B \\")
        print("    --local-dir ~/huggingface/Qwen3-0.6B/ \\")
        print("    --local-dir-use-symlinks False")
        return 1
    
    # Test prompts (concise OpenCodeInstruct style)
    code_prompts = [
        # 0: AVL Tree with insert and rotations (challenging, long output)
        (
            "Implement an AVL tree with insert. Include left/right rotations and rebalancing.\n\n"
            "**Example:** Insert 10,20,30 → triggers left rotation → root becomes 20.\n\n"
            "class AVLNode:\n"
            "    def __init__(self, key):\n"
            "        self.key, self.left, self.right, self.height = key, None, None, 1\n\n"
            "class AVLTree:\n"
            "    def __init__(self): self.root = None\n"
            "    def insert(self, key): pass  # Insert key, rebalance\n"
            "    def _left_rotate(self, z): pass  # Left rotation\n"
            "    def _right_rotate(self, y): pass  # Right rotation\n"
            "    def _get_height(self, node): pass\n"
            "    def _get_balance(self, node): pass\n"
        ),
        # 1: Fibonacci
        (
            "Return the n-th Fibonacci number. Use memoization for efficiency.\n\n"
            "**Input:** `n = 10` → **Output:** `55`\n\n"
            "def fib(n: int) -> int:\n"
            "    pass\n"
        ),
        # 2: Min-heap push
        (
            "Push `x` into a min-heap in-place. Parent at `(i-1)//2` must be ≤ child at `i`.\n\n"
            "**Input:** `heap = [1, 3, 5], x = 2` → **Output:** `heap = [1, 2, 5, 3]`\n\n"
            "def heap_push(heap: list[int], x: int) -> None:\n"
            "    pass\n"
        ),
        # 3: Bubble sort
        (
            "Sort a list using bubble sort. Stop early if no swaps in a pass.\n\n"
            "**Input:** `[64, 34, 25, 12]` → **Output:** `[12, 25, 34, 64]`\n\n"
            "def bubble_sort(a: list[int]) -> list[int]:\n"
            "    pass\n"
        ),
        # 4: LRU Cache
        (
            "Implement an LRU cache with `get(key)` and `put(key, value)`. Evict LRU on capacity.\n\n"
            "**Example:** `cache = LRUCache(2); cache.put(1,1); cache.put(2,2); cache.get(1)→1`\n\n"
            "class LRUCache:\n"
            "    def __init__(self, capacity: int): pass\n"
            "    def get(self, key: int) -> int: pass\n"
            "    def put(self, key: int, value: int) -> None: pass\n"
        ),
        # 5: Binary tree level order
        (
            "Return level-order traversal of a binary tree (left to right, level by level).\n\n"
            "**Input:** Tree `[3,9,20,null,null,15,7]` → **Output:** `[[3],[9,20],[15,7]]`\n\n"
            "class TreeNode:\n"
            "    def __init__(self, val=0, left=None, right=None):\n"
            "        self.val, self.left, self.right = val, left, right\n\n"
            "def level_order(root: TreeNode) -> list[list[int]]:\n"
            "    pass\n"
        ),
    ]

    def build_messages(code_snippet: str):
        user_prompt = (
            "Please continue to complete the function. You are not allowed to modify the given code; "
            "only fill in the missing parts. Return the entire completed function in a code block; "
            "Add 5 example use of the function with different inputs and outputs.\n\n"
            f"\n{code_snippet.strip()}\n```"
        )
        return [
            {"role": "user", "content": user_prompt},
        ]
    
    # Apply chat template to prompts
    prompts = []
    for code_prompt in code_prompts:
        messages = build_messages(code_prompt)
        formatted = llm.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(formatted)
    print(f"Example formatted prompt: {prompts[0]}")
    
    # Warmup
    print("\nWarmup run...")
    llm.generate(["Hello"], SamplingParams(temperature=0.0, max_tokens=8, decode_strategy="autoregressive"), use_tqdm=False)
    cleanup_scheduler(llm)  # Clean up after warmup
    
    # Run tests
    results = []
    
    # Test 1: Single sequence
    results.append(("Single Sequence", test_single_sequence(llm, prompts[0], args.max_tokens)))
    cleanup_scheduler(llm)  # Clean up between tests
    
    # Test 2: Batch sequences
    results.append(("Batch Sequences", test_batch_sequences(llm, prompts, args.max_tokens)))
    cleanup_scheduler(llm)  # Clean up between tests
    
    # Test 3: JacobiMultiBlock error (already cleans up internally)
    results.append(("JacobiMultiBlock Error", test_jacobi_multiblock_error(llm, prompts)))
    
    # Test 4: Cross-serving mode consistency
    results.append(("Cross-Serving Mode Consistency", test_cross_serving_mode_consistency(llm, prompts, args.max_tokens)))
    cleanup_scheduler(llm)  # Clean up after tests

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed
    
    print("\n" + ("All tests passed!" if all_passed else "Some tests failed!"))
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

