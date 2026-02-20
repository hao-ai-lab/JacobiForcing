#!/usr/bin/env python3
"""
Test script for non-greedy Jacobi decoding in nano-vLLM.

Usage:
    python tests/test_jacobi_decoding_nongreedy.py --model /path/to/model

This tests:
    1. Non-greedy Jacobi decoding with temperature > 0
    2. Distribution-wise comparison between autoregressive and non-greedy Jacobi decoding
    3. Uses Jensen-Shannon divergence to quantify distribution similarity
"""

import argparse
import os
import sys
import time
from typing import List, Tuple, Dict
from collections import Counter, defaultdict
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy

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


def compute_token_distributions(token_sequences: List[List[int]], vocab_size: int = None) -> Dict[int, np.ndarray]:
    """
    Compute token distributions at each position across multiple sequences.
    
    Args:
        token_sequences: List of token sequences (each is a list of token IDs)
        vocab_size: Optional vocabulary size. If None, inferred from max token ID.
    
    Returns:
        Dictionary mapping position index -> probability distribution (numpy array)
    """
    if not token_sequences:
        return {}
    
    # Find max position and vocab size
    max_pos = max(len(seq) for seq in token_sequences)
    if vocab_size is None:
        max_token_id = max(max(seq, default=0) for seq in token_sequences)
        vocab_size = max_token_id + 1
    
    distributions = {}
    
    for pos in range(max_pos):
        # Collect tokens at this position
        tokens_at_pos = [seq[pos] for seq in token_sequences if len(seq) > pos]
        
        if not tokens_at_pos:
            continue
        
        # Count tokens
        token_counts = Counter(tokens_at_pos)
        
        # Build probability distribution
        dist = np.zeros(vocab_size, dtype=np.float64)
        total = len(tokens_at_pos)
        
        for token_id, count in token_counts.items():
            if 0 <= token_id < vocab_size:
                dist[token_id] = count / total
        
        distributions[pos] = dist
    
    return distributions


def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute Jensen-Shannon divergence between two distributions.
    
    JS divergence is symmetric and bounded in [0, 1].
    JS(p||q) = 0.5 * KL(p||m) + 0.5 * KL(q||m), where m = 0.5 * (p + q)
    
    Args:
        p: First probability distribution
        q: Second probability distribution
    
    Returns:
        JS divergence (0 = identical, 1 = maximally different)
    """
    # Ensure same length
    min_len = min(len(p), len(q))
    p = p[:min_len]
    q = q[:min_len]
    
    # Normalize to ensure they sum to 1
    p = p / (p.sum() + 1e-10)
    q = q / (q.sum() + 1e-10)
    
    # Compute JS divergence using scipy
    js_div = jensenshannon(p, q)
    
    # jensenshannon returns sqrt(JS), so square it to get JS divergence
    return js_div ** 2


def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Compute KL divergence KL(p||q).
    
    Args:
        p: First probability distribution
        q: Second probability distribution
        epsilon: Small value to avoid log(0)
    
    Returns:
        KL divergence (0 = identical, larger = more different)
    """
    min_len = min(len(p), len(q))
    p = p[:min_len]
    q = q[:min_len]
    
    # Normalize
    p = p / (p.sum() + epsilon)
    q = q / (q.sum() + epsilon)
    
    # Add epsilon to avoid log(0)
    p = p + epsilon
    q = q + epsilon
    p = p / p.sum()
    q = q / q.sum()
    
    return entropy(p, qk=q)


def total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Compute total variation distance between two distributions.
    
    TV(p, q) = 0.5 * sum(|p_i - q_i|)
    
    Args:
        p: First probability distribution
        q: Second probability distribution
    
    Returns:
        Total variation distance (0 = identical, 1 = maximally different)
    """
    min_len = min(len(p), len(q))
    p = p[:min_len]
    q = q[:min_len]
    
    # Normalize
    p = p / (p.sum() + 1e-10)
    q = q / (q.sum() + 1e-10)
    
    return 0.5 * np.sum(np.abs(p - q))


def test_nongreedy_autoregressive_decoding(
    llm: LLM, 
    prompt: str, 
    temperature: float,
    max_tokens: int,
    num_samples: int,
    print_samples: int = 3
) -> Tuple[List[List[int]], float]:
    """Generate multiple samples using non-greedy autoregressive decoding."""
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        decode_strategy="autoregressive",
    )
    
    token_sequences = []
    generated_texts = []
    start = time.perf_counter()
    
    for i in range(num_samples):
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        token_ids = outputs[0]["token_ids"]
        token_sequences.append(token_ids)
        
        # Decode and store text for first few samples
        if i < print_samples:
            generated_text = llm.tokenizer.decode(token_ids, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        cleanup_scheduler(llm)
    
    elapsed = time.perf_counter() - start
    
    # Print sample generations
    if generated_texts:
        print(f"\n  Sample generations (first {len(generated_texts)}):")
        for i, text in enumerate(generated_texts):
            print(f"\n  [{i+1}] {text[:200]}{'...' if len(text) > 200 else ''}")
    
    return token_sequences, elapsed


def test_nongreedy_jacobi_decoding(
    llm: LLM,
    prompt: str,
    temperature: float,
    max_tokens: int,
    num_samples: int,
    block_len: int = 64,
    print_samples: int = 3
) -> Tuple[List[List[int]], float, dict]:
    """Generate multiple samples using non-greedy Jacobi decoding."""
    
    # Reset decoder stats
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
        temperature=temperature,
        max_tokens=max_tokens,
        decode_strategy="jacobi",
        jacobi_block_len=block_len,
        jacobi_max_iterations=128,
    )
    
    token_sequences = []
    generated_texts = []
    start = time.perf_counter()
    
    for i in range(num_samples):
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        token_ids = outputs[0]["token_ids"]
        token_sequences.append(token_ids)
        
        # Decode and store text for first few samples
        if i < print_samples:
            generated_text = llm.tokenizer.decode(token_ids, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        cleanup_scheduler(llm)
    
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
    
    # Print sample generations
    if generated_texts:
        print(f"\n  Sample generations (first {len(generated_texts)}):")
        for i, text in enumerate(generated_texts):
            print(f"\n  [{i+1}] {text[:200]}{'...' if len(text) > 200 else ''}")
    
    return token_sequences, elapsed, acceptance_stats


def compare_distributions(
    ar_distributions: Dict[int, np.ndarray],
    jacobi_distributions: Dict[int, np.ndarray],
    metric: str = "js"
) -> Dict[int, float]:
    """
    Compare distributions at each position using the specified metric.
    
    Args:
        ar_distributions: Token distributions from autoregressive decoding
        jacobi_distributions: Token distributions from Jacobi decoding
        metric: Metric to use ("js", "kl", "tv")
    
    Returns:
        Dictionary mapping position -> similarity score
    """
    comparisons = {}
    
    # Find common positions
    common_positions = set(ar_distributions.keys()) & set(jacobi_distributions.keys())
    
    for pos in sorted(common_positions):
        p = ar_distributions[pos]
        q = jacobi_distributions[pos]
        
        if metric == "js":
            score = jensen_shannon_divergence(p, q)
        elif metric == "kl":
            score = kl_divergence(p, q)
        elif metric == "tv":
            score = total_variation_distance(p, q)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        comparisons[pos] = score
    
    return comparisons


def test_distribution_similarity(
    llm: LLM,
    prompt: str,
    temperature: float = 1.0,
    max_tokens: int = 64,
    num_samples: int = 100,
    block_len: int = 64,
    similarity_threshold: float = 0.1
):
    """
    Test that non-greedy Jacobi decoding produces similar token distributions
    to autoregressive decoding.
    
    Args:
        llm: LLM instance
        prompt: Input prompt
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        num_samples: Number of samples to generate for distribution estimation
        block_len: Jacobi block length
        similarity_threshold: Maximum JS divergence threshold for passing (lower = stricter)
    
    Returns:
        Tuple of (test_passed: bool, results_dict)
    """
    print("\n" + "="*60)
    print("TEST: Distribution Similarity (Non-Greedy Jacobi vs Autoregressive)")
    print("="*60)
    print(f"\nPrompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    print(f"Temperature: {temperature}")
    print(f"Max tokens: {max_tokens}")
    print(f"Number of samples: {num_samples}")
    print(f"Jacobi block length: {block_len}")
    
    # Generate autoregressive samples
    print("\n[Generating Autoregressive Samples]")
    ar_sequences, ar_time = test_nongreedy_autoregressive_decoding(
        llm, prompt, temperature, max_tokens, num_samples, print_samples=3
    )
    print(f"  Generated {len(ar_sequences)} samples in {ar_time:.3f}s")
    print(f"  Avg time per sample: {ar_time/num_samples:.4f}s")
    
    # Generate Jacobi samples
    print("\n[Generating Non-Greedy Jacobi Samples]")
    jacobi_sequences, jacobi_time, jacobi_stats = test_nongreedy_jacobi_decoding(
        llm, prompt, temperature, max_tokens, num_samples, block_len, print_samples=3
    )
    print(f"  Generated {len(jacobi_sequences)} samples in {jacobi_time:.3f}s")
    print(f"  Avg time per sample: {jacobi_time/num_samples:.4f}s")
    
    if jacobi_stats['num_chunk_calls'] > 0:
        avg_tokens_per_call = jacobi_stats['tokens_accepted'] / jacobi_stats['num_chunk_calls']
        print(f"  Avg tokens per call: {avg_tokens_per_call:.2f}")
        if jacobi_stats['num_jacobi_iterations'] > 0:
            avg_tokens_per_iter = jacobi_stats['tokens_accepted'] / jacobi_stats['num_jacobi_iterations']
            print(f"  Avg tokens per iteration (TPF): {avg_tokens_per_iter:.2f}")
    
    # Compute distributions
    print("\n[Computing Token Distributions]")
    vocab_size = None
    if hasattr(llm, 'model_runner') and hasattr(llm.model_runner, 'config'):
        if hasattr(llm.model_runner.config, 'hf_config'):
            vocab_size = llm.model_runner.config.hf_config.vocab_size

    ar_dists = compute_token_distributions(ar_sequences, vocab_size)
    jacobi_dists = compute_token_distributions(jacobi_sequences, vocab_size)
    
    print(f"  AR positions: {len(ar_dists)}")
    print(f"  Jacobi positions: {len(jacobi_dists)}")
    
    # Compare distributions
    print("\n[Comparing Distributions]")
    js_scores = compare_distributions(ar_dists, jacobi_dists, metric="js")
    kl_scores = compare_distributions(ar_dists, jacobi_dists, metric="kl")
    tv_scores = compare_distributions(ar_dists, jacobi_dists, metric="tv")
    
    if not js_scores:
        print("  ⚠️  No common positions to compare")
        return False, {}
    
    # Print statistics
    js_values = list(js_scores.values())
    kl_values = list(kl_scores.values())
    tv_values = list(tv_scores.values())
    
    print(f"\n  [Jensen-Shannon Divergence]")
    print(f"    Mean: {np.mean(js_values):.4f}")
    print(f"    Median: {np.median(js_values):.4f}")
    print(f"    Std: {np.std(js_values):.4f}")
    print(f"    Min: {np.min(js_values):.4f}")
    print(f"    Max: {np.max(js_values):.4f}")
    
    print(f"\n  [KL Divergence]")
    print(f"    Mean: {np.mean(kl_values):.4f}")
    print(f"    Median: {np.median(kl_values):.4f}")
    print(f"    Max: {np.max(kl_values):.4f}")
    
    print(f"\n  [Total Variation Distance]")
    print(f"    Mean: {np.mean(tv_values):.4f}")
    print(f"    Median: {np.median(tv_values):.4f}")
    print(f"    Max: {np.max(tv_values):.4f}")
    
    # Check similarity threshold
    max_js = np.max(js_values)
    mean_js = np.mean(js_values)
    
    print(f"\n[Similarity Check]")
    print(f"  Threshold: JS divergence < {similarity_threshold}")
    print(f"  Max JS divergence: {max_js:.4f}")
    print(f"  Mean JS divergence: {mean_js:.4f}")
    
    # Pass if mean JS divergence is below threshold
    # (Max might have outliers, mean is more robust)
    test_passed = mean_js < similarity_threshold
    
    if test_passed:
        print(f"  ✔️  PASS - Mean JS divergence ({mean_js:.4f}) < threshold ({similarity_threshold})")
    else:
        print(f"  ❌  FAIL - Mean JS divergence ({mean_js:.4f}) >= threshold ({similarity_threshold})")
    
    # Show worst positions
    if len(js_scores) > 0:
        worst_positions = sorted(js_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n  [Worst 5 Positions (by JS divergence)]")
        for pos, js_val in worst_positions:
            print(f"    Position {pos}: JS={js_val:.4f}, KL={kl_scores.get(pos, 0):.4f}, TV={tv_scores.get(pos, 0):.4f}")
    
    # Show generation length statistics
    print(f"\n  [Generation Length Statistics]")
    ar_lengths = [len(seq) for seq in ar_sequences]
    jacobi_lengths = [len(seq) for seq in jacobi_sequences]
    print(f"    AR: mean={np.mean(ar_lengths):.1f}, std={np.std(ar_lengths):.1f}, min={min(ar_lengths)}, max={max(ar_lengths)}")
    print(f"    Jacobi: mean={np.mean(jacobi_lengths):.1f}, std={np.std(jacobi_lengths):.1f}, min={min(jacobi_lengths)}, max={max(jacobi_lengths)}")
    
    # Show example comparison
    if len(ar_sequences) > 0 and len(jacobi_sequences) > 0:
        print(f"\n  [Example Output Comparison]")
        for i in range(min(2, len(ar_sequences), len(jacobi_sequences))):
            ar_text = llm.tokenizer.decode(ar_sequences[i], skip_special_tokens=True)
            jacobi_text = llm.tokenizer.decode(jacobi_sequences[i], skip_special_tokens=True)
            print(f"\n    Example {i+1}:")
            print(f"    AR ({len(ar_sequences[i])} tokens): {ar_text[:150]}{'...' if len(ar_text) > 150 else ''}")
            print(f"    Jacobi ({len(jacobi_sequences[i])} tokens): {jacobi_text[:150]}{'...' if len(jacobi_text) > 150 else ''}")
    
    results = {
        'test_passed': test_passed,
        'num_samples': num_samples,
        'ar_time': ar_time,
        'jacobi_time': jacobi_time,
        'js_scores': js_scores,
        'kl_scores': kl_scores,
        'tv_scores': tv_scores,
        'mean_js': mean_js,
        'max_js': max_js,
        'jacobi_stats': jacobi_stats,
    }
    
    return test_passed, results


def test_nongreedy_autoregressive_decoding_batched(
    llm: LLM, 
    prompts: List[str], 
    temperature: float,
    max_tokens: int,
    num_samples_per_prompt: int,
    batch_size: int = 6
) -> Tuple[List[List[List[int]]], float]:
    """
    Generate multiple samples per prompt using batched non-greedy autoregressive decoding.
    
    Args:
        llm: LLM instance
        prompts: List of prompts to generate samples for
        temperature: Sampling temperature
        max_tokens: Maximum tokens per sample
        num_samples_per_prompt: Number of samples to generate for each prompt
        batch_size: Number of samples to generate in parallel per batch
    
    Returns:
        Tuple of (token_sequences_per_prompt, elapsed_time)
        token_sequences_per_prompt[i] is a list of token sequences for prompts[i]
    """
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        decode_strategy="autoregressive",
    )
    
    token_sequences_per_prompt = [[] for _ in prompts]
    start = time.perf_counter()
    
    for prompt_idx, prompt in enumerate(prompts):
        # Generate samples in batches
        for batch_start in range(0, num_samples_per_prompt, batch_size):
            batch_end = min(batch_start + batch_size, num_samples_per_prompt)
            batch_prompts = [prompt] * (batch_end - batch_start)
            
            outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
            for output in outputs:
                token_sequences_per_prompt[prompt_idx].append(output["token_ids"])
            cleanup_scheduler(llm)
    
    elapsed = time.perf_counter() - start
    
    return token_sequences_per_prompt, elapsed


def test_nongreedy_jacobi_decoding_batched(
    llm: LLM,
    prompts: List[str],
    temperature: float,
    max_tokens: int,
    num_samples_per_prompt: int,
    block_len: int = 64,
    batch_size: int = 6
) -> Tuple[List[List[List[int]]], float, dict]:
    """
    Generate multiple samples per prompt using batched non-greedy Jacobi decoding.
    
    Args:
        llm: LLM instance
        prompts: List of prompts to generate samples for
        temperature: Sampling temperature
        max_tokens: Maximum tokens per sample
        num_samples_per_prompt: Number of samples to generate for each prompt
        block_len: Jacobi block length
        batch_size: Number of samples to generate in parallel per batch
    
    Returns:
        Tuple of (token_sequences_per_prompt, elapsed_time, stats)
    """
    # Reset decoder stats
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
        temperature=temperature,
        max_tokens=max_tokens,
        decode_strategy="jacobi",
        jacobi_block_len=block_len,
        jacobi_max_iterations=128,
    )
    
    token_sequences_per_prompt = [[] for _ in prompts]
    start = time.perf_counter()
    
    for prompt_idx, prompt in enumerate(prompts):
        # Generate samples in batches
        for batch_start in range(0, num_samples_per_prompt, batch_size):
            batch_end = min(batch_start + batch_size, num_samples_per_prompt)
            batch_prompts = [prompt] * (batch_end - batch_start)
            
            outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
            for output in outputs:
                token_sequences_per_prompt[prompt_idx].append(output["token_ids"])
            cleanup_scheduler(llm)
    
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
    
    return token_sequences_per_prompt, elapsed, acceptance_stats


def test_batch_distribution_similarity(
    llm: LLM,
    prompts: List[str],
    temperature: float = 1.0,
    max_tokens: int = 64,
    num_samples_per_prompt: int = 50,
    block_len: int = 64,
    batch_size: int = 6,
    similarity_threshold: float = 0.1
):
    """
    Test that batched non-greedy Jacobi decoding produces similar token distributions
    to batched autoregressive decoding across multiple prompts.
    
    Args:
        llm: LLM instance
        prompts: List of prompts to test
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate per sample
        num_samples_per_prompt: Number of samples to generate per prompt
        block_len: Jacobi block length
        batch_size: Number of samples to generate in parallel
        similarity_threshold: Maximum JS divergence threshold for passing
    
    Returns:
        Tuple of (test_passed: bool, results_dict)
    """
    print("\n" + "="*60)
    print(f"TEST: Batched Distribution Similarity (Non-Greedy Jacobi vs Autoregressive)")
    print("="*60)
    print(f"\nNumber of prompts: {len(prompts)}")
    print(f"Samples per prompt: {num_samples_per_prompt}")
    print(f"Batch size: {batch_size}")
    print(f"Temperature: {temperature}")
    print(f"Max tokens: {max_tokens}")
    print(f"Jacobi block length: {block_len}")
    
    # Generate autoregressive samples (batched)
    print("\n[Generating Batched Autoregressive Samples]")
    ar_sequences_per_prompt, ar_time = test_nongreedy_autoregressive_decoding_batched(
        llm, prompts, temperature, max_tokens, num_samples_per_prompt, batch_size
    )
    total_ar_samples = sum(len(seqs) for seqs in ar_sequences_per_prompt)
    print(f"  Generated {total_ar_samples} total samples in {ar_time:.3f}s")
    print(f"  Avg time per sample: {ar_time/total_ar_samples:.4f}s")
    for i, seqs in enumerate(ar_sequences_per_prompt):
        print(f"    Prompt {i}: {len(seqs)} samples")
    
    # Generate Jacobi samples (batched)
    print("\n[Generating Batched Non-Greedy Jacobi Samples]")
    jacobi_sequences_per_prompt, jacobi_time, jacobi_stats = test_nongreedy_jacobi_decoding_batched(
        llm, prompts, temperature, max_tokens, num_samples_per_prompt, block_len, batch_size
    )
    total_jacobi_samples = sum(len(seqs) for seqs in jacobi_sequences_per_prompt)
    print(f"  Generated {total_jacobi_samples} total samples in {jacobi_time:.3f}s")
    print(f"  Avg time per sample: {jacobi_time/total_jacobi_samples:.4f}s")
    for i, seqs in enumerate(jacobi_sequences_per_prompt):
        print(f"    Prompt {i}: {len(seqs)} samples")
    
    if jacobi_stats['num_chunk_calls'] > 0:
        avg_tokens_per_call = jacobi_stats['tokens_accepted'] / jacobi_stats['num_chunk_calls']
        print(f"  Avg tokens per call: {avg_tokens_per_call:.2f}")
        if jacobi_stats['num_jacobi_iterations'] > 0:
            avg_tokens_per_iter = jacobi_stats['tokens_accepted'] / jacobi_stats['num_jacobi_iterations']
            print(f"  Avg tokens per iteration (TPF): {avg_tokens_per_iter:.2f}")
    
    # Compute distributions per prompt
    print("\n[Computing Token Distributions Per Prompt]")
    try:
        vocab_size = llm.model_runner.config.hf_config.vocab_size
    except AttributeError:
        vocab_size = None
    
    all_js_scores = []
    all_kl_scores = []
    all_tv_scores = []
    prompt_results = []
    
    for prompt_idx in range(len(prompts)):
        print(f"\n  [Prompt {prompt_idx}]")
        ar_dists = compute_token_distributions(ar_sequences_per_prompt[prompt_idx], vocab_size)
        jacobi_dists = compute_token_distributions(jacobi_sequences_per_prompt[prompt_idx], vocab_size)
        
        print(f"    AR positions: {len(ar_dists)}, Jacobi positions: {len(jacobi_dists)}")
        
        if not ar_dists or not jacobi_dists:
            print(f"    ⚠️  Skipping prompt {prompt_idx}: insufficient data")
            continue
        
        # Compare distributions
        js_scores = compare_distributions(ar_dists, jacobi_dists, metric="js")
        kl_scores = compare_distributions(ar_dists, jacobi_dists, metric="kl")
        tv_scores = compare_distributions(ar_dists, jacobi_dists, metric="tv")
        
        if not js_scores:
            print(f"    ⚠️  No common positions to compare")
            continue
        
        js_values = list(js_scores.values())
        kl_values = list(kl_scores.values())
        tv_values = list(tv_scores.values())
        
        mean_js = np.mean(js_values)
        max_js = np.max(js_values)
        mean_kl = np.mean(kl_values)
        mean_tv = np.mean(tv_values)
        
        all_js_scores.extend(js_values)
        all_kl_scores.extend(kl_values)
        all_tv_scores.extend(tv_values)
        
        prompt_passed = mean_js < similarity_threshold
        status = "✔️ PASS" if prompt_passed else "❌ FAIL"
        print(f"    {status} - Mean JS: {mean_js:.4f}, Max JS: {max_js:.4f}")
        
        prompt_results.append({
            'prompt_idx': prompt_idx,
            'mean_js': mean_js,
            'max_js': max_js,
            'mean_kl': mean_kl,
            'mean_tv': mean_tv,
            'num_positions': len(js_scores),
            'passed': prompt_passed
        })
    
    if not all_js_scores:
        print("\n  ⚠️  No valid comparisons found")
        return False, {}
    
    # Overall statistics
    print("\n[Overall Statistics Across All Prompts]")
    print(f"\n  [Jensen-Shannon Divergence]")
    print(f"    Mean: {np.mean(all_js_scores):.4f}")
    print(f"    Median: {np.median(all_js_scores):.4f}")
    print(f"    Std: {np.std(all_js_scores):.4f}")
    print(f"    Min: {np.min(all_js_scores):.4f}")
    print(f"    Max: {np.max(all_js_scores):.4f}")
    
    print(f"\n  [KL Divergence]")
    print(f"    Mean: {np.mean(all_kl_scores):.4f}")
    print(f"    Median: {np.median(all_kl_scores):.4f}")
    print(f"    Max: {np.max(all_kl_scores):.4f}")
    
    print(f"\n  [Total Variation Distance]")
    print(f"    Mean: {np.mean(all_tv_scores):.4f}")
    print(f"    Median: {np.median(all_tv_scores):.4f}")
    print(f"    Max: {np.max(all_tv_scores):.4f}")
    
    # Check similarity threshold
    overall_mean_js = np.mean(all_js_scores)
    overall_max_js = np.max(all_js_scores)
    prompts_passed = sum(1 for r in prompt_results if r['passed'])
    
    print(f"\n[Similarity Check]")
    print(f"  Threshold: JS divergence < {similarity_threshold}")
    print(f"  Overall mean JS divergence: {overall_mean_js:.4f}")
    print(f"  Overall max JS divergence: {overall_max_js:.4f}")
    print(f"  Prompts passed: {prompts_passed}/{len(prompt_results)}")
    
    # Pass if overall mean JS divergence is below threshold
    test_passed = overall_mean_js < similarity_threshold
    
    if test_passed:
        print(f"  ✔️  PASS - Overall mean JS divergence ({overall_mean_js:.4f}) < threshold ({similarity_threshold})")
    else:
        print(f"  ❌  FAIL - Overall mean JS divergence ({overall_mean_js:.4f}) >= threshold ({similarity_threshold})")
    
    # Show worst prompts
    if prompt_results:
        worst_prompts = sorted(prompt_results, key=lambda x: x['mean_js'], reverse=True)[:3]
        print(f"\n  [Worst 3 Prompts (by mean JS divergence)]")
        for result in worst_prompts:
            print(f"    Prompt {result['prompt_idx']}: Mean JS={result['mean_js']:.4f}, "
                  f"Max JS={result['max_js']:.4f}, Positions={result['num_positions']}")
    
    results = {
        'test_passed': test_passed,
        'num_prompts': len(prompts),
        'num_samples_per_prompt': num_samples_per_prompt,
        'ar_time': ar_time,
        'jacobi_time': jacobi_time,
        'overall_mean_js': overall_mean_js,
        'overall_max_js': overall_max_js,
        'prompts_passed': prompts_passed,
        'prompt_results': prompt_results,
        'jacobi_stats': jacobi_stats,
    }
    
    return test_passed, results


def main():
    parser = argparse.ArgumentParser(description="Test non-greedy Jacobi decoding distribution similarity")
    parser.add_argument("--model-path", type=str, default=os.path.expanduser("~/huggingface/Qwen3-0.6B/"),
                        help="Path to model directory")
    parser.add_argument("--tokenizer-path", type=str, default=None,
                        help="Path to tokenizer directory (defaults to model path)")
    parser.add_argument("--enforce-eager", action="store_true", default=False,
                        help="Disable CUDA graphs")
    parser.add_argument("--max-tokens", type=int, default=64,
                        help="Max tokens to generate")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples to generate for distribution estimation")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--block-len", type=int, default=64,
                        help="Jacobi block length")
    parser.add_argument("--similarity-threshold", type=float, default=0.1,
                        help="Maximum JS divergence threshold for passing (default: 0.1)")
    args = parser.parse_args()
    
    print("="*60)
    print("Nano-vLLM Non-Greedy Jacobi Decoding Distribution Test")
    print("="*60)
    print(f"Model: {args.model_path}")
    print(f"Temperature: {args.temperature}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Similarity threshold: {args.similarity_threshold}")
    
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
        print("\nMake sure you have downloaded the model:")
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
    
    # Apply chat template to single prompt (use first prompt)
    try:
        messages = build_messages(code_prompts[0])
        formatted_prompt = llm.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except:
        formatted_prompt = code_prompts[0]
    
    # Warmup
    print("\nWarmup run...")
    llm.generate(["Hello"], SamplingParams(temperature=0.0, max_tokens=8, decode_strategy="autoregressive"), use_tqdm=False)
    cleanup_scheduler(llm)
    
    # Quick generation preview
    print("\n" + "="*60)
    print("QUICK GENERATION PREVIEW")
    print("="*60)
    print(f"\nPrompt preview: {formatted_prompt[:150]}...")
    
    print("\n[Autoregressive - 2 samples]")
    ar_samples = llm.generate([formatted_prompt] * 2, SamplingParams(
        temperature=args.temperature, max_tokens=min(32, args.max_tokens), 
        decode_strategy="autoregressive"
    ), use_tqdm=False)
    for i, sample in enumerate(ar_samples):
        text = llm.tokenizer.decode(sample["token_ids"], skip_special_tokens=True)
        print(f"  Sample {i+1} ({len(sample['token_ids'])} tokens): {text[:120]}...")
    cleanup_scheduler(llm)
    
    print("\n[Jacobi - 2 samples]")
    jacobi_samples = llm.generate([formatted_prompt] * 2, SamplingParams(
        temperature=args.temperature, max_tokens=min(32, args.max_tokens),
        decode_strategy="jacobi", jacobi_block_len=args.block_len, jacobi_max_iterations=128
    ), use_tqdm=False)
    for i, sample in enumerate(jacobi_samples):
        text = llm.tokenizer.decode(sample["token_ids"], skip_special_tokens=True)
        print(f"  Sample {i+1} ({len(sample['token_ids'])} tokens): {text[:120]}...")
    cleanup_scheduler(llm)
    
    # Run single prompt test
    test_passed, results = test_distribution_similarity(
        llm,
        formatted_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_samples=args.num_samples,
        block_len=args.block_len,
        similarity_threshold=args.similarity_threshold
    )
    
    # Test prompts for batched testing (use subset of code_prompts)
    test_prompts_batched = [
        code_prompts[1],  # Fibonacci
        code_prompts[2],  # Min-heap push
        code_prompts[3],  # Bubble sort
    ]
    
    # Apply chat template to batched prompts
    formatted_prompts_batched = []
    for code_prompt in test_prompts_batched:
        messages = build_messages(code_prompt)
        formatted = llm.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_prompts_batched.append(formatted)
    
    # Run batched test
    cleanup_scheduler(llm)
    batch_test_passed, batch_results = test_batch_distribution_similarity(
        llm,
        formatted_prompts_batched,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        num_samples_per_prompt=min(50, args.num_samples),  # Use smaller number for batched test
        block_len=args.block_len,
        batch_size=6,
        similarity_threshold=args.similarity_threshold
    )
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    status = "PASSED" if test_passed else "FAILED"
    print(f"  Single Prompt Distribution Similarity Test: {status}")
    
    if results:
        print(f"\n  Mean JS divergence: {results['mean_js']:.4f}")
        print(f"  Max JS divergence: {results['max_js']:.4f}")
        print(f"  AR time: {results['ar_time']:.3f}s")
        print(f"  Jacobi time: {results['jacobi_time']:.3f}s")
        if results['jacobi_time'] > 0:
            speedup = results['ar_time'] / results['jacobi_time']
            print(f"  Speedup: {speedup:.2f}x")
    
    batch_status = "PASSED" if batch_test_passed else "FAILED"
    print(f"\n  Batched Distribution Similarity Test: {batch_status}")
    
    if batch_results:
        print(f"\n  Batched test - Mean JS divergence: {batch_results['overall_mean_js']:.4f}")
        print(f"  Batched test - Max JS divergence: {batch_results['overall_max_js']:.4f}")
        print(f"  Batched test - Prompts passed: {batch_results['prompts_passed']}/{batch_results['num_prompts']}")
        print(f"  Batched AR time: {batch_results['ar_time']:.3f}s")
        print(f"  Batched Jacobi time: {batch_results['jacobi_time']:.3f}s")
        if batch_results['jacobi_time'] > 0:
            batch_speedup = batch_results['ar_time'] / batch_results['jacobi_time']
            print(f"  Batched speedup: {batch_speedup:.2f}x")
    
    all_passed = test_passed and batch_test_passed
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())