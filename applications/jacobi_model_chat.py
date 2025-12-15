import time
import threading

import torch
import streamlit as st
from transformers import Qwen2ForCausalLM, AutoTokenizer
from transformers.generation.streamers import TextIteratorStreamer

import sys
from pathlib import Path
path_root = Path(__file__).parents[1]
sys.path.append(str(path_root))

from applications.jacobi_streaming_driver import jacobi_stream_chat

# --------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------

@st.cache_resource
def load_model_and_tokenizer(model_path: str, tokenizer_path: str):
    """
    Load Qwen2 model + tokenizer. If your modeling module is available,
    monkey-patch jacobi_forward_greedy_multiblock onto Qwen2ForCausalLM.
    """
    # Detect device (prefer cuda:0)
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    dtype = torch.bfloat16 if "cuda" in device else torch.float32
    attn_impl = "flash_attention_2" if "cuda" in device else "eager"

    # Try to patch your custom decoding method (optional, doesn't crash if missing)
    try:
        path_root = Path(__file__).parents[1]
        sys.path.append(str(path_root))

        from modeling.cllm2_qwen2_modeling_kv_terminate_on_eos_improved_multiblock_lookahead_unified import (
            jacobi_forward_greedy_multiblock,
        )

        Qwen2ForCausalLM.jacobi_forward_greedy_multiblock = jacobi_forward_greedy_multiblock
        patched = True

    except Exception:
        patched = False

    print(f"enabling patch: {patched}")

    # Load model with flash-attn (where possible)
    model = Qwen2ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map="cuda",
        attn_implementation=attn_impl,
    )

    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device, patched


# --------------------------------------------------------------------
# Generation helpers
# --------------------------------------------------------------------


def stream_answer_hf_generate(
    model: Qwen2ForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
    messages,
    placeholder,
    max_new_tokens: int = 1024,
    temperature: float = 0.8,
    top_p: float = 0.2,
):
    """
    Streaming generation using HuggingFace generate() + TextIteratorStreamer.
    Updates `placeholder` in real time.

    Returns:
        assistant_text: full decoded assistant text
        new_token_count: # tokens in assistant_text
        gen_time_sec: time spent INSIDE model.generate (forward pass only)
    """
    # Build chat prompt using HF chat template
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer(
        [prompt],
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_special_tokens=True,
        skip_prompt=True,
    )

    gen_kwargs = dict(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature is not None and temperature > 0),
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        streamer=streamer,
    )

    # ---- forward-pass-only timing around model.generate ----
    forward_time = [0.0]
    use_cuda = isinstance(device, str) and ("cuda" in device)

    def _run_generate():
        #if use_cuda:
        #    torch.cuda.synchronize()
        
        t0 = time.perf_counter()
        model.generate(**gen_kwargs)
        
        #if use_cuda:
        #    torch.cuda.synchronize()
        
        t1 = time.perf_counter()
        forward_time[0] = t1 - t0

    # Run generate in a background thread so we can iterate over streamer
    thread = threading.Thread(target=_run_generate)
    thread.start()

    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        # live update
        placeholder.markdown(generated_text)

    # Ensure generate has completed and timing is recorded
    thread.join()

    gen_time = forward_time[0]

    # Count tokens in the generated text only
    if generated_text.strip():
        token_ids = tokenizer(
            generated_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).input_ids
        new_tokens = int(token_ids.shape[1])
    else:
        new_tokens = 0

    return generated_text.strip(), token_ids, new_tokens, gen_time


# --------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------

st.set_page_config(page_title="Jacobi Forcing Model", layout="wide")

st.title("Jacobi Forcing Model Chatbot Demo")

st.markdown(
    """
The model is beta and fine-tuned on coding dataset. For optimal speedup please test coding tasks.\n

Uncheck `Jacobi decoding (MR)` on the left panel to test the AR baseline.\n

The **cumulative tokens per second** tracks TPS across the whole session.
"""
)

# ---------------- Sidebar: model config & stats ----------------

default_model_path = "/raid/lah003/shiftedattn-10-16-7b-qwen2p5-coder-n32w16-n16distill-data-v2-ar-1-cyclic-noise-all-1e-6/ckpt-344092"
default_tokenizer_path = "/home/lah003/models/Qwen2.5-Coder-7B-Instruct"

st.sidebar.header("Model settings")

model_path = st.sidebar.text_input("Model path", value=default_model_path)
tokenizer_path = st.sidebar.text_input("Tokenizer path", value=default_tokenizer_path)

max_new_tokens = st.sidebar.slider(
    "Max new tokens per reply",
    min_value=32,
    max_value=2048,
    value=1024,
    step=32,
)
temperature = st.sidebar.slider(
    "Temperature (0 = greedy)",
    min_value=0.0,
    max_value=1.5,
    value=0.0,
    step=0.05,
)

#top_p = st.sidebar.slider(
#    "Top-p",
#    min_value=0.1,
#    max_value=1.0,
#    value=0.20,
#    step=0.05,
#)
top_p = 0.20

use_jacobi = st.sidebar.checkbox(
    "jacobi decoding (MR)", value=True
)

# Jacobi-specific knobs (FOLDABLE)
with st.sidebar.expander("Jacobi decoding parameters", expanded=False):
    n_token_seq_len = st.number_input(
        "Block size (n_token_seq_len)",
        min_value=4,
        max_value=512,
        value=64,
        step=4,
    )
    K = st.number_input(
        "Block count K",
        min_value=1,
        max_value=16,
        value=2,
        step=1,
    )
    r = st.slider(
        "Spawn threshold r (fraction of block)",
        min_value=0.0,
        max_value=1.0,
        value=0.85,
        step=0.05,
    )
    n_gram_pool_size = st.number_input(
        "N-gram pool size",
        min_value=1,
        max_value=64,
        value=4,
        step=1,
    )

# Load model/tokenizer only once
model, tokenizer, device, jacobi_decoding = load_model_and_tokenizer(
    model_path=model_path,
    tokenizer_path=tokenizer_path,
)

st.sidebar.write(f"**Device:** `{device}`")
st.sidebar.write(f"**flash_attention_2:** `{'cuda' in device}`")
st.sidebar.write(f"**Jacobi decoding available:** `{jacobi_decoding}`")

# ---------------- Session state init ----------------

if "messages" not in st.session_state:
    st.session_state.messages = []

#if "system_prompt" not in st.session_state:
#    st.session_state.system_prompt = (
#        "You are a helpful coding assistant. Answer as concisely as possible."
#    )

if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0

if "total_time" not in st.session_state:
    st.session_state.total_time = 0.0

# System prompt editor
#st.sidebar.subheader("System prompt")
#new_system_prompt = st.sidebar.text_area(
#    "System prompt",
#    value=st.session_state.system_prompt,
#    height=150,
#)
#st.session_state.system_prompt = new_system_prompt

# Reset button
if st.sidebar.button("Reset conversation & stats"):
    st.session_state.messages = []
    st.session_state.total_tokens = 0
    st.session_state.total_time = 0.0
    st.rerun()

# ---------------- METRIC PLACEHOLDERS + helper ----------------

col1, col2, col3 = st.columns(3)
metric_tokens = col1.empty()
metric_time = col2.empty()
metric_tps = col3.empty()

def update_metrics():
    total_tokens = st.session_state.total_tokens
    total_time = st.session_state.total_time
    cumulative_tps = total_tokens / total_time if total_time > 0 else 0.0

    metric_tokens.metric("Total generated tokens", total_tokens)
    metric_time.metric("Total generation time (s)", f"{total_time:.2f}")
    metric_tps.metric("Cumulative TPS", f"{cumulative_tps:.2f}")

# initial render of metrics
update_metrics()

st.markdown("---")

# ---------------- Chat history rendering ----------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- Chat input & generation ----------------

user_input = st.chat_input("I am a coding expert...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Build full message list including system
    full_messages = []
    #if st.session_state.system_prompt:
    #    full_messages.append({"role": "system", "content": st.session_state.system_prompt})
    full_messages.extend(st.session_state.messages)
    
    prompt = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Generate assistant response
    with st.chat_message("assistant"):
        # Placeholder that we'll update in streaming mode
        placeholder = st.empty()

        if use_jacobi and jacobi_decoding:
            # Streaming Jacobi decoding
            # gen_time here is forward-only time (sum of jacobi_forward_greedy_multiblock calls)
            assistant_text, output_ids, new_tokens, gen_time = jacobi_stream_chat(
                model=model,
                tokenizer=tokenizer,
                messages=full_messages,
                n_token_seq_len=int(n_token_seq_len),
                max_new_tokens=max_new_tokens,
                K=int(K),
                r=float(r),
                n_gram_pool_size=int(n_gram_pool_size),
                on_text=lambda text: placeholder.markdown(text),
            )
        else:
            # Streaming HF generate with flash-attn
            # gen_time here is forward-only time inside model.generate
            assistant_text, output_ids, new_tokens, gen_time = stream_answer_hf_generate(
                model=model,
                tokenizer=tokenizer,
                device=device,
                messages=full_messages,
                placeholder=placeholder,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

        # Update stats in session state (atomic accumulation of forward time)
        st.session_state.messages.append({"role": "assistant", "content": assistant_text})
        st.session_state.total_tokens += new_tokens
        st.session_state.total_time += gen_time

        # Update metrics NOW (no click required)
        update_metrics()

        # Per-turn stats under the message
        turn_tps = new_tokens / gen_time if gen_time > 0 else 0.0
        cum_tps = (
            st.session_state.total_tokens / st.session_state.total_time
            if st.session_state.total_time > 0
            else 0.0
        )
        st.caption(
            f"Generated {new_tokens} tokens in {gen_time:.4f}s "
            f"({turn_tps:.2f} tok/s this turn, {cum_tps:.2f} tok/s cumulative)"
        )
