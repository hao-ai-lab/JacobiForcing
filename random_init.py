from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from einops import rearrange
import torch.nn.functional as F
import torch
import random
import math

# sampling helpers
def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(-1, ind, val)
    return probs

def safe_div(num, den, eps = 1e-10):
    return num / max(den, eps)

def find_first_true_index(bool_tensor, dim = -1):
    return (bool_tensor.cumsum(dim = dim) == 0).sum(dim = dim)

#TODO: support bsz>1 for Diffusion Decoding
@torch.inference_mode()
def diffusion_decoding(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    n_token_seq_len,
    filter_thres=0.9,
    temperature = 1.,
    lenience = 1.,
    ):

    batch, prompt_len, out, device = 1, int(torch.sum(attention_mask[0])), input_ids.clone(), input_ids.device
    seq_lens = torch.full((batch,), prompt_len, device = device, dtype = torch.long)

    ### Initialization draft distribution q(x) with 0-1 distribution from prompt
    q_sampled = []
    q_logits_all = []
    for _ in range(n_token_seq_len):
        q_sample = torch.tensor([random.choice(input_ids[0].tolist())]).to(dtype=torch.long, device=model.device).unsqueeze(dim=0)
        out = torch.cat((out, q_sample), dim=1)
        q_logits = torch.full((batch, 151936), float('-inf'), device=model.device)
        q_logits.scatter_(1, q_sample, 0.0) 
        q_sampled.append(q_sample)
        q_logits_all.append(q_logits)
    q_sampled = torch.cat(q_sampled, dim = 1)
    q_logits_all = torch.stack(q_logits_all, dim = -2)
    q_logits = q_logits_all


    ### Diffusion decoding
    total_accepted = 0
    itr=0
    while total_accepted < n_token_seq_len:

        ### verify and speculate with larger network within a forward pass
        out_attention_mask = torch.full_like(out, 1).to(model.device)
        logits = model(out, out_attention_mask).logits
        p_logits = logits[:, prompt_len+total_accepted-1:, :]
        p_logits = top_k(p_logits, thres = filter_thres)

        ### prob and prob of draft distribution (p(x) and q(x))
        p_prob = safe_div(p_logits, temperature).softmax(dim = -1)[:, :, :len(tokenizer)]
        q_prob = safe_div(q_logits, temperature).softmax(dim = -1)[:, :, :len(tokenizer)]

        p, prob_next = p_prob[:, :-1], p_prob[:, -1]

        p = p.gather(-1, q_sampled.unsqueeze(dim=-1))
        q = q_prob.gather(-1, q_sampled.unsqueeze(dim=-1)) * lenience

        p, q = [rearrange(t, 'b n 1 -> b n') for t in (p, q)]
        r = random_uniform = torch.zeros_like(q).float().uniform_(0, 1)

        accepted = find_first_true_index(r > (p / q))
        total_accepted += int(accepted[0])
        accepted.clamp_(max = n_token_seq_len - 1)

        out = out[:, :prompt_len+total_accepted]

        ### sample the additional token to better bound the worst case
        if int(accepted[0]) == q_prob.shape[1]:
            print(f'acc: {accepted[0]}, p.shape: {p.shape}, q.shape: {q.shape}')
            print(f'p_prob.shape: {p_prob.shape}, q_prob.shape: {q_prob.shape}, int(acc): {int(accepted[0])}')
        else:
            adjusted_prob = F.relu(p_prob[:, int(accepted[0]), :] - q_prob[:, int(accepted[0]), :])
            adjusted_prob = adjusted_prob / adjusted_prob.sum(dim = -1, keepdim = True)

            has_rejected = torch.tensor(total_accepted < n_token_seq_len, device=out.device)
            prob_next = torch.where(
                rearrange(has_rejected, '... -> ... 1'),
                adjusted_prob,
                prob_next
            )
            next_token = torch.multinomial(prob_next, 1)

            out = torch.cat((out, next_token), dim = -1)
            total_accepted += 1

        ### update q(x) with self-speculated p(x) and sample new drafts tokens
        q_logits = p_logits[:, int(accepted[0])+1:-1, :]
        q_sampled = gumbel_sample(q_logits, temperature = temperature, dim = -1)
        out = torch.cat((out, q_sampled), dim = -1)

        generated_str = tokenizer.batch_decode(out[0, prompt_len:], skip_special_tokens=False)
        print(f'Itr: {itr}, Accepted tokens: {int(accepted[0])}')
        # generated_str = ''.join(generated_str)
        # print(f'Generated string: {generated_str}')
            
        itr+=1
    
    return out

### Load dataset...
dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
problem = dataset["problem"][0]

# prompt = """
# Solve the following math problem efficiently and clearly. Please reason step by step, 
# separate logical reasoning steps with two newline characters (\n\n), and put your final answer within \\boxed{{}}.
# Problem: {problem}
# """
prompt = """
Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. Please structure your response into two main sections: Thought and Solution. In the Thought section, detail your reasoning process using the specified format: <|begin_of_thought|> {{thought with steps separated with '\n\n'}} <|end_of_thought|> Each step should include detailed considerations such as analisying questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. The solution should remain a logical, accurate, concise expression style and detail necessary step needed to reach the conclusion, formatted as follows: <|begin_of_solution|> {{final formatted, precise, and clear solution}} <|end_of_solution|> Now, try to solve the following question through the above guidelines: {problem}
"""
prompt = prompt.format(problem=problem)

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='cuda',
    torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
print(f'Prompt from user: {text}')
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
input_ids = model_inputs['input_ids']
attention_mask = torch.full_like(input_ids, 1).to(model.device)

### Decoding with Diffusion decoding
while not (input_ids == tokenizer.eos_token_id).any():
    
    generated_ids = diffusion_decoding(
        model,
        tokenizer,
        input_ids=input_ids,
        attention_mask=attention_mask,
        n_token_seq_len=16,
        filter_thres=0.9,
        temperature = 1.,
        lenience = 1.
        )

    input_ids = generated_ids
    attention_mask = torch.full_like(input_ids, 1).to(model.device)

generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(f'---------Generated Answer----------')
print(response)