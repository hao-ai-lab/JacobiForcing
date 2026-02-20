import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # For greedy decoding (temperature=0), just return argmax
        # For sampling (temperature>0), use Gumbel-max trick
        is_greedy = (temperatures == 0.0).all()
        
        if is_greedy:
            # Pure greedy: argmax without any randomness
            return logits.argmax(dim=-1)
        else:
            # Sampling with temperature
            logits = logits.float().div_(temperatures.unsqueeze(dim=1))
            probs = torch.softmax(logits, dim=-1)
            sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
            return sample_tokens
