
import torch
from torch.nn import functional as F

@torch.no_grad()
def compute_perplexity(logits, tgt, pad_id) -> float:
    B, T = tgt.shape
    perplexity = F.cross_entropy(logits[:, :T, :].reshape(B*(T-1), -1), tgt[:, 1:].reshape(-1), reduction='mean',
                                 ignore_index=pad_id)
    perplexity = perplexity.exp()

    return perplexity.item()