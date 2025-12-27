import math
from typing import Optional

import torch
import torch.nn as nn

from src.utils import subsequent_mask


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        # self.register_buffer("bias", torch.zeros(d_model), persistent=False)    # for backward compatibility

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class   SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class RelativePositionBias(nn.Module):
    """
    Lightweight relative positional bias similar to T5.
    Returns per-head bias that can be added to attention scores.
    """

    def __init__(self, num_buckets: int = 32, max_distance: int = 128, num_heads: int = 8):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.num_heads = num_heads
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
    def _relative_position_bucket(self, relative_position: torch.Tensor) -> torch.Tensor:
        num_buckets = self.num_buckets
        max_distance = self.max_distance
        ret = 0
        relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact
        val_if_large = max_exact + (
            (torch.log(relative_position.float() / max_exact + 1) / math.log(max_distance / max_exact))
            * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))
        ret = torch.where(is_small, relative_position, val_if_large)
        return ret

    def forward(self, qlen: int, klen: int, device: torch.device) -> torch.Tensor:
        context_position = torch.arange(qlen, dtype=torch.long, device=device)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position
        rp_bucket = self._relative_position_bucket(relative_position)
        values = self.relative_attention_bias(rp_bucket)  # (qlen, klen, num_heads)
        # reshape to (num_heads, qlen, klen) to be compatible with attention mask broadcasting
        return values.permute(2, 0, 1)


def build_causal_mask(length: int, device: torch.device) -> torch.Tensor:
    """Build a float causal mask with -inf on future positions."""
    mask = subsequent_mask(length).to(device)  # shape (1, L, L) True on allowed positions
    float_mask = torch.zeros(length, length, device=device)
    float_mask = float_mask.masked_fill(~mask[0], float("-inf"))
    return float_mask


def combine_bias_and_mask(
    base_mask: Optional[torch.Tensor],  # (L,S) float, e.g. 0/-inf
    bias: Optional[torch.Tensor],       # (H,L,S) or (L,S)
    batch_size: int,
    num_heads: int,
) -> Optional[torch.Tensor]:
    """
    Return attn_mask suitable for torch MultiheadAttention.
    - If 2D: (L,S)
    - If 3D: (B*H,L,S)
    """
    if bias is None and base_mask is None:
        return None

    # Case 1: bias is per-head (H,L,S) -> expand to (B*H,L,S)
    if bias is not None and bias.dim() == 3:
        # add base_mask (L,S) -> broadcast to (H,L,S)
        if base_mask is not None:
            bias = bias + base_mask  # broadcast

        # expand to (B,H,L,S) then reshape
        bias = bias.unsqueeze(0).expand(batch_size, -1, -1, -1)  # (B,H,L,S)
        return bias.reshape(batch_size * num_heads, bias.size(-2), bias.size(-1))  # (B*H,L,S)

    # Case 2: bias is 2D (L,S)
    out = bias
    if base_mask is not None:
        out = out + base_mask if out is not None else base_mask
    return out


def replace_layer_norm(module: nn.Module, norm_type: str):
    if norm_type != "rmsnorm":
        return
    for name, child in module.named_children():
        if isinstance(child, nn.LayerNorm):
            setattr(module, name, RMSNorm(child.normalized_shape[0], child.eps))
        else:
            replace_layer_norm(child, norm_type)


class TransformerNMT(nn.Module):
    def __init__(
        self,
        src_vocab,
        tgt_vocab,
        d_model=512,
        nhead=4,
        num_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        embedding_weight: torch.Tensor = None,
        padding_idx: int = 0,
        pos_embedding: str = "absolute",
        norm_type: str = "layernorm",
        max_pos_len: int = 512,
        relative_num_buckets: int = 32,
        relative_max_distance: int = 128,
    ):
        super().__init__()
        effective_dim = embedding_weight.size(1) if embedding_weight is not None else d_model
        self.padding_idx = padding_idx
        self.pos_embedding = pos_embedding
        self.nhead = nhead

        if embedding_weight is not None:
            self.src_emb = nn.Embedding.from_pretrained(
                embedding_weight, freeze=False, padding_idx=padding_idx
            )
            self.tgt_emb = nn.Embedding.from_pretrained(
                embedding_weight, freeze=False, padding_idx=padding_idx
            )
        else:
            self.src_emb = nn.Embedding(src_vocab, effective_dim, padding_idx=padding_idx)
            self.tgt_emb = nn.Embedding(tgt_vocab, effective_dim, padding_idx=padding_idx)

        self.pe = SinusoidalPositionalEncoding(max_pos_len, effective_dim)
        self.relative_bias = (
            RelativePositionBias(relative_num_buckets, relative_max_distance, nhead)
            if pos_embedding == "relative"
            else None
        )

        self.transformer = nn.Transformer(
            d_model=effective_dim,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            bias=False
        )
        if norm_type == "rmsnorm":
            replace_layer_norm(self.transformer, norm_type)
        self.generator = nn.Linear(effective_dim, tgt_vocab)

    def _apply_positional_encoding(self, x: torch.Tensor) -> torch.Tensor:
        if self.pos_embedding == "absolute":
            return self.pe(x)
        return x

    def forward(self, batch):
        src_key_padding_mask = batch.src == self.padding_idx
        tgt_key_padding_mask = batch.tgt_input == self.padding_idx

        src = self._apply_positional_encoding(self.src_emb(batch.src))
        tgt = self._apply_positional_encoding(self.tgt_emb(batch.tgt_input))

        tgt_len = batch.tgt_input.size(1)
        src_len = batch.src.size(1)
        tgt_causal_mask = build_causal_mask(tgt_len, batch.tgt_input.device)
        tgt_bias = self.relative_bias(tgt_len, tgt_len, batch.tgt_input.device) if self.relative_bias is not None else None
        src_bias = self.relative_bias(src_len, src_len, batch.src.device) if self.relative_bias is not None else None
        tgt_mask = combine_bias_and_mask(tgt_causal_mask, tgt_bias, batch.tgt_input.size(0), self.nhead)
        src_mask = combine_bias_and_mask(None, src_bias, batch.src.size(0), self.nhead)

        out = self.transformer(
            src,
            tgt,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        logits = self.generator(out)
        return logits

    def greedy_decode(self, src, max_len, sos_idx, eos_idx):
        self.eval()
        with torch.no_grad():
            src_key_padding_mask = src == self.padding_idx
            src_emb = self._apply_positional_encoding(self.src_emb(src))
            memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

            ys = torch.tensor([[sos_idx]], device=src.device)
            outputs = []
            for _ in range(max_len):
                tgt_emb = self._apply_positional_encoding(self.tgt_emb(ys))
                tgt_len = ys.size(1)
                tgt_causal_mask = build_causal_mask(tgt_len, ys.device)
                tgt_bias = self.relative_bias(tgt_len, tgt_len, ys.device) if self.relative_bias is not None else None
                tgt_mask = combine_bias_and_mask(tgt_causal_mask, tgt_bias, ys.size(0), self.nhead)
                out = self.transformer.decoder(
                    tgt_emb,
                    memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=src_key_padding_mask,
                )
                logits = self.generator(out[:, -1, :])
                next_token = logits.argmax(-1)
                outputs.append(next_token.item())
                ys = torch.cat([ys, next_token.unsqueeze(0)], dim=1)
                if next_token.item() == eos_idx:
                    break
        return outputs

    def beam_search(self, src, beam_size, max_len, sos_idx, eos_idx):
        self.eval()
        with torch.no_grad():
            src_key_padding_mask = src == self.padding_idx
            src_emb = self._apply_positional_encoding(self.src_emb(src))
            memory = self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)
            
            # Initialize beams with (sequence, score)
            beams = [([sos_idx], 0.0)]
            
            for _ in range(max_len):
                new_beams = []
                for seq, score in beams:
                    # Stop expanding if sequence already ends with EOS
                    if seq[-1] == eos_idx:
                        new_beams.append((seq, score))
                        continue
                    
                    # Prepare input tensor from current sequence
                    ys = torch.tensor([seq], device=src.device)
                    
                    # Decode next token
                    tgt_emb = self._apply_positional_encoding(self.tgt_emb(ys))
                    tgt_len = ys.size(1)
                    tgt_causal_mask = build_causal_mask(tgt_len, ys.device)
                    tgt_bias = self.relative_bias(tgt_len, tgt_len, ys.device) if self.relative_bias is not None else None
                    tgt_mask = combine_bias_and_mask(tgt_causal_mask, tgt_bias, ys.size(0), self.nhead)
                    out = self.transformer.decoder(
                        tgt_emb,
                        memory,
                        tgt_mask=tgt_mask,
                        memory_key_padding_mask=src_key_padding_mask,
                    )
                    logits = self.generator(out[:, -1, :])
                    log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)
                    
                    # Get top k candidates
                    topk = torch.topk(log_probs, beam_size)
                    for prob, idx in zip(topk.values, topk.indices):
                        new_seq = seq + [idx.item()]
                        new_score = score + prob.item()
                        new_beams.append((new_seq, new_score))
                
                # Keep only top beam_size candidates
                beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
                
                # If all beams end with EOS, stop early
                if all(seq[-1] == eos_idx for seq, _ in beams):
                    break
            
            # Return the sequence with highest score
            return beams[0][0]