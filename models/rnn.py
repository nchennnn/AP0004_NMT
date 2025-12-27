from typing import Tuple

import torch
import torch.nn as nn

from src.utils import PAD_TOKEN, EOS_TOKEN


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_size,
        hidden_size,
        num_layers=2,
        dropout=0.25,
        embedding_weight: torch.Tensor = None,
        padding_idx: int = 0,
    ):
        super().__init__()
        self.padding_idx = padding_idx

        if embedding_weight is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embedding_weight, freeze=False, padding_idx=padding_idx
            )
        else:
            self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        self.rnn = nn.LSTM(emb_size, hidden_size, num_layers=num_layers, batch_first=True, bias=True, bidirectional=False, dropout=dropout)


    def forward(self, src, lengths=None):
        embedded = self.embedding(src)
        if lengths is None:
            lengths = (src != self.padding_idx).sum(dim=1)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden


def dot_attention(query, keys, values, mask=None):
    scores = torch.bmm(keys, query.unsqueeze(-1)).squeeze(-1)
    if mask is not None:
        scores_length = scores.shape[1]
        mask = mask[:, :scores_length]
        scores = scores.masked_fill(~mask, -1e9)
    attn = torch.softmax(scores, dim=-1)
    context = torch.bmm(attn.unsqueeze(1), values).squeeze(1)
    return context, attn


def general_attention(query, keys, values, linear, mask=None):
    proj = linear(keys)
    scores = torch.bmm(proj, query.unsqueeze(-1)).squeeze(-1)
    if mask is not None:
        scores_length = scores.shape[1]
        mask = mask[:, :scores_length]
        scores = scores.masked_fill(~mask, -1e9)
    attn = torch.softmax(scores, dim=-1)
    context = torch.bmm(attn.unsqueeze(1), values).squeeze(1)
    return context, attn


def additive_attention(query, keys, values, linear_q, linear_k, v, mask=None):
    q_proj = linear_q(query).unsqueeze(1)
    k_proj = linear_k(keys)
    scores = v(torch.tanh(q_proj + k_proj)).squeeze(-1)
    if mask is not None:
        scores_length = scores.shape[1]
        mask = mask[:, :scores_length]
        scores = scores.masked_fill(~mask, -1e9)
    attn = torch.softmax(scores, dim=-1)
    context = torch.bmm(attn.unsqueeze(1), values).squeeze(1)
    return context, attn


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_size,
        hidden_size,
        attention="dot",
        num_layers=2,
        dropout=0.25,
        embedding_weight: torch.Tensor = None,
        padding_idx: int = 0,
    ):
        super().__init__()

        if embedding_weight is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embedding_weight, freeze=False, padding_idx=padding_idx
            )
        else:
            self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=padding_idx)
        

        self.rnn = nn.LSTM(emb_size + hidden_size, hidden_size, num_layers=num_layers, batch_first=True, bias=True, bidirectional=False, dropout=dropout)


        self.attention_type = attention
        if attention == "multiplicative":
            self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        elif attention == "additive":
            self.linear_q = nn.Linear(hidden_size, hidden_size)
            self.linear_k = nn.Linear(hidden_size, hidden_size)
            self.v = nn.Linear(hidden_size, 1, bias=False)
        self.generator = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
        self.vocab_projection = nn.Linear(hidden_size, vocab_size, bias=False)
        self.hidden_size = hidden_size

    def forward(self, tgt_input, hidden, encoder_outputs, src_mask):
        embedded = self.embedding(tgt_input)
        outputs = []
        attn_scores = []
        o_prev = torch.zeros(embedded.size(0), self.hidden_size, device=embedded.device)
        for t in range(embedded.size(1)):
            rnn_input = torch.cat((embedded[:,t,:], o_prev), dim=-1).unsqueeze(1)
            rnn_output, hidden = self.rnn(rnn_input, hidden)
            dec_rnn_output = rnn_output.squeeze(1)
            if self.attention_type == "dot":
                context, attn = dot_attention(dec_rnn_output, encoder_outputs, encoder_outputs, src_mask) 
            elif self.attention_type == "multiplicative":
                context, attn = general_attention(dec_rnn_output, encoder_outputs, encoder_outputs, self.linear, src_mask)
            else:
                context, attn = additive_attention(dec_rnn_output, encoder_outputs, encoder_outputs, self.linear_q, self.linear_k, self.v, src_mask)

            o_prev = self.generator(torch.cat([dec_rnn_output, context], dim=-1))
            outputs.append(o_prev)
            attn_scores.append(attn)
        outputs = self.vocab_projection(torch.stack(outputs, dim=1))
        attn_scores = torch.stack(attn_scores, dim=1)
        return outputs, hidden, attn_scores


class Seq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab,
        tgt_vocab,
        emb_size=128,
        hidden_size=256,
        num_layers=2,
        attention="dot",
        embedding_weight: torch.Tensor = None,
        padding_idx: int = 0,
        dropout=0.25,
    ):
        super().__init__()
        self.padding_idx = padding_idx
        # If pretrained embedding is provided, override emb_size
        effective_emb = embedding_weight.size(1) if embedding_weight is not None else emb_size
        self.encoder = Encoder(
            src_vocab,
            effective_emb,
            hidden_size,
            num_layers=num_layers,
            embedding_weight=embedding_weight,
            padding_idx=padding_idx,
            dropout=dropout,
        )
        self.decoder = Decoder(
            tgt_vocab,
            effective_emb,
            hidden_size,
            num_layers=num_layers,
            attention=attention,
            embedding_weight=embedding_weight,
            padding_idx=padding_idx,
            dropout=dropout,
        )

    def forward(self, batch):
        src_mask = batch.src != self.padding_idx
        enc_outputs, hidden = self.encoder(batch.src)
        logits, _, attn = self.decoder(batch.tgt_input, hidden, enc_outputs, src_mask)
        return logits, attn

    def forward_free_running(self, batch, max_len=None):
        """
        Free running decoding during training: feed previous prediction instead of ground truth.
        This is non-differentiable due to argmax but useful to simulate inference-time behavior.
        """
        src_mask = batch.src != self.padding_idx
        enc_out, hidden = self.encoder(batch.src)
        # initialize with <s>
        ys = batch.tgt_input[:, :1]
        outputs = []
        max_steps = max_len or batch.tgt_output.size(1)
        for _ in range(max_steps):
            logits, hidden, _ = self.decoder(ys[:, -1:], hidden, enc_out, src_mask)
            step_logits = logits[:, -1]
            outputs.append(step_logits)
            next_token = step_logits.argmax(-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
        # stack outputs back to time-major logits
        stacked = torch.stack(outputs, dim=1)
        return stacked

    def greedy_decode(self, src, max_len, sos_idx, eos_idx):
        self.eval()
        with torch.no_grad():
            src_mask = src != self.padding_idx
            enc_out, hidden = self.encoder(src)
            ys = torch.tensor([[sos_idx]], device=src.device)
            outputs = []
            for _ in range(max_len):
                logits, hidden, _ = self.decoder(ys[:, -1:], hidden, enc_out, src_mask)
                next_token = logits[:, -1].argmax(-1)
                outputs.append(next_token.item())
                ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
                if next_token.item() == eos_idx:
                    break
        return outputs

    def beam_search(self, src, beam_size, max_len, sos_idx, eos_idx):
        self.eval()
        with torch.no_grad():
            src_mask = src != self.padding_idx
            enc_out, hidden = self.encoder(src)
            beams = [(hidden, [sos_idx], 0.0)]
            for _ in range(max_len):
                new_beams = []
                for h, seq, score in beams:
                    last = torch.tensor([[seq[-1]]], device=src.device)
                    logits, new_h, _ = self.decoder(last, h, enc_out, src_mask)
                    log_probs = torch.log_softmax(logits[:, -1], dim=-1).squeeze(0)
                    topk = torch.topk(log_probs, beam_size)
                    for prob, idx in zip(topk.values, topk.indices):
                        new_seq = seq + [idx.item()]
                        new_score = score + prob.item()
                        new_beams.append((new_h, new_seq, new_score))
                beams = sorted(new_beams, key=lambda x: x[2], reverse=True)[:beam_size]
                if any(seq[-1] == eos_idx for _, seq, _ in beams):
                    break
            return beams[0][1]
