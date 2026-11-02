"""Autoregressive decoding: greedy and beam search.

Both decoders re-run the full decoder stack over the growing target
sequence at every step (no KV cache): O(t^2) compute per sequence, but
simple to verify and fast enough at max_len=128.

Both accept a raw model or a DDP wrapper (unwrapped via .module).
"""

import torch


@torch.no_grad()
def greedy_decode_batch(
    model,
    src_tokens: torch.Tensor,
    max_len: int,
    bos_idx: int,
    eos_idx: int,
    pad_idx: int,
    device: torch.device = None,
) -> torch.Tensor:
    """Greedy decoding. src_tokens: (B, src_len) -> decoded: (B, <=max_len).

    Finished sequences are frozen by emitting PAD after their EOS.
    """
    m = model.module if hasattr(model, "module") else model
    m.eval()
    device = src_tokens.device if device is None else device
    B = src_tokens.size(0)

    # ---- encode once ----
    src_mask = m.make_src_mask(src_tokens, pad_idx)
    memory = m.encoder(src_tokens, src_mask)

    ys = torch.full((B, 1), bos_idx, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        # no look-ahead mask needed: ys only ever contains past tokens
        out = m.decoder(ys, memory, tgt_mask=None, memory_mask=src_mask)
        logits = m.out(out[:, -1])               # (B, vocab)
        next_tokens = logits.argmax(dim=-1)      # (B,)
        next_tokens = next_tokens.masked_fill(finished, pad_idx)

        ys = torch.cat([ys, next_tokens.unsqueeze(1)], dim=1)
        finished |= next_tokens == eos_idx

        if finished.all():
            break

    return ys


@torch.no_grad()
def beam_decode_batch(
    model,
    src_tokens: torch.Tensor,
    max_len: int,
    bos_idx: int,
    eos_idx: int,
    pad_idx: int,
    beam_size: int = 5,
    length_penalty: float = 1.0,
    device: torch.device = None,
) -> torch.Tensor:
    """fairseq/HF-aligned beam search. src_tokens: (B, src_len) -> (B, <=max_len).

    Ranking uses the fairseq length penalty ((len + 5) / 6)^alpha form,
    simplified to (len + 5)^alpha. Finished beams are parked on EOS so
    they stay valid candidates without expanding.
    """
    m = model.module if hasattr(model, "module") else model
    m.eval()
    device = src_tokens.device if device is None else device
    B = src_tokens.size(0)

    # ---- encode once, replicate per beam ----
    src_mask = m.make_src_mask(src_tokens, pad_idx)
    memory = m.encoder(src_tokens, src_mask)
    memory = memory.repeat_interleave(beam_size, dim=0)
    src_mask = src_mask.repeat_interleave(beam_size, dim=0)

    ys = torch.full((B * beam_size, 1), bos_idx, dtype=torch.long, device=device)

    # all beams but the first start at -inf so step 1 doesn't duplicate
    beam_scores = torch.zeros(B, beam_size, device=device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)  # (B*beam,)

    finished = torch.zeros(B * beam_size, dtype=torch.bool, device=device)
    vocab_size = m.out.weight.size(0)

    for _ in range(1, max_len):
        logits = m.out(
            m.decoder(ys, memory, tgt_mask=None, memory_mask=src_mask)[:, -1]
        )
        log_probs = torch.log_softmax(logits, dim=-1)

        # finished beams cannot expand: park them on EOS at cost 0
        log_probs[finished] = -1e9
        log_probs[finished, eos_idx] = 0.0

        scores = log_probs + beam_scores.unsqueeze(1)

        # candidates carry the RAW cumulative score; dividing by a constant
        # (every beam has the same length here) does not change the topk,
        # so normalisation matters only at final selection below
        length = ys.size(1)
        norm_scores = scores / ((length + 5) ** length_penalty)

        norm_scores = norm_scores.view(B, beam_size * vocab_size)
        scores = scores.view(B, beam_size * vocab_size)

        topk_norm, topk_ids = norm_scores.topk(beam_size, dim=1)
        topk_scores = scores.gather(1, topk_ids)

        beam_ids = topk_ids // vocab_size
        token_ids = topk_ids % vocab_size

        # gather source beams per batch element
        offset = torch.arange(B, device=device).unsqueeze(1) * beam_size
        flat_beam_ids = (beam_ids + offset).view(-1)

        ys = torch.cat([ys[flat_beam_ids], token_ids.view(-1, 1)], dim=1)
        beam_scores = topk_scores.view(-1)
        finished = finished[flat_beam_ids] | (token_ids.view(-1) == eos_idx)

        if finished.view(B, beam_size).all(dim=1).all():
            break

    # ---- final selection: best length-normalised beam per example ----
    ys = ys.view(B, beam_size, -1)
    beam_scores = beam_scores.view(B, beam_size)

    lengths = (ys != pad_idx).sum(dim=-1)
    norm_scores = beam_scores / ((lengths + 5) ** length_penalty)
    best_ids = norm_scores.argmax(dim=1)

    results = [ys[b, best_ids[b]] for b in range(B)]
    return torch.nn.utils.rnn.pad_sequence(
        results, batch_first=True, padding_value=pad_idx
    )
