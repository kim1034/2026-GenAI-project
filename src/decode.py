"""Self-contained batched decoding with fixed rows and last-position logits."""

from contextlib import nullcontext

import numpy as np

from src.sampling import sample_from_logits


def decode(
    model,
    prompt_ids,
    *,
    n_rows: int,
    temperature: float,
    uniforms: np.ndarray,
    max_new_tokens: int,
    eos_token_ids: list[int],
    noise_hook=None,
    noise_base=None,
) -> tuple[np.ndarray, list[int], list[str]]:
    import torch

    if prompt_ids.ndim != 2 or prompt_ids.shape[0] != 1:
        raise ValueError("prompt_ids must have shape [1, prompt length]")
    if uniforms.shape != (n_rows, max_new_tokens) or n_rows < 1:
        raise ValueError("one fixed uniform sequence is required per response row")
    if noise_hook is None and noise_base is not None:
        raise ValueError("a noise base requires the injection hook")
    if not eos_token_ids:
        raise ValueError("at least one EOS token ID is required")

    context = noise_hook.with_base(noise_base) if noise_hook is not None and noise_base is not None else nullcontext()
    device = prompt_ids.device
    finished = np.zeros(n_rows, dtype=bool)
    lengths = [max_new_tokens] * n_rows
    finish_reasons = ["length"] * n_rows
    generated = []
    with torch.inference_mode(), context:
        ids = prompt_ids.repeat(n_rows, 1)
        output = model.model(input_ids=ids, use_cache=True, return_dict=True)
        past = output.past_key_values
        logits = model.lm_head(output.last_hidden_state[:, -1:, :])[:, -1, :]
        for step in range(max_new_tokens):
            next_ids = sample_from_logits(logits.double().cpu().numpy(), uniforms[:, step], temperature)
            next_ids[finished] = eos_token_ids[0]
            just_finished = (~finished) & np.isin(next_ids, eos_token_ids)
            for row in np.flatnonzero(just_finished):
                lengths[int(row)] = step + 1
                finish_reasons[int(row)] = "eos"
            finished |= just_finished
            generated.append(next_ids.copy())
            if finished.all():
                break
            output = model.model(
                input_ids=torch.as_tensor(next_ids[:, None], dtype=torch.long, device=device),
                past_key_values=past, use_cache=True, return_dict=True,
            )
            past = output.past_key_values
            logits = model.lm_head(output.last_hidden_state[:, -1:, :])[:, -1, :]
    return np.stack(generated, axis=1), lengths, finish_reasons
