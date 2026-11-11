"""Teacher-forced sensitivity to the shared rho-scaled MLP perturbation."""

import re
import numpy as np

from src.diag import effective_dimension
from src.hook import MLPNoise


def summarize_gradient(values):
    raw = np.asarray(values, dtype=np.float64)
    centered = raw - raw.mean()
    energy = np.square(centered)
    return {"norm": float(np.linalg.norm(centered)),
            "deff": effective_dimension(raw),
            "top10": float(np.sort(energy)[-10:].sum() / energy.sum()),
            "mean": float(raw.mean())}


class GradientDiagnostic:
    def __init__(self, model, tokenizer, config, rho):
        self.model = model
        self.tokenizer = tokenizer
        self.d = config["noise"]["d"]
        self.model.requires_grad_(False)
        layers = config["noise"]["layers_0idx"]
        self.hook = MLPNoise(model, layers, {l: float(rho[str(l + 1)]) for l in layers})

    def close(self):
        self.hook.remove()

    def hidden(self, prompt, tokens, e):
        import torch
        ids = torch.tensor([prompt + tokens[:-1]], dtype=torch.long, device="cuda")
        with self.hook.with_base(e[None, :]):
            hidden = self.model.model(input_ids=ids, use_cache=False).last_hidden_state
        return hidden[0, len(prompt) - 1:, :]

    def values(self, prompt, tokens, e, fixed=None):
        """Chunk vocabulary projections; never allocate logits for prompt positions."""
        import torch
        hidden = self.hidden(prompt, tokens, e)
        target = torch.tensor(tokens, dtype=torch.long, device="cuda")
        margins, top_ids = [], []
        seq = torch.zeros((), device="cuda")
        for start in range(0, len(tokens), 32):
            logits = self.model.lm_head(hidden[start:start + 32])
            seq = seq + torch.log_softmax(logits, dim=-1).gather(1, target[start:start + 32, None]).sum()
            with torch.no_grad():
                top = logits.topk(2, dim=-1)
                margins.extend((top.values[:, 0] - top.values[:, 1]).tolist())
                top_ids.extend(top.indices.tolist())
        if fixed is None:
            positions = [i for i, token in enumerate(tokens)
                         if re.search(r"[0-9]", self.tokenizer.decode([token], skip_special_tokens=False))]
            eligible = positions or list(range(len(tokens)))
            t_star = min(eligible, key=lambda i: (margins[i], i))
            fixed = {"t_star": t_star, "top_ids": top_ids[t_star],
                     "margin_min": min(margins), "margin_min_digit": min(margins[i] for i in positions) if positions else None,
                     "digit_fallback": not bool(positions)}
        selected = self.model.lm_head(hidden[fixed["t_star"]:fixed["t_star"] + 1])[0]
        first, second = fixed["top_ids"]
        margin = selected[first] - selected[second]
        return margin, seq, fixed

    def compute(self, prompt, tokens, *, fd_direction=None):
        import torch
        if not tokens:
            raise ValueError("diagnostics require original generated token IDs, including EOS")
        e = torch.zeros(self.d, device="cuda", requires_grad=True)
        margin, seq, fixed = self.values(prompt, tokens, e)
        g_m = torch.autograd.grad(margin, e, retain_graph=True)[0].detach().cpu().numpy()
        g_seq = torch.autograd.grad(seq, e)[0].detach().cpu().numpy()
        info = {**fixed, "margin": float(margin.detach()), "mean_logprob": float(seq.detach()) / len(tokens)}
        del margin, seq, e
        for kind, gradient in (("g_m", g_m), ("g_seq", g_seq)):
            info.update({f"{key}_{kind}": value for key, value in summarize_gradient(gradient).items()})
        if fd_direction is not None:
            direction = np.asarray(fd_direction, dtype=np.float32)
            h = 1e-3 * np.sqrt(self.d)
            u = torch.tensor(direction, device="cuda")
            with torch.no_grad():
                plus = self.values(prompt, tokens, h * u, fixed)[:2]
                minus = self.values(prompt, tokens, -h * u, fixed)[:2]
            info["finite_difference"] = {}
            for index, (kind, gradient) in enumerate((("g_m", g_m), ("g_seq", g_seq))):
                fd = (float(plus[index]) - float(minus[index])) / (2 * h)
                dot = float(np.dot(gradient.astype(np.float64), direction))
                error = abs(fd - dot) / max(abs(fd), abs(dot), 1e-12)
                info["finite_difference"][kind] = {"central_difference": fd, "autograd": dot,
                                                    "relative_error": error, "passed": error < 0.05}
        return info, g_m, g_seq
