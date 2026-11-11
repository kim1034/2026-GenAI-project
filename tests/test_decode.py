from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from src.decode import decode


def test_token_limit_does_not_run_an_unused_forward():
    class Toy:
        def __init__(self):
            self.calls = 0

        def model(self, input_ids, **kwargs):
            self.calls += 1
            return SimpleNamespace(last_hidden_state=torch.zeros((*input_ids.shape, 2)), past_key_values=None)

        def lm_head(self, hidden):
            logits = torch.zeros((*hidden.shape[:-1], 3))
            logits[..., 1] = 1
            return logits

    model = Toy()
    tokens, lengths, reasons = decode(model, torch.tensor([[2, 2]]), n_rows=2,
        temperature=0, uniforms=np.zeros((2, 3)), max_new_tokens=3, eos_token_ids=[0])
    assert tokens.tolist() == [[1, 1, 1], [1, 1, 1]]
    assert model.calls == 3
    assert lengths == [3, 3]
    assert reasons == ["length", "length"]

    model = Toy()
    tokens, lengths, reasons = decode(model, torch.tensor([[2, 2]]), n_rows=2,
        temperature=0, uniforms=np.zeros((2, 3)), max_new_tokens=3, eos_token_ids=[1])
    assert tokens.tolist() == [[1], [1]]
    assert model.calls == 1
    assert lengths == [1, 1]
    assert reasons == ["eos", "eos"]
