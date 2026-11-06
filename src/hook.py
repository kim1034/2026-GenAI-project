"""Inject one fixed noise vector per response at every selected MLP activation."""

from contextlib import contextmanager


class MLPNoise:
    def __init__(self, model, layer_ids: list[int], layer_scale: dict[int, float]):
        if set(layer_ids) != set(layer_scale):
            raise ValueError("each selected layer requires one scale")
        self.layer_scale = layer_scale
        self.base = None
        self.calls = 0
        self.handles = [
            model.model.layers[layer_id].mlp.act_fn.register_forward_hook(self._hook(layer_id))
            for layer_id in layer_ids
        ]

    def _hook(self, layer_id: int):
        def apply(_module, _inputs, output):
            if self.base is None:
                return output
            if output.ndim != 3 or self.base.ndim != 2 or output.shape[0] != self.base.shape[0] or output.shape[2] != self.base.shape[1]:
                raise ValueError("noise shape does not match MLP activation")
            self.calls += 1
            return output + self.layer_scale[layer_id] * self.base[:, None, :]
        return apply

    @contextmanager
    def with_base(self, base):
        if self.base is not None:
            raise RuntimeError("noise hook is already active")
        self.base = base
        try:
            yield
        finally:
            self.base = None

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
