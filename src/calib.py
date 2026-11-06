"""Measure unperturbed MLP activation RMS by layer during greedy decoding."""

from math import sqrt


class RMSRecorder:
    def __init__(self, model, layer_ids: list[int]):
        self.sums = {layer: 0.0 for layer in layer_ids}
        self.counts = {layer: 0 for layer in layer_ids}
        self.handles = [
            model.model.layers[layer].mlp.act_fn.register_forward_hook(self._hook(layer))
            for layer in layer_ids
        ]

    def _hook(self, layer: int):
        def record(_module, _inputs, output):
            self.sums[layer] += output.detach().square().sum().double()
            self.counts[layer] += output.numel()
            return output
        return record

    def reset(self):
        for layer in self.sums:
            self.sums[layer] = 0.0
            self.counts[layer] = 0

    def snapshot(self) -> dict:
        return {"sum_squares": {str(layer + 1): float(value) for layer, value in self.sums.items()},
                "counts": {str(layer + 1): value for layer, value in self.counts.items()}}

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def combine_rms(snapshots: list[dict]) -> dict[str, float]:
    if not snapshots:
        raise ValueError("RMS calibration requires at least one problem")
    layers = snapshots[0]["sum_squares"]
    result = {}
    for layer in layers:
        total_sum = sum(item["sum_squares"][layer] for item in snapshots)
        total_count = sum(item["counts"][layer] for item in snapshots)
        if total_count <= 0:
            raise ValueError(f"no activation values recorded for layer {layer}")
        result[layer] = sqrt(total_sum / total_count)
    return result
