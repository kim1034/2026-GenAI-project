"""Load the pinned Qwen model on one CUDA device for deterministic evaluation."""


def load_model(config: dict):
    revision = config["model"]["revision"]
    if not revision:
        raise ValueError("model.revision must be pinned before loading the model")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("this experiment requires a CUDA GPU")
    if config["model"]["dtype"] != "float32" or config["model"]["attn_implementation"] != "eager":
        raise ValueError("v1.2 requires float32 and eager attention")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["id"], revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        config["model"]["id"], revision=revision,
        torch_dtype=torch.float32, attn_implementation="eager",
    ).to("cuda")
    model.eval()
    expected = config["model"]["expect"]
    for name in ("num_hidden_layers", "intermediate_size", "hidden_act"):
        if getattr(model.config, name) != expected[name]:
            raise ValueError(f"unexpected model {name}: {getattr(model.config, name)}")
    return model, tokenizer
