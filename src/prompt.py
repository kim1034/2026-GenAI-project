"""Render the fixed GSM8K chat prompt and fingerprint its exact token sequence."""

import hashlib
import json
from pathlib import Path


def load_messages(system_path: Path, fewshot_path: Path, question: str) -> list[dict[str, str]]:
    system = system_path.read_text(encoding="utf-8").strip()
    examples = json.loads(fewshot_path.read_text(encoding="utf-8"))
    if len(examples) != 8:
        raise ValueError("the experiment requires exactly eight fixed examples")
    messages = [{"role": "system", "content": system}]
    for example in examples:
        messages.extend(
            [
                {"role": "user", "content": example["question"]},
                {"role": "assistant", "content": example["answer"]},
            ]
        )
    messages.append({"role": "user", "content": question})
    return messages


def render_prompt(tokenizer, messages: list[dict[str, str]]) -> tuple[str, list[int], str]:
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(rendered, add_special_tokens=False, truncation=False)["input_ids"]
    digest = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    return rendered, ids, digest
