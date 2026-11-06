"""Prepare the fixed GSM8K development splits from the local source files."""

import hashlib
import json
import re
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path

import yaml
import numpy as np

from src.rng import rng_for


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
REPORTS = ROOT / "reports"
PROMPTS = ROOT / "prompts"
THRESHOLD = 0.5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize(question: str) -> str:
    return " ".join(question.lower().split())


def grams(question: str) -> set[tuple[str, ...]]:
    words = normalize(question).split()
    return {tuple(words[i : i + 5]) for i in range(len(words) - 4)}


def similarity(a: set[tuple[str, ...]], b: set[tuple[str, ...]]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def read_raw(split: str) -> list[dict[str, str]]:
    path = RAW / f"gsm8k_{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").split("\n") if line]
    assert len(rows) == {"train": 7473, "test": 1319}[split]
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_processed(split: str, rows: list[dict[str, str]]) -> None:
    output = []
    for index, row in enumerate(rows):
        gold_raw = row["answer"].rsplit("####", 1)[1].strip()
        gold = Decimal(gold_raw.replace(",", ""))
        output.append(
            {
                "id": f"{split}-{index:05d}" if split == "train" else f"test-{index:04d}",
                "question": row["question"],
                "answer": row["answer"],
                "gold_raw": gold_raw,
                "gold_number": str(gold),
            }
        )
    write_jsonl(PROCESSED / f"{split}.jsonl", output)


def nearest_test(
    query: str,
    query_grams: set[tuple[str, ...]],
    test_normalized: dict[str, list[int]],
    test_grams: list[set[tuple[str, ...]]],
    inverted: dict[tuple[str, ...], list[int]],
) -> tuple[float, int | None]:
    exact = test_normalized.get(normalize(query), [])
    if exact:
        return 1.0, exact[0]
    overlaps: Counter[int] = Counter()
    for gram in query_grams:
        overlaps.update(inverted.get(gram, []))
    best_score, best_index = 0.0, None
    for index, intersection in overlaps.items():
        union = len(query_grams) + len(test_grams[index]) - intersection
        score = intersection / union
        if score > best_score:
            best_score, best_index = score, index
    return best_score, best_index


def main() -> None:
    source = PROMPTS / "source_gsm8k_cot.yaml"
    samples = yaml.safe_load(source.read_text(encoding="utf-8"))["fewshot_config"]["samples"]
    assert len(samples) == 8
    fewshot = []
    for sample in samples:
        assert re.search(r"The answer is [-+]?\d[\d,]*(?:\.\d+)?\.$", sample["target"])
        fewshot.append({"question": sample["question"], "answer": sample["target"]})
    PROMPTS.mkdir(exist_ok=True)
    fewshot_path = PROMPTS / "gsm8k_8shot.json"
    fewshot_path.write_text(json.dumps(fewshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    train = read_raw("train")
    test = read_raw("test")
    make_processed("train", train)
    make_processed("test", test)

    fewshot_normalized = {normalize(row["question"]) for row in fewshot}
    fewshot_grams = [grams(row["question"]) for row in fewshot]
    test_grams = [grams(row["question"]) for row in test]
    test_normalized: dict[str, list[int]] = defaultdict(list)
    inverted: dict[tuple[str, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(test):
        test_normalized[normalize(row["question"])].append(index)
        for gram in test_grams[index]:
            inverted[gram].append(index)

    fewshot_excluded = []
    test_excluded = []
    candidates = []
    candidate_scores = {}
    max_candidate_score = (0.0, None, None)
    for index, row in enumerate(train):
        question = row["question"]
        question_grams = grams(question)
        overlaps_fewshot = normalize(question) in fewshot_normalized or any(
            similarity(question_grams, example) >= THRESHOLD for example in fewshot_grams
        )
        test_score, test_index = nearest_test(
            question, question_grams, test_normalized, test_grams, inverted
        )
        if overlaps_fewshot:
            fewshot_excluded.append(index)
        if test_score >= THRESHOLD:
            test_excluded.append((index, test_index, test_score))
        if not overlaps_fewshot and test_score < THRESHOLD:
            candidates.append(index)
            candidate_scores[index] = (test_score, test_index)
            if test_score > max_candidate_score[0]:
                max_candidate_score = (test_score, index, test_index)

    test_fewshot_hits = []
    max_test_fewshot = (0.0, None, None)
    for test_index, row in enumerate(test):
        question_grams = test_grams[test_index]
        for example_index, example in enumerate(fewshot):
            score = 1.0 if normalize(row["question"]) == normalize(example["question"]) else similarity(
                question_grams, fewshot_grams[example_index]
            )
            if score > max_test_fewshot[0]:
                max_test_fewshot = (score, test_index, example_index)
            if score >= THRESHOLD:
                test_fewshot_hits.append((test_index, example_index, score))

    selected = rng_for(1, 0).permutation(candidates)[:300]
    assert len(selected) == 300
    shuffled = rng_for(1, 1).permutation(selected)
    groups = {
        "dev_base": shuffled[:50],
        "dev_select": shuffled[50:250],
        "dev_anal": shuffled[250:],
        "test": np.arange(len(test)),
    }
    SPLITS.mkdir(parents=True, exist_ok=True)
    for name, indices in groups.items():
        prefix, width = ("test", 4) if name == "test" else ("train", 5)
        text = "".join(f"{prefix}-{int(index):0{width}d}\n" for index in indices)
        (SPLITS / f"{name}.ids").write_text(text, encoding="ascii")

    chosen = {int(index) for indices in list(groups.values())[:3] for index in indices}
    assert len(chosen) == 300
    assert not chosen.intersection(fewshot_excluded)
    assert not chosen.intersection(index for index, _, _ in test_excluded)
    assert max_candidate_score[0] < THRESHOLD
    selected_score, selected_train_index = max(
        (candidate_scores[index][0], index) for index in chosen
    )
    selected_test_index = candidate_scores[selected_train_index][1]

    REPORTS.mkdir(parents=True, exist_ok=True)
    leakage = [
        "# GSM8K 누출 점검",
        "",
        "질문 정규화: 소문자화, 공백 압축, 양끝 공백 제거. 단어는 공백으로 나누며 문장부호는 유지한다. 연속 5단어 집합의 Jaccard 유사도 0.5 이상을 근접 중복으로 본다.",
        "",
        f"- train 중 8-shot 예시와 겹친 문제: {len(fewshot_excluded)}개 — " + (", ".join(f"train-{i:05d}" for i in fewshot_excluded) or "없음"),
        f"- train 중 test와 근접 중복인 문제: {len(test_excluded)}개 — " + (", ".join(f"train-{i:05d}/test-{j:04d} ({score:.3f})" for i, j, score in test_excluded) or "없음"),
        f"- 제외 목록의 합집합: {len(set(fewshot_excluded) | {i for i, _, _ in test_excluded})}개",
        f"- 개발 후보 수: {len(candidates)}개",
        f"- test와 8-shot 예시의 최대 Jaccard: {max_test_fewshot[0]:.6f} (test-{max_test_fewshot[1]:04d}, 예시 {max_test_fewshot[2] + 1})" if max_test_fewshot[1] is not None else "- test와 8-shot 예시의 최대 Jaccard: 0",
        "- test와 8-shot 예시의 0.5 이상 목록: " + (", ".join(f"test-{i:04d}/예시 {j+1} ({score:.3f})" for i, j, score in test_fewshot_hits) or "없음"),
        f"- 개발 후보와 test의 최대 Jaccard: {max_candidate_score[0]:.6f} (train-{max_candidate_score[1]:05d}, test-{max_candidate_score[2]:04d})" if max_candidate_score[1] is not None else "- 개발 후보와 test의 최대 Jaccard: 0",
        f"- 개발 300문제와 test의 최대 Jaccard: {selected_score:.6f} (train-{selected_train_index:05d}, test-{selected_test_index:04d})",
        "",
        "시험 문제의 모델 출력이나 성능은 이 점검에서 생성하거나 보지 않았다.",
    ]
    (REPORTS / "leakage.md").write_text("\n".join(leakage) + "\n", encoding="utf-8")

    split_hashes = [f"{sha256(SPLITS / f'{name}.ids')}  {name}.ids" for name in groups]
    (SPLITS / "SHA256SUMS").write_text("\n".join(split_hashes) + "\n", encoding="ascii")
    data_card = [
        "# GSM8K 데이터 카드",
        "",
        "| 분할 | 문제 수 | 목록 SHA-256 |",
        "| --- | ---: | --- |",
    ]
    for name, indices in groups.items():
        data_card.append(f"| {name} | {len(indices)} | `{sha256(SPLITS / f'{name}.ids')}` |")
    data_card.extend(["", "| 분할 | 정답 자릿수 분포 | 소수 정답 |", "| --- | --- | ---: |"])
    for name, indices in groups.items():
        source_rows = test if name == "test" else train
        digits = Counter()
        noninteger = 0
        for index in indices:
            gold_raw = source_rows[int(index)]["answer"].rsplit("####", 1)[1].strip()
            gold = Decimal(gold_raw.replace(",", ""))
            noninteger += gold != gold.to_integral_value()
            digits[len(str(abs(int(gold))))] += 1
        distribution = ", ".join(f"{size}자리 {count}" for size, count in sorted(digits.items()))
        data_card.append(f"| {name} | {distribution} | {noninteger} |")
    data_card.extend(
        [
            "",
            f"원본 train SHA-256: `{sha256(RAW / 'gsm8k_train.jsonl')}`",
            f"원본 test SHA-256: `{sha256(RAW / 'gsm8k_test.jsonl')}`",
            f"8-shot 예시 SHA-256: `{sha256(fewshot_path)}`",
            "",
            "질문 토큰 길이는 모델 토크나이저를 확보한 뒤 기록한다. Greedy 정확도, 무효율, 레이블 비율은 GPU 기준선 실행 뒤 기록한다.",
        ]
    )
    (REPORTS / "data_card.md").write_text("\n".join(data_card) + "\n", encoding="utf-8")
    print(f"few-shot: {len(fewshot)}, SHA-256 {sha256(fewshot_path)}")
    print(f"excluded: few-shot {len(fewshot_excluded)}, test near duplicate {len(test_excluded)}")
    print(f"candidate count: {len(candidates)}; dev split sizes: 50/200/50")
    print(f"test/few-shot maximum Jaccard: {max_test_fewshot[0]:.6f}")
    print(f"dev candidate/test maximum Jaccard: {max_candidate_score[0]:.6f}")
    print(f"selected dev/test maximum Jaccard: {selected_score:.6f}")


if __name__ == "__main__":
    main()
