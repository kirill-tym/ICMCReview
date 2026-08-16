import json
import random
import os
from verifier import check_solution


def generate_solvable_puzzle(num_range):
    while True:
        nums = [random.randint(num_range[0], num_range[1]) for _ in range(4)]
        a, b, c, d = nums
        candidates = [
            f"({a} + {b}) * ({c} - {d})",
            f"({a} * {b}) - ({c} + {d})",
            f"{a} * {b} * {c} / {d}",
            f"({a} + {b} + {c}) * {d}"
        ]
        for i in range(4):
            cand = candidates[i]
            if check_solution(nums, f"<answer>{cand}</answer>") == 1.0:
                return nums, cand, i


def create_dataset(filename, count, num_range):
    data = []
    for _ in range(count):
        nums, solution, tip = generate_solvable_puzzle(num_range)
        a, b, c, d = nums
        prompt = f"Make 24 using numbers: {nums}. Output reasoning in <think>...</think> and final math expression in <answer>...</answer>."
        if tip == 0:
            completion = f"<think>We have numbers {nums}. Let's try combining them to get 24. First, {a} + {b} = {a + b}. Then {c} - {d} = {c - d}. Finally, {a + b} * {c - d} = 24. This matches the target 24!</think>\n<answer>({a} + {b}) * ({c} - {d})</answer>"
        elif tip == 1:
            completion = f"<think>We have numbers {nums}. Let's try combining them to get 24. First, {a} * {b} = {a * b}. Then {c} + {d} = {c + d}. Finally, {a * b} - {c + d} = 24. This matches the target 24!</think>\n<answer>({a} * {b}) - ({c} + {d})</answer>"
        elif tip == 2:
            completion = f"<think>We have numbers {nums}. Let's try combining them to get 24. First, {a} * {b} = {a * b}. Then {a * b} * {c} = {a * b * c}. Finally, {a * b * c} / {d} = 24. This matches the target 24!</think>\n<answer>{a} * {b} * {c} / {d}</answer>"
        else:
            completion = f"<think>We have numbers {nums}. Let's try combining them to get 24. First, {a} + {b} = {a + b}. Then {a + b} + {c} = {a + b + c}. Finally, {a + b + c} * {d} = 24. This matches the target 24!</think>\n<answer>({a} + {b} + {c}) * {d}</answer>"
        data.append({"prompt": prompt, "completion": completion, "numbers": nums})

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved {count} samples to {filename}")


if __name__ == "__main__":
    random.seed(42)
    create_dataset("../data/sft_train.jsonl", 600, (1, 9))  # SFT Train (ID: числа 1-9)
    create_dataset("../data/test_id.jsonl", 100, (1, 9))  # Test ID (числа 1-9)
    create_dataset("../data/test_ood.jsonl", 100, (10, 25))  # Test OOD (числа 10-25)
