import itertools
import json
import os
import random


def solve_24(nums):
    """Алгоритмический поиск верного выражения для 24."""
    ops = ['+', '-', '*', '/']
    # Перебираем все перестановки чисел
    for p in set(itertools.permutations(nums)):
        a, b, c, d = p
        # Перебираем все комбинации операторов
        for op1, op2, op3 in itertools.product(ops, repeat=3):
            # 5 возможных структур скобок для 4 операндов
            exprs = [
                f"(({a} {op1} {b}) {op2} {c}) {op3} {d}",
                f"({a} {op1} ({b} {op2} {c})) {op3} {d}",
                f"({a} {op1} {b}) {op2} ({c} {op3} {d})",
                f"{a} {op1} (({b} {op2} {c}) {op3} {d})",
                f"{a} {op1} ({b} {op2} ({c} {op3} {d}))",
            ]
            for expr in exprs:
                try:
                    # Проверяем равенство 24 с учетом погрешности float
                    if abs(eval(expr) - 24.0) < 1e-5:
                        return expr
                except ZeroDivisionError:
                    continue
    return None


def generate_dataset():
    os.makedirs("../data", exist_ok=True)

    print("🧠 Поиск уникальных решаемых комбинаций 24-Game...")

    unique_problems = {}
    # Генерируем комбинации из 4 чисел от 1 до 10
    all_combinations = list(itertools.combinations_with_replacement(range(1, 11), 4))
    random.seed(42)
    random.shuffle(all_combinations)

    for combo in all_combinations:
        solution = solve_24(combo)
        if solution:
            # Ключ — отсортированный кортеж, чтобы исключить дубликаты
            unique_problems[combo] = solution

    all_samples = list(unique_problems.items())
    print(f"✅ Найдено всего уникальных решаемых задач: {len(all_samples)}")

    # Жесткое разделение Train / Holdout Test (80% / 20%)
    split_idx = int(len(all_samples) * 0.8)
    train_data = all_samples[:split_idx]
    test_data = all_samples[split_idx:]

    # Сохраняем Train (SFT / RL)
    train_file = "../data/sft_train.jsonl"
    with open(train_file, "w", encoding="utf-8") as f:
        for nums, expr in train_data:
            shuffled_nums = list(nums)
            random.shuffle(shuffled_nums)
            item = {
                "messages": [
                    {"role": "user", "content": f"Make 24 using numbers: {shuffled_nums}"},
                    {"role": "assistant", "content": f"<answer>{expr}</answer>"}
                ],
                "numbers": shuffled_nums
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Сохраняем Holdout Test (Только отложенные примеры!)
    test_file = "../data/test_holdout.jsonl"
    with open(test_file, "w", encoding="utf-8") as f:
        for nums, expr in test_data:
            shuffled_nums = list(nums)
            random.shuffle(shuffled_nums)
            item = {
                "prompt": f"Make 24 using numbers: {shuffled_nums}",
                "numbers": shuffled_nums,
                "reference_solution": expr
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"📊 Сформировано:")
    print(f"   • Train samples: {len(train_data)} -> {train_file}")
    print(f"   • Holdout Test samples (Unseen): {len(test_data)} -> {test_file}")


if __name__ == "__main__":
    generate_dataset()