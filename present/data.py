import json
import itertools
from fractions import Fraction
import random

SYSTEM_PROMPT = """You are an expert arithmetic solver. Your task is to reach the number 24 using the exact 4 values assigned to the given cards. You can use +, -, *, /, and parentheses. Each value must be used exactly once.

Respond in the following format:
<think>
Step by step reasoning about arithmetic combinations...
</think>
<answer>
(expression that equals 24)
</answer>"""


def solve_24(numbers):
    """Ищет хотя бы одно решение для 4 чисел и возвращает кортеж (шаги, формула)."""

    def helper(state):
        if len(state) == 1:
            val, expr, steps = state[0]
            if val == 24:
                return steps, expr
            return None

        for i in range(len(state)):
            for j in range(len(state)):
                if i == j: continue
                val1, expr1, steps1 = state[i]
                val2, expr2, steps2 = state[j]

                next_state_base = [state[k] for k in range(len(state)) if k != i and k != j]

                ops = [
                    ('+', val1 + val2, f"{val1} + {val2} = {val1 + val2}"),
                    ('-', val1 - val2, f"{val1} - {val2} = {val1 - val2}"),
                    ('*', val1 * val2, f"{val1} * {val2} = {val1 * val2}")
                ]
                if val2 != 0:
                    ops.append(('/', val1 / val2, f"{val1} / {val2} = {val1 / val2}"))

                for op, new_val, step_str in ops:
                    new_expr = f"({expr1} {op} {expr2})"
                    new_state = next_state_base + [(new_val, new_expr, steps1 + steps2 + [step_str])]
                    result = helper(new_state)
                    if result:
                        return result
        return None

    initial_state = [(Fraction(n), str(n), []) for n in numbers]
    return helper(initial_state)


def multiply_dataset(base_data, target_count=10000):
    """Аугментирует датасет за счет перестановок карт в промпте до нужного количества."""
    augmented_data = []
    seen_prompts = set()

    # Сначала добавляем оригинальные примеры
    for item in base_data:
        cards_str = ", ".join(item["cards"])
        seen_prompts.add(cards_str)
        augmented_data.append(item)

    attempts = 0
    max_attempts = target_count * 10  # Защита от бесконечного цикла

    # Добиваем до 10 000 уникальными перестановками
    while len(augmented_data) < target_count and attempts < max_attempts:
        for item in base_data:
            attempts += 1
            if len(augmented_data) >= target_count:
                break

            cards = item["cards"].copy()
            random.shuffle(cards)
            cards_str = ", ".join(cards)

            if cards_str not in seen_prompts:
                seen_prompts.add(cards_str)

                new_item = {
                    "messages": [
                        item["messages"][0],  # System Prompt
                        {"role": "user",
                         "content": f"Make 24 using these cards: {cards_str}. Rules: J = 10, Q = 10, K = 10."},
                        item["messages"][2]  # Assistant Answer
                    ],
                    "target_numbers": item["target_numbers"],
                    "cards": cards
                }
                augmented_data.append(new_item)

    return augmented_data


def generate_dataset():
    faces = {1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8',
             9: '9', 10: '10', 11: 'J', 12: 'Q', 13: 'K'}

    all_id_data = []
    all_ood_data = []

    # Фиксируем seed для стабильности
    random.seed(42)

    all_combos = list(itertools.combinations_with_replacement(range(1, 14), 4))

    for combo in all_combos:
        combo_list = list(combo)
        random.shuffle(combo_list)
        card_names = [faces[c] for c in combo_list]
        cards_str = ", ".join(card_names)

        # 1. IN-DISTRIBUTION (ID): J = 10, Q = 10, K = 10
        id_numbers = [10 if c > 10 else c for c in combo_list]
        id_solution = solve_24(id_numbers)

        if id_solution:
            steps, expr = id_solution
            think_text = "\n".join(steps) + f"\nResult is 24."
            answer_text = expr.replace('Fraction(', '').replace(', 1)', '')

            all_id_data.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",
                     "content": f"Make 24 using these cards: {cards_str}. Rules: J = 10, Q = 10, K = 10."},
                    {"role": "assistant",
                     "content": f"<think>\n{think_text}\n</think>\n<answer>\n{answer_text}\n</answer>"}
                ],
                "target_numbers": id_numbers,
                "cards": card_names
            })

        # 2. OUT-OF-DISTRIBUTION (OOD): J = 11, Q = 12, K = 13
        ood_numbers = combo_list.copy()
        ood_solution = solve_24(ood_numbers)

        if ood_solution and any(c > 10 for c in combo_list):
            all_ood_data.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",
                     "content": f"Make 24 using these cards: {cards_str}. Rules: J = 11, Q = 12, K = 13."}
                ],
                "target_numbers": ood_numbers,
                "cards": card_names
            })

    # Перемешиваем выборки
    random.shuffle(all_id_data)
    random.shuffle(all_ood_data)

    # Делим базовый ID на Train (80%) и Test (20%)
    split_idx = int(len(all_id_data) * 0.8)
    train_id = all_id_data[:split_idx]
    test_id = all_id_data[split_idx:]

    print("Аугментация датасета для SFT до 10,000 примеров...")
    train_id_sft = multiply_dataset(train_id, target_count=10000)

    # Сохранение сплитов
    with open("data/train_id.jsonl", "w") as f:
        for item in train_id:
            f.write(json.dumps(item) + "\n")

    with open("data/train_id_sft.jsonl", "w") as f:
        for item in train_id_sft:
            f.write(json.dumps(item) + "\n")

    with open("data/test_id.jsonl", "w") as f:
        for item in test_id:
            f.write(json.dumps(item) + "\n")

    with open("data/test_ood.jsonl", "w") as f:
        for item in all_ood_data:
            f.write(json.dumps(item) + "\n")

    print(f"\nУспешно сгенерировано:")
    print(f" - Train ID (базовый для RL): {len(train_id)} примеров (train_id.jsonl)")
    print(f" - Train ID SFT (расширенный): {len(train_id_sft)} примеров (train_id_sft.jsonl)")
    print(f" - Test ID:  {len(test_id)} примеров (test_id.jsonl)")
    print(f" - Test OOD: {len(all_ood_data)} примеров (test_ood.jsonl)")


if __name__ == "__main__":
    generate_dataset()