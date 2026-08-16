import json
import itertools
from fractions import Fraction
import random
import re

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
                if i == j:
                    continue
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
    """Аугментирует датасет за счет перестановок карт в промпте с сохранением правил."""
    augmented_data = []
    seen_prompts = set()

    for item in base_data:
        cards_str = ", ".join(item["cards"])
        seen_prompts.add(cards_str)
        augmented_data.append(item)

    attempts = 0
    max_attempts = target_count * 10

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

                # Извлекаем динамические правила из оригинального сообщения
                orig_user_msg = item["messages"][1]["content"]
                rules_match = re.search(r"(Rules: .*)", orig_user_msg)
                rules_str = rules_match.group(1) if rules_match else "Rules: J = 10, Q = 10, K = 10."

                new_item = {
                    "messages": [
                        item["messages"][0], # System Prompt
                        {"role": "user",
                         "content": f"Make 24 using these cards: {cards_str}. {rules_str}"},
                        item["messages"][2] # Assistant Answer
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
            steps, expr = ood_solution
            think_text = "\n".join(steps) + f"\nResult is 24."
            answer_text = expr.replace('Fraction(', '').replace(', 1)', '')

            all_ood_data.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",
                     "content": f"Make 24 using these cards: {cards_str}. Rules: J = 11, Q = 12, K = 13."},
                    {"role": "assistant",
                     "content": f"<think>\n{think_text}\n</think>\n<answer>\n{answer_text}\n</answer>"}
                ],
                "target_numbers": ood_numbers,
                "cards": card_names
            })

    # Перемешиваем выборки
    random.shuffle(all_id_data)
    random.shuffle(all_ood_data)

    # 1. Делим ID на Train (80%) и Test ID (20%)
    split_id_idx = int(len(all_id_data) * 0.8)
    train_id = all_id_data[:split_id_idx]
    test_id = all_id_data[split_id_idx:]

    print(f"Длина id даты: {len(all_id_data)}")
    print("-" * 65)

    # 2. Делим OOD: Оставляем ровно 250 примеров на Test OOD, остальное в Train OOD
    target_test_ood = 250
    split_ood_idx = max(0, len(all_ood_data) - target_test_ood)
    train_ood = all_ood_data[:split_ood_idx]
    test_ood = all_ood_data[split_ood_idx:]

    print(f"Длина ood даты: {len(all_ood_data)}\n")

    print("Аугментация датасетов до 10k каждый...")
    train_id_sft = multiply_dataset(train_id, target_count=10000)
    train_ood_sft = multiply_dataset(train_ood, target_count=10000)

    # Собираем мощный 50/50 датасет на 20 000 примеров
    train_mixed_sft = train_id_sft + train_ood_sft
    random.shuffle(train_mixed_sft)

    # Сохраняем файлики
    with open("train_id.jsonl", "w") as f:
        for item in train_id:
            f.write(json.dumps(item) + "\n")

    with open("train_id_sft.jsonl", "w") as f:
        for item in train_id_sft:
            f.write(json.dumps(item) + "\n")

    with open("train_mixed_sft.jsonl", "w") as f:
        for item in train_mixed_sft:
            f.write(json.dumps(item) + "\n")

    with open("test_id.jsonl", "w") as f:
        for item in test_id:
            f.write(json.dumps(item) + "\n")

    # В test_ood сохраняем промпты без ответа ассистента для чистой оценки
    # Сохраняем test_ood в полном формате (test_base.py сам отрежет ассистента для промпта)
    with open("test_ood.jsonl", "w") as f:
        for item in test_ood:
            f.write(json.dumps(item) + "\n")

    print(f"\nГотово! Результаты генерации:")
    print(f" - Train ID (для чистого RL): {len(train_id)} примеров (train_id.jsonl)")
    print(f" - Train Mixed SFT (10k ID + 10k OOD): {len(train_mixed_sft)} примеров (train_mixed_sft.jsonl) <-- ОБУЧАЕМ TEACHER НА НЕМ!")
    print(f" - Test ID: {len(test_id)} примеров (test_id.jsonl)")
    print(f" - Test OOD: {len(test_ood)} примеров (test_ood.jsonl)")


if __name__ == "__main__":
    generate_dataset()