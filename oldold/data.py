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

                # Формируем новый стейт без i и j элементов
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


def generate_dataset():
    faces = {1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8',
             9: '9', 10: '10', 11: 'J', 12: 'Q', 13: 'K'}

    id_data = []
    ood_data = []

    # Перебираем все возможные комбинации 4 карт с возвращением
    all_combos = list(itertools.combinations_with_replacement(range(1, 14), 4))

    for combo in all_combos:
        card_names = [faces[c] for c in combo]

        # In-Distribution (ID): J, Q, K = 10
        id_numbers = [10 if c > 10 else c for c in combo]
        id_solution = solve_24(id_numbers)

        # Out-of-Distribution (OOD): J=11, Q=12, K=13
        ood_numbers = list(combo)
        ood_solution = solve_24(ood_numbers)

        # Если есть решение в ID, добавляем в трейн
        if id_solution:
            steps, expr = id_solution
            think_text = "\n".join(steps) + f"\nResult is 24."
            answer_text = expr.replace('Fraction(', '').replace(', 1)', '')  # Очистка дробей для ответа

            id_data.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",
                     "content": f"Make 24 using these cards: {', '.join(card_names)}. Note: J, Q, K are worth 10."},
                    {"role": "assistant",
                     "content": f"<think>\n{think_text}\n</think>\n<answer>\n{answer_text}\n</answer>"}
                ],
                "target_numbers": id_numbers,
                "cards": card_names
            })

        # Для честного OOD-теста:
        # Берём примеры, где ЕСТЬ J, Q или K, и которые РЕШАЮТСЯ по OOD-правилам
        if ood_solution and any(c > 10 for c in combo):
            ood_data.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",
                     "content": f"Make 24 using these cards: {', '.join(card_names)}. Note: J=11, Q=12, K=13."}
                ],
                "target_numbers": ood_numbers,  # Важно: тут лежат 11, 12, 13
                "cards": card_names
            })

    # Перемешиваем
    random.seed(42)
    random.shuffle(id_data)
    random.shuffle(ood_data)

    print(f"Generated {len(id_data)} ID examples and {len(ood_data)} OOD examples.")

    with open("../old/data/train_id.jsonl", "w") as f:
        for item in id_data:
            f.write(json.dumps(item) + "\n")

    with open("../old/data/test_ood.jsonl", "w") as f:
        for item in ood_data:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    generate_dataset()