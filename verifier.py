import re


def check_solution(prompt_numbers: list[int], completion: str, target: int = 24) -> float:
    """
    Проверяет:
    1. Наличие тегов <answer>...</answer>
    2. Использование ТОЛЬКО предоставленных чисел
    3. Равенство итогового выражения target (24)
    """
    match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    if not match:
        return 0.0

    expr_str = match.group(1).strip()

    # Извлекаем все числа из выражения
    used_numbers = [int(n) for n in re.findall(r'\d+', expr_str)]
    if sorted(used_numbers) != sorted(prompt_numbers):
        return 0.0  # Модель использовала не те числа или изменила их количество

    # Безопасное вычисление выражения
    try:
        if not re.match(r'^[\d\s\+\-\*\/\(\)]+$', expr_str):
            return 0.0
        result = eval(expr_str)
        if abs(result - target) < 1e-5:
            return 1.0
    except Exception:
        return 0.0

    return 0.0
