import re


def check_solution(numbers, text):
    """
    Проверяет, содержит ли ответ выражение в <answer>...</answer>,
    равное 24 и использующее ровно заданные числа.
    """
    pattern = r"<answer>(.*?)</answer>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return False

    expr = match.group(1).strip()

    # Извлекаем числа из выражения
    expr_numbers = [int(n) for n in re.findall(r"\d+", expr)]
    if sorted(expr_numbers) != sorted(numbers):
        return False

    # Проверяем математику
    try:
        if not re.match(r"^[\d\s\+\-\*\/\(\)]+$", expr):
            return False
        val = eval(expr)
        return abs(val - 24.0) < 1e-5
    except Exception:
        return False