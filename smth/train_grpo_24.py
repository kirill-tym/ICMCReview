import ast
import re
import math
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig


# ==========================================
# 1. Безопасный парсер и эвалуатор выражений
# ==========================================
def safe_eval(expr_str: str):
    """Безопасно вычисляет значение математического выражения."""
    try:
        expr_str = expr_str.strip()
        # Разрешаем только цифры, скобки и базовые арифметические операторы
        if not re.match(r"^[\d\s\+\-\*\/\(\)]+$", expr_str):
            return None
        node = ast.parse(expr_str, mode='eval')

        def _eval(n):
            if isinstance(n, ast.Expression):
                return _eval(n.body)
            elif isinstance(n, ast.BinOp):
                left, right = _eval(n.left), _eval(n.right)
                if left is None or right is None: return None
                if isinstance(n.op, ast.Add): return left + right
                if isinstance(n.op, ast.Sub): return left - right
                if isinstance(n.op, ast.Mult): return left * right
                if isinstance(n.op, ast.Div): return left / right if right != 0 else None
            elif isinstance(n, ast.UnaryOp):
                op = _eval(n.operand)
                return -op if isinstance(n.op, ast.USub) and op is not None else op
            elif isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                return float(n.value)
            return None

        return _eval(node)
    except Exception:
        return None


def extract_numbers_from_expr(expr_str: str):
    """Извлекает все числа из выражения."""
    return [int(n) for n in re.findall(r'\b\d+\b', expr_str)]


def extract_answer_content(completion: str):
    """Извлекает текст из внутри тегов <answer>...</answer>."""
    match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


# ==========================================
# 2. Набор функций наград (Dense Reward System)
# ==========================================

def format_reward_func(completions, **kwargs) -> list[float]:
    """1. Награда за соблюдение формата CoT и XML тегов."""
    rewards = []
    for completion in completions:
        score = 0.0
        if "<think>" in completion and "</think>" in completion:
            score += 0.05
        if "<answer>" in completion and "</answer>" in completion:
            score += 0.1
        rewards.append(score)
    return rewards


def numbers_usage_reward_func(completions, target_numbers, **kwargs) -> list[float]:
    """2. Награда за использование ровно тех 4 чисел, которые заданы в условии."""
    rewards = []
    for completion, nums in zip(completions, target_numbers):
        score = 0.0
        expr = extract_answer_content(completion)
        if expr:
            used_nums = extract_numbers_from_expr(expr)
            if sorted(used_nums) == sorted(nums):
                score += 0.25
        rewards.append(score)
    return rewards


def closeness_reward_func(completions, **kwargs) -> list[float]:
    """3. Непрерывная награда за близость итогового ответа к 24."""
    rewards = []
    for completion in completions:
        score = 0.0
        expr = extract_answer_content(completion)
        if expr:
            val = safe_eval(expr)
            if val is not None and not math.isnan(val) and not math.isinf(val):
                diff = abs(24.0 - val)
                # Чем ближе к 24, тем выше награда (максимум 0.4)
                score += max(0.0, 0.4 - 0.02 * diff)
        rewards.append(score)
    return rewards


def exact_success_reward_func(completions, target_numbers, **kwargs) -> list[float]:
    """4. Финальный джекпот за точное решение задачи."""
    rewards = []
    for completion, nums in zip(completions, target_numbers):
        score = 0.0
        expr = extract_answer_content(completion)
        if expr:
            used_nums = extract_numbers_from_expr(expr)
            val = safe_eval(expr)
            if sorted(used_nums) == sorted(nums) and val is not None:
                if math.isclose(val, 24.0, abs_tol=1e-5):
                    score += 1.0  # Главный бонус!
        rewards.append(score)
    return rewards


# ==========================================
# 3. Подготовка датасета и промптов
# ==========================================

SYSTEM_PROMPT = """You are an expert arithmetic solver. Your task is to reach the number 24 using exact specified 4 numbers and operators +, -, *, /. Each number must be used exactly once.

Respond in the following format:
<think>
Step by step reasoning about arithmetic combinations...
</think>
<answer>
(expression that equals 24)
</answer>"""


def create_dataset():
    # Пример набора чисел для Game of 24
    raw_data = [
                   {"numbers": [4, 1, 8, 7]},
                   {"numbers": [1, 5, 5, 5]},
                   {"numbers": [3, 3, 8, 8]},
                   {"numbers": [2, 3, 4, 6]},
                   {"numbers": [1, 2, 3, 4]},
                   {"numbers": [10, 10, 4, 4]},
                   {"numbers": [6, 6, 6, 6]},
                   {"numbers": [8, 4, 2, 1]},
               ] * 100  # Дублируем для создания эпох

    formatted_data = []
    for item in raw_data:
        nums_str = ", ".join(map(str, item["numbers"]))
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Make 24 using these numbers: {nums_str}"}
        ]
        formatted_data.append({
            "prompt": prompt,
            "target_numbers": item["numbers"]
        })
    return Dataset.from_list(formatted_data)


# ==========================================
# 4. Запуск GRPO обучения
# ==========================================

def main():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"  # Укажи путь к своей SFT модели

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = create_dataset()

    training_args = GRPOConfig(
        output_dir="./grpo_game24_results",
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_steps=10,  # Заменили deprecated warmup_ratio
        lr_scheduler_type="cosine",
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,  # Теперь 1 * 16 = 16 (кратно num_generations)

        # КРИТИЧЕСКИЕ НАСТРОЙКИ GRPO
        num_generations=16,  # Generates 16 outputs per prompt
        max_completion_length=512,
        temperature=0.8,
        beta=0.04,
    )

    trainer = GRPOTrainer(
        model=model_id,
        reward_funcs=[
            format_reward_func,
            numbers_usage_reward_func,
            closeness_reward_func,
            exact_success_reward_func
        ],
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting GRPO training...")
    trainer.train()


if __name__ == "__main__":
    main()