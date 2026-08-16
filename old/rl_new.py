import os

os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import ast
import re
import math
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig


# --- Вспомогательные функции парсинга ---
def safe_eval(expr_str: str):
    """Безопасно вычисляет математическое выражение."""
    try:
        expr_str = expr_str.strip()
        if not re.match(r"^[\d\s\+\-\*\/\(\)]+$", expr_str):
            return None
        node = ast.parse(expr_str, mode='eval')

        def _eval(n):
            if isinstance(n, ast.Expression):
                return _eval(n.body)
            elif isinstance(n, ast.BinOp):
                left, right = _eval(n.left), _eval(n.right)
                if left is None or right is None:
                    return None
                if isinstance(n.op, ast.Add): return left + right
                if isinstance(n.op, ast.Sub): return left - right
                if isinstance(n.op, ast.Mult): return left * right
                if isinstance(n.op, ast.Div): return left / right if right != 0 else None
            elif isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                return float(n.value)
            return None

        return _eval(node)
    except Exception:
        return None


def extract_numbers(expr_str: str):
    """Извлекает все числа из выражения."""
    return [int(n) for n in re.findall(r'\b\d+\b', expr_str)]


def extract_answer(completion: str):
    """Извлекает выражение из тегов <answer>."""
    match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    return match.group(1).strip() if match else None


# --- Чистая Outcome-Based функция награды (ICML 2025) ---
def outcome_math_reward_func(completions, target_numbers, **kwargs):
    """
    Чистая бинарная награда:
    +1.0 выдается ТОЛЬКО если:
      1. Выражение находится в тегах <answer>.
      2. Использованы ровно 4 заданных числа из условия.
      3. Результат математически равен 24.0.
    В остальных случаях — строго 0.0.
    """
    rewards = []
    for c, target in zip(completions, target_numbers):
        expr = extract_answer(c)
        score = 0.0
        if expr and sorted(extract_numbers(expr)) == sorted(target):
            val = safe_eval(expr)
            if val is not None and math.isclose(val, 24.0, abs_tol=1e-5):
                score = 1.0
        rewards.append(score)
    return rewards


def main():
    model_id = "./sft_format_teacher/final"
    dataset_path = "train_id.jsonl"
    output_dir = "../grpo_game24_outcome_v3"

    print(f"Загрузка токенизатора из {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Загрузка датасета {dataset_path}...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    def format_prompt(example):
        prompt_msgs = [m for m in example["messages"] if m["role"] != "assistant"]
        example["prompt"] = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        return example

    dataset = dataset.map(format_prompt)

    print("Загрузка модели для GRPO...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )

    # Конфигурация GRPO для 3x эпох, высокого exploration и экономии диска
    training_args = GRPOConfig(
        output_dir=output_dir,
        learning_rate=2e-6,
        num_train_epochs=3,
        logging_steps=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,  # Возвращаем 8, так как группа снова 8
        num_generations=16,  # Безопасно для памяти MacBook
        generation_batch_size=16,  # <--- ТОТ САМЫЙ ПАРАМЕТР, КОТОРЫЙ ЧИНИТ ОШИБКУ
        max_completion_length=512,
        temperature=0.9,  # ОСТАВЛЯЕМ ВЫСОКИМ! Заставим ее думать вариативно
        top_p=0.95,  # Отрезаем бред
        beta=0.0,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        use_vllm=False,
        report_to="none"
    )

    print("Инициализация GRPOTrainer с единственной Outcome-Based наградой...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[outcome_math_reward_func],
        args=training_args,
        train_dataset=dataset,
    )

    print("Старт GRPO обучения...")
    trainer.train()

    print(f"Сохранение финальной RL модели в {output_dir}/final")
    trainer.save_model(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))


if __name__ == "__main__":
    main()