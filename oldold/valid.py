import json
import torch
import ast
import re
import math
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# --- Функции парсинга и проверки (из твоего пайплайна) ---
def safe_eval(expr_str: str):
    try:
        expr_str = expr_str.strip()
        if not re.match(r"^[\d\s\+\-\*\/\(\)]+$", expr_str): return None
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
            elif isinstance(n, ast.Constant):
                return float(n.value)
            return None

        return _eval(node)
    except Exception:
        return None


def extract_numbers(expr_str: str):
    return [int(n) for n in re.findall(r'\b\d+\b', expr_str)]


def extract_answer(completion: str):
    match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    return match.group(1).strip() if match else None


# --- Универсальная функция оценки датасета ---
def evaluate_dataset(model, tokenizer, dataset_path, desc="Оценка"):
    try:
        with open(dataset_path, "r") as f:
            data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"Файл {dataset_path} не найден! Пропускаем...")
        return 0.0

    success_count = 0
    total = len(data)

    for item in tqdm(data, desc=desc):
        prompt_msgs = [m for m in item["messages"] if m["role"] != "assistant"]
        prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        target = item["target_numbers"]
        expr = extract_answer(generated_text)

        if expr:
            used_nums = extract_numbers(expr)
            if sorted(used_nums) == sorted(target):
                val = safe_eval(expr)
                if val is not None and math.isclose(val, 24.0, abs_tol=1e-5):
                    success_count += 1

    return (success_count / total) * 100 if total > 0 else 0.0


# --- Главный цикл ---
def main():
    # Автоматически подхватит MPS
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Используем устройство: {device}")

    # Укажи тут свои актуальные пути
    tokenizer_path = "sft_checkpoints/checkpoint-20"
    sft_model_path = "sft_checkpoints/checkpoint-78"
    rl_model_path = "grpo_game24/checkpoint-3708"

    # Датасеты
    id_dataset = "train_id.jsonl"  # Или test_id.jsonl, если ты оставил сплит
    ood_dataset = "test_ood.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 1. Оцениваем SFT
    print("\n" + "=" * 50)
    print("1. Загрузка и оценка SFT модели")
    print("=" * 50)
    sft_model = AutoModelForCausalLM.from_pretrained(sft_model_path, torch_dtype=torch.bfloat16).to(device)
    sft_model.eval()

    sft_id_acc = evaluate_dataset(sft_model, tokenizer, id_dataset, "SFT на ID ")
    sft_ood_acc = evaluate_dataset(sft_model, tokenizer, ood_dataset, "SFT на OOD")

    # Чистим память перед загрузкой второй модели
    del sft_model
    if device == "mps":
        torch.mps.empty_cache()

    # 2. Оцениваем RL
    print("\n" + "=" * 50)
    print("2. Загрузка и оценка RL модели")
    print("=" * 50)
    rl_model = AutoModelForCausalLM.from_pretrained(rl_model_path, torch_dtype=torch.bfloat16).to(device)
    rl_model.eval()

    rl_id_acc = evaluate_dataset(rl_model, tokenizer, id_dataset, "RL на ID ")
    rl_ood_acc = evaluate_dataset(rl_model, tokenizer, ood_dataset, "RL на OOD")

    del rl_model
    if device == "mps":
        torch.mps.empty_cache()

    # 3. Финальный отчет
    print("\n" + "=" * 60)
    print("ИТОГОВАЯ ТАБЛИЦА СРАВНЕНИЯ (Accuracy %)")
    print("=" * 60)
    print(f"Модель       | In-Distribution | Out-Of-Distribution | Падение")
    print("-" * 60)
    print(f"SFT Checkpoint | {sft_id_acc:>14.2f}% | {sft_ood_acc:>18.2f}% | {(sft_id_acc - sft_ood_acc):>6.2f}%")
    print(f"RL Checkpoint  | {rl_id_acc:>14.2f}% | {rl_ood_acc:>18.2f}% | {(rl_id_acc - rl_ood_acc):>6.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()