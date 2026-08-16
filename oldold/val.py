import json
import torch
import ast
import re
import math
from transformers import AutoTokenizer, AutoModelForCausalLM


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


def main():
    # Путь к последнему чекпойнту RL-модели
    model_id = "./grpo_game24/checkpoint-3708"

    # Токенизатор можно брать из SFT (или из той же папки, если там сохранился config)
    tokenizer_id = "./sft_checkpoints/checkpoint-78"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    with open("../old/test_ood.jsonl", "r") as f:
        ood_data = [json.loads(line) for line in f]

    success_count = 0
    total = len(ood_data)

    print(f"Evaluating {total} OOD examples...")

    for i, item in enumerate(ood_data):
        # 1. ФИЛЬТРУЕМ ASSISTANT (Как в SFT/RL)
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
        is_success = False

        if expr:
            used_nums = extract_numbers(expr)
            if sorted(used_nums) == sorted(target):
                val = safe_eval(expr)
                if val is not None and math.isclose(val, 24.0, abs_tol=1e-5):
                    is_success = True

        # Отладка первых 3 примеров
        if i < 3:
            print(f"\n--- EXAMPLE {i + 1} ---")
            print(f"Target numbers: {target}")
            print(f"Extracted expr: {expr}")
            print(f"Extracted nums: {extract_numbers(expr) if expr else None}")
            print(f"Success: {is_success}")
            print("-" * 30)

        if is_success:
            success_count += 1

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{total} | Current Accuracy: {(success_count / (i + 1)) * 100:.2f}%")

    print(f"\nFinal OOD Semantic Success Rate: {(success_count / total) * 100:.2f}%")


if __name__ == "__main__":
    main()