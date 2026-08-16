import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import ast
import re
import math
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig

def safe_eval(expr_str: str):
    try:
        expr_str = expr_str.strip()
        if not re.match(r"^[\d\s\+\-\*\/\(\)]+$", expr_str): return None
        node = ast.parse(expr_str, mode='eval')
        def _eval(n):
            if isinstance(n, ast.Expression): return _eval(n.body)
            elif isinstance(n, ast.BinOp):
                left, right = _eval(n.left), _eval(n.right)
                if left is None or right is None: return None
                if isinstance(n.op, ast.Add): return left + right
                if isinstance(n.op, ast.Sub): return left - right
                if isinstance(n.op, ast.Mult): return left * right
                if isinstance(n.op, ast.Div): return left / right if right != 0 else None
            elif isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                return float(n.value)
            return None
        return _eval(node)
    except Exception: return None

def extract_numbers(expr_str: str):
    return [int(n) for n in re.findall(r'\b\d+\b', expr_str)]

def extract_answer(completion: str):
    match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    return match.group(1).strip() if match else None

# --- Награды ---
def format_reward_func(completions, **kwargs):
    return [0.1 if "<think>" in c and "</answer>" in c else 0.0 for c in completions]

def numbers_usage_reward_func(completions, target_numbers, **kwargs):
    rewards = []
    for c, target in zip(completions, target_numbers):
        expr = extract_answer(c)
        if expr and sorted(extract_numbers(expr)) == sorted(target):
            rewards.append(0.3)
        else:
            rewards.append(0.0)
    return rewards

def math_reward_func(completions, target_numbers, **kwargs):
    rewards = []
    for c, target in zip(completions, target_numbers):
        expr = extract_answer(c)
        score = 0.0
        if expr and sorted(extract_numbers(expr)) == sorted(target):
            val = safe_eval(expr)
            if val is not None:
                if math.isclose(val, 24.0, abs_tol=1e-5):
                    score += 1.0
                else:
                    score += max(0.0, 0.5 - 0.05 * abs(24.0 - val))
        rewards.append(score)
    return rewards

def main():
    model_id = "./sft_checkpoints/checkpoint-20"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files="train_id.jsonl", split="train")

    def format_prompt(example):
        prompt_msgs = [m for m in example["messages"] if m["role"] != "assistant"]
        example["prompt"] = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        return example

    dataset = dataset.map(format_prompt)

    print("Loading model for Full Fine-Tuning...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    )

    training_args = GRPOConfig(
        output_dir="../grpo_game24",
        learning_rate=2e-6,
        logging_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        num_generations=16,
        max_completion_length=512,
        temperature=0.8,
        beta=0.0,      # КЛЮЧЕВОЙ МОМЕНТ: отключаем KL-штраф, убирая потребность в ref_model
        use_vllm=False,
    )

    print("Initializing GRPOTrainer for Full FT...")
    trainer = GRPOTrainer(
        model=model,  # Передаем единственный чистый объект модели
        reward_funcs=[format_reward_func, numbers_usage_reward_func, math_reward_func],
        args=training_args,
        train_dataset=dataset,
    )

    print("Starting Full FT GRPO training...")
    trainer.train()

if __name__ == "__main__":
    main()