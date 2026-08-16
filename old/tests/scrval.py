import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re


def print_ood_logs(model_path, dataset_path, num_samples=3):
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Загружаем модель с {model_path} на {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    with open(dataset_path, "r") as f:
        data = [json.loads(line) for line in f]

    print("\n" + "=" * 80)
    print(" АНАЛИЗ ОШИБОК НА TEST OOD (J=11, Q=12, K=13)")
    print("=" * 80)

    count = 0
    for item in data:
        if count >= num_samples:
            break

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

        # Пытаемся красиво извлечь think и answer
        think_match = re.search(r"<think>(.*?)</think>", generated_text, re.DOTALL)
        ans_match = re.search(r"<answer>(.*?)</answer>", generated_text, re.DOTALL)

        think_block = think_match.group(1).strip() if think_match else "Нет блока <think>"
        answer_block = ans_match.group(1).strip() if ans_match else "Нет блока <answer>"
        target = item["target_numbers"]

        print(f"\n[Пример {count + 1}]")
        print(f"ПРОМПТ:\n{prompt_msgs[0]['content']}")
        print(f"ЦЕЛЕВЫЕ ЧИСЛА: {target}")
        print("-" * 40)
        print(f"РАССУЖДЕНИЯ <think>:\n{think_block}")
        print("-" * 40)
        print(f"ОТВЕТ <answer>: {answer_block}")
        print("=" * 80)

        count += 1


if __name__ == "__main__":
    # Укажи путь к своей RL-модели
    rl_model_path = "../../present/sft_format_teacher/final"
    ood_data_path = "../data/test_id.jsonl"

    print_ood_logs(rl_model_path, ood_data_path, num_samples=3)