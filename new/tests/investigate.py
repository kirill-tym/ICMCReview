import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def extract_numbers(text):
    # Ищем все числа, включая отрицательные
    numbers = re.findall(r'-?\d+', text)
    # Сортируем и возвращаем кортеж (независимо от того, как числа перемешаны)
    return tuple(sorted(numbers))


def main():
    print("=== ЭТАП 1: Проверка наборов чисел (Data Leak) ===")
    train_bags = set()
    with open("../data/train_mixed_sft.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            # Берем только user prompt, чтобы не цеплять числа из system prompt
            user_msg = next((m["content"] for m in data["messages"] if m["role"] == "user"), "")
            train_bags.add(extract_numbers(user_msg))

    for test_file in ["test_id.jsonl", "test_ood.jsonl"]:
        test_bags = set()
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    user_msg = next((m["content"] for m in data["messages"] if m["role"] == "user"), "")
                    test_bags.add(extract_numbers(user_msg))

            intersection = train_bags.intersection(test_bags)
            print(
                f"[{test_file}] Совпадающих наборов чисел с трейном: {len(intersection)} уникальных наборов из {len(test_bags)}")
        except FileNotFoundError:
            print(f"Файл {test_file} не найден.")

    print("\n=== ЭТАП 2: Генерация первых 3 ответов ===")
    model_path = "../../../base_model/final"
    device = "mps"

    print(f"Загрузка модели из {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16
    ).to(device)

    for test_file in ["test_id.jsonl", "test_ood.jsonl"]:
        print(f"\n--- Первые 3 примера из {test_file} ---")
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                lines = [next(f) for _ in range(3)]

            for i, line in enumerate(lines):
                data = json.loads(line)
                prompt_messages = data["messages"][:-1]
                expected_answer = data["messages"][-1]["content"]

                text = tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                inputs = tokenizer([text], return_tensors="pt").to(device)

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=150,
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=False,
                        temperature=None,
                        top_p=None
                    )

                input_len = inputs.input_ids.shape[1]
                generated_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)

                user_prompt = next((m['content'] for m in prompt_messages if m['role'] == 'user'), "")
                print(f"\n[Пример {i + 1}]")
                print(f"Промпт (User): {user_prompt.strip()}")
                print(f"Ожидалось (Target): {expected_answer.strip()}")
                print(f"Выдала модель (Pred): {generated_text.strip()}")
        except Exception as e:
            print(f"Ошибка при обработке {test_file}: {e}")


if __name__ == "__main__":
    main()