import os
# Перенаправляем трафик через зеркало, которое обычно не режется
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# На всякий случай оставляем отключение hf_transfer
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import torch
import json
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def check_correctness(predicted_text, expected_text):
    """Извлекает выражение, проверяет числа на читеринг и вычисляет результат"""

    def get_expr(text):
        # Достаем всё, что находится между тегами (включая переносы строк)
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None

    pred_expr = get_expr(predicted_text)
    exp_expr = get_expr(expected_text)

    # Если модель вообще не выдала тег или таргет битый
    if not pred_expr or not exp_expr:
        return False

    # 1. Проверка на галлюцинации (модель должна использовать только разрешенные числа)
    # Достаем все числа из таргета и из ответа модели, сортируем и сравниваем
    pred_nums = sorted(re.findall(r'\d+', pred_expr))
    exp_nums = sorted(re.findall(r'\d+', exp_expr))

    if pred_nums != exp_nums:
        return False

    # 2. Проверка правильности математики
    try:
        # Вычисляем строку как питоновский код
        result = eval(pred_expr)
        # Проверяем равенство 24 (используем abs для защиты от погрешности float)
        if abs(float(result) - 24.0) < 1e-5:
            return True
    except Exception:
        # Если модель сгенерила невалидный синтаксис (например: "24 + * 2")
        pass

    return False


def evaluate_dataset(model, tokenizer, dataset_path, device):
    print(f"\n[{dataset_path}] Оценка точности...")
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Файл {dataset_path} не найден. Пропуск.")
        return

    correct = 0
    total = len(lines)

    for line in tqdm(lines, desc=f"Инференс {dataset_path}"):
        data = json.loads(line)
        messages = data["messages"]

        # Изолируем prompt (System + User), убираем Assistant
        prompt_messages = messages[:-1]

        # Вытаскиваем ожидаемый ответ из исходного Assistant message
        expected_response = messages[-1]["content"]

        # Рендерим промпт
        prompt = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        # Жадная генерация для точной и воспроизводимой проверки
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False  # Убрали temperature=0.0 и добавили явный флаг жадной генерации
            )

        # Декодируем только новые токены (ответ модели)
        generated_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        if check_correctness(generated_text, expected_response):
            correct += 1

    accuracy = (correct / total) * 100
    print(f"--> Точность на {dataset_path}: {accuracy:.2f}% ({correct}/{total})")


def main():
    model_path = "Qwen/Qwen2.5-Math-7B"

    # Определение устройства (выберет MPS для Mac M-серии)
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Используемое устройство: {device}")

    print(f"Загрузка базовой модели из {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16  # Заменили устаревший torch_dtype на dtype
    ).to(device)

    model.eval()

    # Запускаем проверку
    evaluate_dataset(model, tokenizer, "test_id.jsonl", device)
    evaluate_dataset(model, tokenizer, "test_ood.jsonl", device)


if __name__ == "__main__":
    main()