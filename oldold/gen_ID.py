import json
import random


def generate_test_id():
    # Читаем уже существующий обучающий датасет
    try:
        with open("../old/data/train_id.jsonl", "r") as f:
            train_data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print("Ошибка: файл train_id.jsonl не найден!")
        return

    test_data = []

    for item in train_data:
        original_cards = item["cards"]
        shuffled_cards = original_cards.copy()

        # Перемешиваем карты так, чтобы порядок точно изменился (если все 4 карты не одинаковые)
        if len(set(original_cards)) > 1:
            while shuffled_cards == original_cards:
                random.shuffle(shuffled_cards)

        # Собираем новый промпт с перемешанными картами
        new_item = {
            "messages": [
                item["messages"][0],  # System prompt
                {
                    "role": "user",
                    "content": f"Make 24 using these cards: {', '.join(shuffled_cards)}. Note: J, Q, K are worth 10."
                }
            ],
            "target_numbers": item["target_numbers"],
            "cards": shuffled_cards
        }
        test_data.append(new_item)

    # Перемешиваем сам датасет и берем 800 примеров для быстрого теста
    random.seed(42)
    random.shuffle(test_data)
    test_sample = test_data[:800]

    # Сохраняем в test_id.jsonl
    with open("../old/data/test_id.jsonl", "w") as f:
        for item in test_sample:
            f.write(json.dumps(item) + "\n")

    print(f"Успешно сгенерирован test_id.jsonl на {len(test_sample)} примеров.")
    print("Порядок карт перемешан, для модели это будут 'новые' промпты.")


if __name__ == "__main__":
    generate_test_id()