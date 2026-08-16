import json

def get_prompts(filename):
    prompts = set()
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                # Берем только промпт (все сообщения, кроме финального ответа Assistant)
                prompt = str(data["messages"][:-1])
                prompts.add(prompt)
    except FileNotFoundError:
        pass
    return prompts

train = get_prompts("../data/train_mixed_sft.jsonl")
test_id = get_prompts("../data/test_id.jsonl")
test_ood = get_prompts("../data/test_ood.jsonl")

print(f"Пересечение Train и Test ID: {len(train.intersection(test_id))} примеров из {len(test_id)}")
print(f"Пересечение Train и Test OOD: {len(train.intersection(test_ood))} примеров из {len(test_ood)}")