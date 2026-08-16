import re
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from verifier import check_solution


def strict_24_reward(completions, numbers, **kwargs):
    rewards = []
    for completion, nums in zip(completions, numbers):
        if isinstance(completion, list):
            text = completion[-1]["content"]
        else:
            text = str(completion)

        # Регуляркой достаем контент строго из ПЕРВОГО закрытого тега <answer>...</answer>
        match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        if not match:
            rewards.append(0.0)
            continue

        expr = match.group(1).strip()

        # Если модель попыталась вставить еще один тег внутри — это спам/хакинг
        if "<answer>" in expr or "</answer>" in expr:
            rewards.append(0.0)
            continue

        # Проверяем математическое решение через verifier
        score = 0.05  # Небольшой бонус за корректно отформатированный ответ
        try:
            if check_solution(nums, expr):
                score = 1.0  # Полная награда за правильное математическое выражение
        except Exception:
            pass

        rewards.append(float(score))

    return rewards


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_path = "../oldold/sft_model"

    print(f"Loading SFT model for RL on device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float32,
    ).to(device)

    dataset = load_dataset("json", data_files={"train": "data/sft_train.jsonl"})

    def format_prompt(example):
        user_message = example["messages"][0]["content"]
        messages = [{"role": "user", "content": user_message}]

        return {
            "prompt": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ),
            "numbers": example["numbers"]
        }

    train_dataset = dataset["train"].map(format_prompt)

    training_args = GRPOConfig(
        output_dir="../oldold/rl_model",
        learning_rate=5e-6,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=4,
        temperature=0.8,  # Добавляем сэмплирование для разнообразия генераций
        max_completion_length=64,
        max_steps=100,  # Немного увеличим число шагов
        logging_steps=5,
        save_strategy="no",
        fp16=False,
        bf16=False,
        dataloader_pin_memory=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=strict_24_reward,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("--- Starting Fixed GRPO RL Training ---")
    trainer.train()

    print("--- Saving Final RL Model ---")
    trainer.model.save_pretrained("./rl_model")
    tokenizer.save_pretrained("./rl_model")
    print("✅ Fixed RL Stage Completed Successfully!")


if __name__ == "__main__":
    main()