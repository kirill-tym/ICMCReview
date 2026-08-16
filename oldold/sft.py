import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig


def main():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Загружаем данные. Не сливаем их в текст, оставляем список messages!
    dataset = load_dataset("json", data_files="train_id.jsonl", split="train")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,  # <--- Меняем float16 на bfloat16
        low_cpu_mem_usage=False,
    )

    args = SFTConfig(
        output_dir="sft_checkpoints",
        max_length=1024,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,

        # --- ИСПРАВЛЕНИЯ ЗДЕСЬ ---
        learning_rate=1e-5,  # 1. Уменьшаем LR на порядок (для Full SFT нужно 1e-5 или 2e-5)
        max_grad_norm=1.0,  # 2. КРИТИЧНО: Gradient Clipping (обрезает градиенты, спасая от NaN)
        warmup_ratio=0.1,  # 3. Плавно разгоняем LR на первых 10% шагов
        weight_decay=0.01,  # 4. Добавляем небольшую L2-регуляризацию
        # -------------------------

        num_train_epochs=1,
        save_strategy="steps",
        save_steps=20,
        logging_steps=5,
        dataset_kwargs={
            "skip_prepare_dataset": False
        }
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,  # <--- Заменили tokenizer на processing_class
    )

    print("Starting SFT training...")
    trainer.train()


if __name__ == "__main__":
    main()