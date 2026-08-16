import torch
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

def format_chat_template(example, tokenizer):
    """
    Применяет chat_template токенизатора для корректной сборки
    System, User и Assistant сообщений в единую строку.
    """
    text = tokenizer.apply_chat_template(example["messages"], tokenize=False)
    return {"text": text}

def main():
    # 1. Настройки модели и путей
    model_id = "Qwen/Qwen2.5-Math-7B"
    dataset_path = "train_mixed_sft.jsonl" # Обучаем на сбалансированных 20k (10k ID + 10k OOD)
    output_dir = "../base_model" # Новая папка для базовой модели

    print(f"Загрузка токенизатора и модели: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Загружаем модель (используем torch_dtype - это стандартный аргумент для transformers)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # 2. Подготовка датасета
    print(f"Загрузка датасета: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    # Применяем форматирование
    dataset = dataset.map(lambda x: format_chat_template(x, tokenizer), num_proc=4)

    # 3. Конфигурация SFT
    training_args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        max_length=512,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=31,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True if torch.backends.mps.is_available() or torch.cuda.is_available() else False,
        report_to="none",
        remove_unused_columns=True,
        dataloader_pin_memory=False,
    )

    # 4. Инициализация тренера
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    # 5. Запуск обучения
    print("Старт базового SFT (Mixed 20k)...")
    trainer.train()

    print(f"Сохранение модели в {output_dir}/final")
    trainer.save_model(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))

if __name__ == "__main__":
    main()