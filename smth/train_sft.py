import os
import shutil
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model_id = "Qwen/Qwen2.5-0.5B-Instruct"

    print(f"Loading model on device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    ).to(device)

    dataset = load_dataset("json", data_files={"train": "data/sft_train.jsonl"})

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    args = SFTConfig(
        output_dir="../oldold/sft_checkpoints",
        max_length=256,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=2,
        logging_steps=10,
        save_strategy="no",
        fp16=False,
        bf16=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("--- Starting SFT Training ---")
    trainer.train()

    print("--- Merging and Saving Clean SFT Model ---")
    trainer.model.eval()
    # 👈 Мерджим на CPU во избежание бага MPS Metal
    merged_model = trainer.model.to("cpu").merge_and_unload()

    target_dir = "../oldold/sft_model"
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    merged_model.save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)

    if os.path.exists("../oldold/sft_checkpoints"):
        shutil.rmtree("../oldold/sft_checkpoints")

    print("✅ SFT Stage Completed Successfully and Cleanly!")


if __name__ == "__main__":
    main()