import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from verifier import check_solution


def evaluate_holdout(model_path):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n==========================================")
    print(f" Evaluating Model on Holdout Test Set: {model_path}")
    print(f"==========================================")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32
    ).to(device)
    model.eval()

    with open("../data/test_holdout.jsonl", "r", encoding="utf-8") as f:
        test_samples = [json.loads(line) for line in f]

    correct = 0
    total = len(test_samples)

    print(f"Testing on {total} unseen samples...\n")

    for i, sample in enumerate(test_samples, 1):
        prompt_text = sample["prompt"]
        numbers = sample["numbers"]

        messages = [{"role": "user", "content": prompt_text}]
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        generated_text = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        is_correct = check_solution(numbers, generated_text)
        if is_correct:
            correct += 1

        print(f"[Test #{i}/{total} | Numbers: {numbers}] -> {'✅ PASS' if is_correct else '❌ FAIL'}")
        if not is_correct:
            print(f"   Model Output: {generated_text.strip()}")
            print(f"   Reference:    {sample['reference_solution']}")

    acc = (correct / total) * 100
    print(f"\n------------------------------------------")
    print(f"Final Generalization Accuracy: {acc:.2f}% ({correct}/{total})")
    print(f"------------------------------------------\n")


if __name__ == "__main__":
    # evaluate_holdout("./sft_model")
    evaluate_holdout("../oldold/rl_model")