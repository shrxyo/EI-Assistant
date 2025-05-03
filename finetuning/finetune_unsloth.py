from unsloth import FastLanguageModel, SFTTrainer
from datasets import load_dataset
from transformers import TrainingArguments

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/llama-3-8b-Instruct",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)

dataset = load_dataset("json", data_files="data/multi_turn_dataset_cleaned.json", split="train")
dataset = dataset.map(lambda x: {"prompt": x["text"]})

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "v_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=42,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="prompt",
    max_seq_length=4096,
    args=TrainingArguments(
        output_dir="emotional-llama-lora",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        fp16=True,
    ),
)

trainer.train()
model.save_pretrained("emotional-llama-lora")
tokenizer.save_pretrained("emotional-llama-lora")