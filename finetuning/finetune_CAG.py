import json
from transformers import AutoTokenizer
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments


class CustomDataset(Dataset):
    def __init__(self, tokenized_data):
        self.data = tokenized_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {key: val.squeeze() for key, val in self.data[idx].items()}

# Load the JSONL dataset
with open("/Users/dhruvpatel/Documents/GitHub/EI-Assistant/preprocessing/data/multi_turn_dataset_cleaned.jsonl", "r") as f:
    processed_data = [json.loads(line) for line in f]

model_name = "victunes/TherapyLlama-8B-v1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the dataset
def tokenize_function(example):
    tokens = tokenizer(example["text"], truncation=True, max_length=512, padding="max_length", return_tensors="pt")
    tokens["labels"] = tokens["input_ids"].clone()  # Labels are the same as input IDs for autoregressive models
    return tokens

tokenized_data = [tokenize_function(entry) for entry in processed_data]

# Create a PyTorch Dataset
train_dataset = CustomDataset(tokenized_data)

# Load the model
model = AutoModelForCausalLM.from_pretrained(model_name)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./cag-therapyllama",  # Output directory
    evaluation_strategy="steps",  # Evaluate at every save step
    save_steps=500,  # Save every 500 steps
    logging_steps=100,  # Log progress every 100 steps
    per_device_train_batch_size=2,  # Batch size per GPU
    num_train_epochs=3,  # Total epochs
    save_total_limit=2,  # Keep only last 2 checkpoints
    fp16=True,  # Mixed precision for faster training
    report_to="none",  # Disable reporting
)

# Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()

model.save_pretrained("./cag-therapyllama")
tokenizer.save_pretrained("./cag-therapyllama")


from transformers import pipeline

# Load the fine-tuned model and tokenizer
generator = pipeline("text-generation", model="./cag-therapyllama", tokenizer="./cag-therapyllama")

# Test input
system_prompt = "You are a helpful and joyous mental therapy assistant..."
user_input = "I've been feeling anxious about an upcoming exam. Any advice?"
formatted_input = f"<SYSTEM>{system_prompt}</SYSTEM> <USER>{user_input}</USER>"

# Generate response
response = generator(formatted_input, max_length=150, num_return_sequences=1)
print(response[0]['generated_text'])
