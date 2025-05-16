import os
import json
import re
import random
from datasets import load_dataset

def clean_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r" +", " ", text)
    return text.strip()

dataset = load_dataset("vibhorag101/phr-mental-therapy-dataset-conversational-format")
conversations = dataset["train"]

conversations = list(conversations)
random.seed(42)
random.shuffle(conversations)
sampled_conversations = conversations[:1200]

processed_data = []

for convo in sampled_conversations:
    messages = convo.get("messages", [])
    if not messages:
        continue
    cleaned_messages = []
    for msg in messages:
        if "role" in msg and "content" in msg:
            role = msg["role"].lower()
            if role not in ["system", "user", "assistant"]:
                continue
            cleaned_messages.append({
                "role": role,
                "content": clean_text(msg["content"])
            })
    if cleaned_messages:
        processed_data.append({"messages": cleaned_messages})

train_size = int(0.8 * len(processed_data))
train_data = processed_data[:train_size]
test_data = processed_data[train_size:]

os.makedirs("data", exist_ok=True)
with open("data/openai_gpt4o_train.jsonl", "w") as f:
    for item in train_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

with open("data/openai_gpt4o_test.jsonl", "w") as f:
    for item in test_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Saved {len(train_data)} train and {len(test_data)} test conversations.")