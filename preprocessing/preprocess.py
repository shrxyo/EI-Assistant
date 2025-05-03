from datasets import load_dataset
import json
import re
import os

#data cleaning
def clean_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n")  
    text = re.sub(r"\n+", "\n", text)                      
    text = re.sub(r" +", " ", text)                        
    return text.strip()

#multi turn string
def format_conversation(messages):
    dialogue_text = ""
    for msg in messages:
        role = msg["role"].lower()  
        content = clean_text(msg["content"])
        dialogue_text += f"{role}: {content}\n"
    return clean_text(dialogue_text)

dataset = load_dataset("vibhorag101/phr-mental-therapy-dataset-conversational-format")
conversations = dataset["train"]

processed_data = []

for convo in conversations:
    convo_id = convo.get("identity", "")
    messages = convo["messages"]
    full_text = format_conversation(messages)

    processed_data.append({
        "id": convo_id,
        "text": full_text
    })

os.makedirs("data", exist_ok=True)

with open("data/multi_turn_dataset_cleaned.json", "w") as f:
    json.dump(processed_data, f, indent=2)

with open("data/multi_turn_dataset_cleaned.jsonl", "w") as f:
    for entry in processed_data:
        f.write(json.dumps(entry) + "\n")
