from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)  

train_file = client.files.create(
    file=open("data/openai_gpt4o_train.jsonl", "rb"),
    purpose="fine-tune"
)
print("Training file uploaded:", train_file.id)

val_file = client.files.create(
    file=open("data/openai_gpt4o_test.jsonl", "rb"),
    purpose="fine-tune"
)
print("Validation file uploaded:", val_file.id)