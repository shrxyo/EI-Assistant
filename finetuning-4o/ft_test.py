

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)  
response = client.chat.completions.create(
    model="ft:gpt-4o-mini-2024-07-18:hawshiuan:therapy-bot:BU1Z7786",
    messages=[
        {"role": "system", "content": "You are a helpful and joyous mental therapy assistant."},
        {"role": "user", "content": "im just very sad."}
    ]
)

print(response.choices[0].message.content)