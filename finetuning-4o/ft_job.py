from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)  

job = client.fine_tuning.jobs.create(
    training_file="file-S6JQPBjdkVhb17sVXKxgRz",      
    validation_file="file-EaTZMnBZfSeh47geAfdQxz",             
    model="gpt-4o-mini-2024-07-18",                                 
    suffix="therapy-bot"                            
)

print("Fine-tuning job ID:", job.id)