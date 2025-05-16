import time
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) 
job_id = "ftjob-D5adIGlSk6BLKfmoFpllHrBn"

while True:
    job_status = client.fine_tuning.jobs.retrieve(job_id)
    print(f"Status: {job_status.status}")
    if job_status.status in ["succeeded", "failed", "cancelled"]:
        print("Final job status:", job_status.status)
        if job_status.status == "succeeded":
            print("Fine-tuned model name:", job_status.fine_tuned_model)
        elif job_status.status == "failed":
            print("Error:", job_status.error)
        break
    time.sleep(30)  