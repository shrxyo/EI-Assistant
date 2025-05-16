import os
import numpy as np
import random
import json
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from openai import OpenAI

class OpenAITherapyBot:
    def _init_(self, api_key=None, model="ft:gpt-4o-mini-2024-07-18:hawshiuan:therapy-bot:BU1Z7786"):

        
        print(f"Initializing OpenAI Therapy Bot with model: {model}")
        

        self.client = OpenAI(api_key=api_key)
        self.model = model
        

        self.conversation_history = []
        self.complete_history = []
        
    
        self.system_prompt = """You are a compassionate therapeutic assistant.
Your goal is to help users process emotions, provide support, and offer
evidence-based coping strategies. Be empathetic and non-judgmental.
IMPORTANT GUIDELINES:
1. ONLY respond based on information explicitly shared by the user
2. NEVER invent or assume details about the user's situation
3. When uncertain, acknowledge limitations or ask clarifying questions
4. Do not guarantee specific outcomes or make definitive claims
5. Do not fabricate studies, statistics, or resources
6. If a question requires specialized knowledge, clearly state your limitations
7. Prioritize being accurate over being comprehensive"""
    
    def add_message(self, role, content):
        """Add a message to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})
        self.complete_history.append({"role": role, "content": content})
    
    def generate_response(self, messages, max_tokens=2500):
        """Generate a response using OpenAI's API."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                frequency_penalty=1.0,
                presence_penalty=0.6
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I encountered an issue generating a response. Let's continue our conversation."


random.seed(42)
np.random.seed(42)


def load_therapy_dataset():
    """Load the therapy dataset."""
    print("Loading mental therapy dataset...")
    ds = load_dataset("vibhorag101/phr-mental-therapy-dataset-conversational-format")
    print(f"Dataset loaded with {len(ds['test'])} test examples")
    return ds


def setup_similarity_model():
    """Load the sentence embedder for similarity calculation."""
    print("Loading similarity model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return embedder


def compute_semantic_similarity(text1, text2, embedder):
    """Compute semantic similarity between two texts."""
    embeddings1 = embedder.encode([text1], convert_to_numpy=True)
    embeddings2 = embedder.encode([text2], convert_to_numpy=True)
    similarity = cosine_similarity(embeddings1, embeddings2)[0][0]
    return float(similarity)


def generate_with_specific_prompt(bot, user_message, system_prompt):
    """Generate a response using a specific system prompt and current conversation history."""
    
    original_history = bot.conversation_history.copy()
    original_complete = bot.complete_history.copy()
    
    
    test_messages = [{"role": "system", "content": system_prompt}] + original_history
    test_messages.append({"role": "user", "content": user_message})
    
    
    response = bot.generate_response(test_messages)
    
    
    bot.conversation_history = original_history
    bot.complete_history = original_complete
    
    return response


def evaluate_openai_bot(bot, embedder, test_set, num_samples=50):
    
    system_prompt = """You are a compassionate therapeutic assistant.
Your goal is to help users process emotions, provide support, and offer
evidence-based coping strategies. Be empathetic and non-judgmental.
IMPORTANT GUIDELINES:
1. ONLY respond based on information explicitly shared by the user
2. NEVER invent or assume details about the user's situation
3. When uncertain, acknowledge limitations or ask clarifying questions
4. Do not guarantee specific outcomes or make definitive claims
5. Do not fabricate studies, statistics, or resources
6. If a question requires specialized knowledge, clearly state your limitations
7. Prioritize being accurate over being comprehensive"""

    
    if len(test_set) > num_samples:
        indices = random.sample(range(len(test_set)), num_samples)
        test_sample = [test_set[i] for i in indices]
    else:
        test_sample = test_set

    conversation_avgs = []
    all_similarities = []


    for idx, conversation in enumerate(tqdm(test_sample, desc="Evaluating conversations")):
        messages = conversation["messages"]
        similarity_scores = []
        
        
        bot.conversation_history = []
        bot.complete_history = []

        print(f"\nEvaluating conversation {idx+1}/{len(test_sample)}")
        
        
        for i in range(len(messages) - 1):
            if messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant":
                user_msg = messages[i]["content"]
                ground_truth = messages[i+1]["content"]
                
                
                generated_response = generate_with_specific_prompt(bot, user_msg, system_prompt)
                
                
                similarity = compute_semantic_similarity(generated_response, ground_truth, embedder)
                similarity_scores.append(similarity)
                all_similarities.append(similarity)
                
                
                bot.add_message("user", user_msg)
                bot.add_message("assistant", ground_truth)

        
        if similarity_scores:
            conv_avg = float(np.mean(similarity_scores))
            conversation_avgs.append(conv_avg)
            print(f"Conversation {idx+1} avg similarity: {conv_avg:.4f} ({len(similarity_scores)} turns)")

    
    overall_avg = float(np.mean(conversation_avgs)) if conversation_avgs else 0.0
    overall_std = float(np.std(conversation_avgs)) if conversation_avgs else 0.0
    turn_avg = float(np.mean(all_similarities)) if all_similarities else 0.0
    turn_std = float(np.std(all_similarities)) if all_similarities else 0.0

    print("\n" + "="*50)
    print(f"EVALUATION RESULTS ({len(conversation_avgs)} conversations)")
    print("="*50)
    print(f"Average conversation similarity: {overall_avg:.4f} ± {overall_std:.4f}")
    print(f"Average turn similarity: {turn_avg:.4f} ± {turn_std:.4f}")
    
    return {
        "conversation_avg": overall_avg,
        "conversation_std": overall_std,
        "turn_avg": turn_avg,
        "turn_std": turn_std
    }


def run_openai_evaluation(num_samples=5, api_key=None):
    """Run the OpenAI evaluation pipeline."""
    try:
        
        dataset = load_therapy_dataset()
        embedder = setup_similarity_model()
        test_set = dataset["test"]
        
        
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                api_key = input("Please enter your OpenAI API key: ")
        
        bot = OpenAITherapyBot(api_key=api_key)
        print(f"Bot initialized with model: {bot.model}")
        
        
        print(f"Starting evaluation on {num_samples} random conversations...")
        results = evaluate_openai_bot(bot, embedder, test_set, num_samples)
        return results
    
    except Exception as e:
        print(f"Error in evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    
    api_key = None  
    run_openai_evaluation(num_samples=240, api_key=api_key)