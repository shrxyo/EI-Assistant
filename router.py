import subprocess

def classify_intent_llama(user_input: str) -> str:
    system_prompt = """You are an intent classifier for a mental health assistant.
Classify the user's message into one of these categories:
- emotional: A message expressing emotional states, like sadness or stress.
- factual: A message requesting specific, actionable advice or techniques (e.g., how to manage panic attacks, grounding techniques, etc.).
- casual: A message with informal conversation or small talk.
- both: A message that includes both emotional and factual aspects.

Just reply with one word: emotional, factual, casual, or both."""

    full_prompt = f"{system_prompt}\n\nUser: {user_input}\nIntent:"

    try:
        result = subprocess.check_output(
            ["ollama", "run", "llama3", full_prompt],
            text=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Error in subprocess: {e}")
        return "unknown"

    reply = result.strip().lower()
    valid_labels = ["emotional", "factual", "casual", "both"]

    for label in valid_labels:
        if label in reply:
            return label

    return "unknown"


def route_input(user_input: str) -> str:
    # Classify the user's input into intent
    intent = classify_intent_llama(user_input)
    
    # If the intent is factual, route to CAG
    if intent == "factual":
        return "CAG"  
    
    # For anything else (emotional, casual, both), route to LLM
    return "LLM"  

# Main function to take user input and route it
def main():
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        else:
            # Get the route (CAG or LLM) based on user input
            module = route_input(user_input)
            print(f"Routing to: {module}")

if __name__ == "__main__":
    main()
