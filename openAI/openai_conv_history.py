import os
from openai import OpenAI
import tiktoken

class OpenAITherapyBot:
    def __init__(self, api_key=None, model="gpt-4o-mini"):
        """
        Initialize the therapy bot using OpenAI API.
        
        Args:
            api_key: Your OpenAI API key (if None, will use OPENAI_API_KEY env variable)
            model: The OpenAI model to use (default: gpt-3.5-turbo)
        """
        print(f"Initializing OpenAI Therapy Bot with model: {model}")
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key="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")
        self.model = model
        
        # Get the tokenizer for the model
        self.encoding = tiktoken.encoding_for_model(model)
        
        # Initialize conversation history
        self.conversation_history = []
        
        
        self.max_tokens = 1000  
        
        # Set the system prompt for therapeutic context
        self.system_prompt = """You are a compassionate therapeutic assistant.
Your goal is to help users process emotions, provide support, and offer
evidence-based coping strategies. Be empathetic and non-judgmental."""
    
    def count_tokens(self, text):
        """Count the number of tokens in text using OpenAI's tokenizer."""
        return len(self.encoding.encode(text))
    
    def count_message_tokens(self, messages):
        """Count tokens in a list of messages using OpenAI's tokenizer."""
       
        token_count = 0
        for message in messages:
        
            token_count += self.count_tokens(message["content"])
            
            token_count += 4  
        return token_count
    
    def add_message(self, role, content):
        """Add a message to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})
    
    def format_conversation(self):
        """Format the conversation history for OpenAI's API."""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        return messages
    
    def generate_response(self, messages, max_tokens=256):
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
    
    def summarize_conversation_segment(self, segment):
        """Summarize a segment of the conversation, preserving emotional content."""
        # Format the conversation segment for summarization
        conversation_text = ""
        for msg in segment:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_text += f"{role}: {msg['content']}\n\n"
        
        # Create summarization messages for OpenAI
        summary_messages = [
            {"role": "system", "content": """You are an expert at summarizing therapeutic conversations. 
Create a concise summary that preserves all emotional themes, key insights, personal details, 
and coping strategies discussed. The summary should capture the emotional journey 
and therapeutic progress for future reference."""},
            {"role": "user", "content": f"Please summarize this therapy conversation segment, focusing on preserving emotional content:\n\n{conversation_text}"}
        ]
        
        # Generate the summary
        try:
            summary = self.generate_response(summary_messages, max_tokens=256)
            print("\n=== SUMMARY CREATED ===")
            print(summary)
            print("======================\n")
            return summary
        except Exception as e:
            print(f"Error generating summary: {e}")
            return "[Summary of previous conversation focusing on emotional content]"
    
    def manage_context_length(self):
        """Manage context length by summarizing older parts of the conversation."""
        # Get messages including system prompt
        messages = self.format_conversation()
        
        # Check current token count
        token_count = self.count_message_tokens(messages)
        
       
        if token_count < self.max_tokens * 0.8:
            return
        
        print(f"Context length ({token_count} tokens) approaching limit of {self.max_tokens}. Summarizing...")
        
        
        keep_recent = min(4, len(self.conversation_history))
        
     
        if len(self.conversation_history) <= keep_recent + 2:
            return
        
        
        summarize_end = len(self.conversation_history) - keep_recent
        summarize_start = 0
        
        
        segment_to_summarize = self.conversation_history[summarize_start:summarize_end]
        
        
        summary = self.summarize_conversation_segment(segment_to_summarize)
        
      
        summary_message = {
            "role": "assistant",
            "content": f"[CONVERSATION SUMMARY: {summary}]"
        }
        
        
        self.conversation_history = [summary_message] + self.conversation_history[summarize_end:]
    
    def chat(self, user_message):
        """Process a user message and generate a response."""
        
        self.add_message("user", user_message)
        
       
        self.manage_context_length()
        
        
        messages = self.format_conversation()
        
        
        response = self.generate_response(messages)
        
        
        self.add_message("assistant", response)
        
        return response

def run_therapy_chat():
    """Run the therapy chatbot using OpenAI API."""
    
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Please enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
    
    print("Initializing OpenAI Therapy Bot...")
    
    
    bot = OpenAITherapyBot(api_key=api_key)
    
    print("\n" + "="*50)
    print("OpenAI Therapy Bot initialized!")
    print("Type 'exit', 'quit', or 'bye' to end the conversation.")
    print("Type 'debug' to see the current token count.")
    print("="*50 + "\n")
    
    
    print("Therapy Bot: Hello, I'm here to support you. What's on your mind today?")
    
    # Chat loop
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check for exit command
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nTherapy Bot: Take care. I'm here if you need to talk again.")
            break
        
        
        if user_input.lower() == "debug":
            token_count = bot.count_message_tokens(bot.format_conversation())
            print(f"\nCurrent token count: {token_count}/{bot.max_tokens}")
            print(f"Messages in history: {len(bot.conversation_history)}")
            continue
        
        
        try:
            response = bot.chat(user_input)
            print(f"\nTherapy Bot: {response}")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("\nTherapy Bot: I'm sorry, I encountered an issue. Let's continue our conversation.")

if __name__ == "__main__":
    run_therapy_chat()