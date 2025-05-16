import os
from openai import OpenAI
import tiktoken
import json 

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
        self.complete_history = []
        
        
        self.max_tokens = 1000 # Intentionally small for testing summarization
        
        # Set the system prompt for therapeutic context
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
    
    def count_tokens(self, text):
        """Count the number of tokens in text using OpenAI's tokenizer."""
        return len(self.encoding.encode(text))
    
    def count_message_tokens(self, messages):
        """Count tokens in a list of messages using OpenAI's tokenizer."""
        # This is a simplified token counter - OpenAI's actual counting has some overhead
        token_count = 0
        for message in messages:
            # Count tokens in content
            token_count += self.count_tokens(message["content"])
            # Add overhead for message formatting (approximation)
            token_count += 4  # ~4 tokens per message for role formatting
        return token_count
    
    def add_message(self, role, content):
        """Add a message to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})
        if not (role == "assistant" and "[CONVERSATION SUMMARY:" in content):
            self.complete_history.append({"role": role, "content": content})
    
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
and therapeutic progress for future reference.Points to note are :
1. ONLY include information EXPLICITLY stated in the conversation text
2. NEVER add details, interpretations, or assumptions not in the original text
3. DO NOT infer emotions or thoughts unless directly expressed by the user
4. Use NEUTRAL language that accurately reflects what was discussed
5. Focus primarily on what the user shared, their concerns and needs
6. Remember: Create a true narrative SUMMARY, not a condensed transcript.
7. If something is unclear in the original text, reflect that ambiguity
8. Create a concise but factually complete record of the conversation"""},
            {"role": "user", "content": f"Please summarize this therapy conversation segment, focusing on preserving emotional content:\n\n{conversation_text}"}
        ]
        
        # Generate the summary
        try:
            summary = self.generate_response(summary_messages, max_tokens=10000)
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
        
        # If we're within the limits, do nothing
        if token_count < self.max_tokens * 0.8:
            return True
        
        print(f"Context length ({token_count} tokens) approaching limit of {self.max_tokens}. Summarizing...")
        
        # Keep the most recent 4 messages (2 exchanges)
        keep_recent = min(4, len(self.conversation_history))
        
        # If we don't have enough messages to summarize yet, do nothing
        if len(self.conversation_history) <= keep_recent + 2:
            return True
        
        # Determine which segment to summarize
        #summarize_end = len(self.conversation_history) - keep_recent
        summarize_start = 0
        for i, msg in enumerate(self.conversation_history):
            if msg["role"] == "assistant" and "[CONVERSATION SUMMARY:" in msg["content"]:
            # Start summarizing after this summary message
                summarize_start = i + 1
        summarize_end = len(self.conversation_history) - keep_recent
        if summarize_start >= summarize_end:
            return True  
        # Extract segment to summarize
        segment_to_summarize = self.conversation_history[summarize_start:summarize_end]
        
        # Generate summary
        summary = self.summarize_conversation_segment(segment_to_summarize)
        
        # Replace summarized messages with the summary
        summary_message = {
            "role": "assistant",
            "content": f"[CONVERSATION SUMMARY: {summary}]"
        }
        
        # Reconstruct conversation history
        self.conversation_history = self.conversation_history[:summarize_start] +[summary_message] + self.conversation_history[summarize_end:]
        return True 
    
    def is_context_exceeding_capacity(self, additional_text=""):
        """Check if adding additional text would exceed 95% of capacity."""
        # Create test messages
        test_messages = self.format_conversation()
        
        # Add hypothetical new message if provided
        if additional_text:
            test_messages.append({"role": "user", "content": additional_text})
        
        # Count tokens
        token_count = self.count_message_tokens(test_messages)
        
        # Return True if we would exceed 95% capacity
        return token_count > self.max_tokens * 0.95
    
    def generate_final_summary(self):
        """Generate a comprehensive summary with metadata for the entire session."""
        # Format the complete conversation history
        conversation_text = ""
        for msg in self.complete_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            conversation_text += f"{role}: {msg['content']}\n\n"
        
        # Create system prompt for comprehensive summarization
        summary_messages = [
            {"role": "system", "content": """You are a specialized AI that creates detailed summaries of therapy conversations.
Review this conversation and provide a summary with metadata in STRICT JSON format.

Your output MUST be valid, parseable JSON with this EXACT structure:
{
  "summary": "A paragraph summarizing the key points of the conversation",
  "emotions": ["emotion1", "emotion2"],
  "topics": ["topic1", "topic2"],
  "user_actions": ["action1", "action2"],
  "assistant_suggestions": ["suggestion1", "suggestion2"],
  "intensity_score": 5
}

CRITICAL REQUIREMENTS:
1. EVERY field in "assistant_suggestions" must be a simple string like "Try breathing exercises"
2. NEVER use objects, nested arrays, _note fields, or any complex structures in any array
3. ALL arrays must contain ONLY simple string elements
4. "intensity_score" must be a number (1-10) without quotes
5. Output MUST be ONLY valid JSON (no markdown, no explanations)
6. ALL strings must use double quotes, never single quotes
7. Keep all suggestions as SHORT, CLEAR phrases
}"""}, 
            {"role": "user", "content": f"Create a complete summary with metadata for this entire therapy session:\n\n{conversation_text}"}
        ]
        
        # Generate the final summary
        try:
            summary_json = self.generate_response(summary_messages, max_tokens=10024)
            print(summary_json)
            # Parse JSON to ensure it's valid
            summary_json = summary_json.strip()
            if summary_json.startswith("```json"):
                summary_json = summary_json.replace("```json", "", 1)
            if summary_json.endswith("```"):
                summary_json = summary_json.rsplit("```", 1)[0]
            summary_json = summary_json.strip()
            parsed_json = json.loads(summary_json)
            
            # Re-serialize with pretty formatting
            formatted_json = json.dumps(parsed_json, indent=2)
            
            return formatted_json
        except Exception as e:
            print(f"Error generating final summary: {e}")
            return "{\"summary\": \"Error generating summary\", \"emotions\": [], \"topics\": [], \"user_actions\": [], \"assistant_suggestions\": [], \"intensity_score\": 1}"

    
    def chat(self, user_message):
        """Process a user message and generate a response."""

        if self.is_context_exceeding_capacity(user_message):
            return "SESSION_END: I apologize, but our conversation has reached its length limit. Let's start a new chat to continue."
        
        # Add user message to history
        self.add_message("user", user_message)
        
        # Manage context length
        if not self.manage_context_length():
            return "SESSION_END: I apologize, but our conversation has reached its length limit. Let's start a new chat to continue."
        
        # Format the full conversation
        messages = self.format_conversation()
        
        # Generate response
        response = self.generate_response(messages)
        
        if self.is_context_exceeding_capacity(response):
            # Add shortened response before ending
            self.add_message("assistant", "I understand your message.")
            return "SESSION_END: I apologize, but our conversation has reached its length limit. Let's start a new chat to continue."
        
        # Add response to history
        self.add_message("assistant", response)
        if self.is_context_exceeding_capacity():
            return f"{response}\n\nSESSION_END: I apologize, but we've reached the conversation limit. Let's start a new chat to continue."
    
        
        return response

def run_therapy_chat():
    """Run the therapy chatbot using OpenAI API."""
    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Please enter your OpenAI API key: ")
        os.environ["OPENAI_API_KEY"] = api_key
    
    print("Initializing OpenAI Therapy Bot...")
    
    # Initialize the bot
    bot = OpenAITherapyBot(api_key=api_key)
    
    print("\n" + "="*50)
    print("OpenAI Therapy Bot initialized!")
    print("Type 'exit', 'quit', or 'bye' to end the conversation.")
    print("Type 'debug' to see the current token count.")
    print("="*50 + "\n")
    
    # Initial greeting
    print("Therapy Bot: Hello, I'm here to support you. What's on your mind today?")
    
    # Chat loop
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check for exit command
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nTherapy Bot: Take care. I'm here if you need to talk again.")
            print("\nGenerating final session summary...")
            final_summary = bot.generate_final_summary()
            print("\nFinal Session Summary:")
            print(final_summary)
            break
        
        # Check for debug command
        if user_input.lower() == "debug":
            token_count = bot.count_message_tokens(bot.format_conversation())
            print(f"\nCurrent token count: {token_count}/{bot.max_tokens}")
            print(f"Messages in history: {len(bot.conversation_history)}")
            print(f"Messages in complete history: {len(bot.complete_history)}")
            continue
        
        # Process message
        try:
            response = bot.chat(user_input)
            if response.startswith("SESSION_END:"):
                # Extract the actual message to display
                display_message = response.replace("SESSION_END:", "").strip()
                print(f"\nTherapy Bot: {display_message}")
                print("\nTherapy Bot: Session ended due to context length limitations.")
                
                # Generate and print final summary
                print("\nGenerating final session summary...")
                final_summary = bot.generate_final_summary()
                print("\nFinal Session Summary:")
                print(final_summary)
                
                break
            else:
                print(f"\nTherapy Bot: {response}")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("\nTherapy Bot: I'm sorry, I encountered an issue. Let's continue our conversation.")

if __name__ == "__main__":
    run_therapy_chat()