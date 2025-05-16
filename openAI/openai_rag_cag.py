import os
from openai import OpenAI
import tiktoken
import json 
import datetime
import chromadb
from chromadb.utils import embedding_functions
def extract_json_from_text(text):
    """Utility function to extract valid JSON from text that might contain other content."""
    # Look for JSON object
    json_start = text.find('{')
    json_end = text.rfind('}')
    # Look for JSON array if no object found
    if json_start == -1:
        json_start = text.find('[')
        json_end = text.rfind(']')
    # If still no valid markers found
    if json_start == -1 or json_end == -1 or json_end < json_start:
        raise ValueError("No valid JSON found in text")
    # Extract the potential JSON
    json_text = text[json_start:json_end+1].strip()
    # Try to parse it
    try:
        parsed_json = json.loads(json_text)
        return json_text
    except json.JSONDecodeError:
        stack = []
        start_pos = None
        for i, char in enumerate(text):
            if char == '{' and not stack:
                start_pos = i
                stack.append(char)
            elif char == '{':
                stack.append(char)
            elif char == '}' and stack:
                stack.pop()
                if not stack and start_pos is not None:
                    # Try parsing this substring
                    try:
                        candidate = text[start_pos:i+1]
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        # Continue searching
                        pass
        # If we get here, no valid JSON was found
        raise ValueError("Could not extract valid JSON from text")
class OpenAITherapyBot:
    def __init__(self, api_key=None, model="gpt-4o-mini"):
        """
        Initialize the therapy bot using OpenAI API.
        Args:
            api_key: Your OpenAI API key (if None, will use OPENAI_API_KEY env variable)
            model: The OpenAI model to use (default: gpt-3.5-turbo)
        """
        print("Initializing ChromaDB...")
        try:
            self.chroma_path = "./therapy_bot_chromadb"  # Local persistent storage
            self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
            )
            print("✅ ChromaDB initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing ChromaDB: {e}")
        print(f"Initializing OpenAI Therapy Bot with model: {model}")
        
        self.client = OpenAI(api_key="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")
        self.model = model
        self.documentation = self._load_documentation("mistral_cag_knowledge.md")
        
        self.encoding = tiktoken.encoding_for_model(model)
        
        self.conversation_history = []
        self.complete_history = []
        
        self.max_tokens = 100000
        
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
7. Prioritize being accurate over being comprehensive
You are a supportive conversational assistant focused on emotional well-being. While you have guidelines about therapy limitations, substance use, and crisis response, implement these principles naturally without sounding robotic or overly cautious.
IMPORTANT BALANCE PRINCIPLES:
1. Prioritize human connection over rigid rule following
2. Lead with empathy before any limitations
3. Use your judgment about context - understand the spirit of guidelines, not just the letter
4. Frame limitations in terms of what you CAN offer rather than what you cannot
5. Skip unnecessary disclaimers when the context clearly doesn't require them
6. Use warm, natural language even when discussing serious topics
7. Share helpful information about mental health concepts freely
8. Only mention professional help when truly needed, not as a default response
When discussing substance use or sensitive topics, maintain a non-judgmental, harm-reduction approach while still providing factual information about risks. Avoid unnecessarily citing guidelines or apologizing for limitations unless absolutely necessary.
Remember: Your primary purpose is to be helpful and supportive. Guidelines exist to ensure safety, not to limit genuine human connection."""
        self.system_prompt = f"\n\n# THERAPY GUIDELINES\n\n{self.documentation}"
    def _load_documentation(self, filepath):
        """Load therapy documentation from Markdown file."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            print(f"✅ Documentation loaded: {len(content)} characters")
            return content
        except Exception as e:
            print(f"❌ Error loading documentation: {e}")
            return "ERROR: Documentation unavailable"
    def _save_to_chromadb(self, summary_json, collection_name="therapy_sessions"):
        """Internal method to save the session summary to ChromaDB for retrieval."""
        try:
            # Parse the JSON string
            summary_data = json.loads(summary_json)
            # Generate a unique ID for this session
            session_id = f"session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            # Get or create the collection
            collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            # Store the document with metadata
            collection.add(
                documents=[summary_data["summary"]],
                metadatas=[{
                    "emotions": ", ".join(summary_data["emotions"]),
                    "topics": ", ".join(summary_data["topics"]),
                    "user_actions": ", ".join(summary_data["user_actions"]),
                    "assistant_suggestions": ", ".join(summary_data["assistant_suggestions"]),
                    "intensity_score": summary_data["intensity_score"],
                    "timestamp": datetime.datetime.now().isoformat()
                }],
                ids=[session_id]
            )
            print(f"✅ Session saved to ChromaDB with ID: {session_id}")
            return True
        except Exception as e:
            print(f"Error saving to ChromaDB: {e}")
            return False
    def _save_to_local_file(self, summary_json):
        """Internal method to save the session summary to a local JSON file."""
        try:
            # Create directory if it doesn't exist
            os.makedirs("therapy_sessions", exist_ok=True)
            # Generate timestamp-based filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"therapy_sessions/session_{timestamp}.json"
            # Write to file with nice formatting
            with open(filename, "w") as f:
                # Parse and re-serialize to get pretty formatting
                summary_data = json.loads(summary_json)
                json.dump(summary_data, f, indent=2)
            print(f"✅ Session saved to file: {filename}")
            return filename
        except Exception as e:
            print(f"Error saving to local file: {e}")
            return None
    def needs_context(self, message):
        """Determine if the user's message needs context from past sessions."""
        try:
            # Create summarization messages for OpenAI
            context_messages = [
                {"role": "system", "content": """Analyze if this message needs context from past therapy sessions. Be strict in your analysis.
Return a JSON object with:
{
    "needs_context": true/false,
    "context_type": "emotional/progress/coping/topical",
    "reason": "brief explanation",
    "relevance_score": 0-10,  # How relevant is the context needed?
    "should_use_context": true/false  # Only true if relevance_score > 7
}"""},
                {"role": "user", "content": f"""Analyze if this message needs context from past therapy sessions. Be strict.
Message: {message}
Consider:
1. Is there a direct reference to past sessions?
2. Is the context crucial for understanding the current message?
3. Would the response be significantly better with context?
4. Is the user explicitly asking about past interactions?
Only recommend context if it's highly relevant and necessary."""}
            ]
            # Generate the analysis
            analysis_text = self.generate_response(context_messages)
            # Extract JSON from text (since OpenAI might add extra text)
            analysis_json = extract_json_from_text(analysis_text)
            analysis = json.loads(analysis_json)
            return analysis
        except Exception as e:
            print(f"Error analyzing context need: {e}")
            return {"needs_context": False, "context_type": None, "reason": "Error in analysis", "relevance_score": 0, "should_use_context": False}
    def retrieve_relevant_context(self, query, context_type="emotional"):
        """Query ChromaDB for relevant past conversations."""
        try:
            print(f"\n Searching for {context_type} context...")
            # Get the collection
            collection = self.chroma_client.get_collection(
                name="therapy_sessions",
                embedding_function=self.embedding_function
            )
            # Query the collection with higher similarity threshold
            results = collection.query(
                query_texts=[query],
                n_results=2,  # Get top 2 most relevant past sessions
                where={"intensity_score": {"$gte": 3}}  # Only get sessions with intensity >= 3
            )
            if results and 'documents' in results and results['documents'][0]:
                print(f"Found {len(results['documents'][0])} relevant past sessions")
                for i, (doc, metadata_list) in enumerate(zip(results['documents'], results['metadatas'])):
                    print(f"\nSession {i+1}:")
                    print(f"Summary: {doc[:100]}...")
                    if isinstance(metadata_list, list) and metadata_list:
                        metadata = metadata_list[0]
                        print(f"Topics: {metadata.get('topics', 'N/A')}")
                        print(f"Emotions: {metadata.get('emotions', 'N/A')}")
                        print(f"Intensity: {metadata.get('intensity_score', 'N/A')}")
                    else:
                        print("No metadata available")
                return results
            else:
                print(" No relevant past sessions found")
                return None
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return None
    def format_context(self, results):
        """Format retrieved context for inclusion in the prompt."""
        if not results or 'documents' not in results or not results['documents'][0]:
            return ""
        context = "\nRelevant past sessions:\n"
        for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            context += f"\nSession {i+1}:\n"
            context += f"Summary: {doc}\n"
            context += f"Topics: {metadata.get('topics', 'N/A')}\n"
            context += f"Emotions: {metadata.get('emotions', 'N/A')}\n"
            context += f"Intensity: {metadata.get('intensity_score', 'N/A')}\n"
        return context
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
        if not (role == "assistant" and "[CONVERSATION SUMMARY:" in content):
            self.complete_history.append({"role": role, "content": content})
    def format_conversation(self):
        """Format the conversation history for OpenAI's API."""
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.conversation_history)
        return messages
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
            {"role": "system", "content": """ You are a specialized AI that creates detailed summaries of therapy conversations for a retrieval system.
Review this conversation and provide a summary with: 
             Note: Output must be in strict JSON format
1. A comprehensive summary capturing all key issues, insights, and progress. Scale the length based on conversation complexity - longer and more detailed for complex conversations, briefer for simple ones.
2. Key emotions expressed by the user (list all relevant emotions)
3. Main topics discussed (list all important topics)
4. User actions during the conversation (list all significant actions)
5. Therapeutic suggestions or techniques offered (list all suggestions made)
6. An intensity score from 1-10 that reflects the emotional intensity of the conversation, where:
   - 1-3: Mild emotional intensity, mostly informational
   - 4-6: Moderate emotional engagement, some vulnerability
   - 7-8: High emotional disclosure, significant distress
   - 9-10: Crisis-level intensity, acute distress
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
            
            try:
            
                self._save_to_chromadb(summary_json)
                self._save_to_local_file(summary_json)
            except Exception as e:
                print(f"Error saving summary: {e}")
            return summary_json
        except Exception as e:
            print(f"Error generating final summary: {e}")
            return "{\"summary\": \"Error generating summary\", \"emotions\": [], \"topics\": [], \"user_actions\": [], \"assistant_suggestions\": [], \"intensity_score\": 1}"
    
    def chat(self, user_message):
        """Process a user message and generate a response."""
        # Check if adding this message would exceed capacity
        if self.is_context_exceeding_capacity(user_message):
          return "SESSION_END: I apologize, but our conversation has reached its length limit. Let's start a new chat to continue."
        # NEW: Check if we need context from past sessions
        try:
          context_analysis = self.needs_context(user_message)
          # Save original system prompt to restore later
          original_system_prompt = self.system_prompt
          # Only retrieve and use context if needed and score is high enough
          if context_analysis.get("needs_context", False) and context_analysis.get("should_use_context", False):
            print(f"Retrieving context: {context_analysis.get('reason')}")
            retrieved_results = self.retrieve_relevant_context(user_message)
            if retrieved_results:
              context = self.format_context(retrieved_results)
              # Temporarily augment system prompt with context
              self.system_prompt += f"""
CRITICAL: When the user asks if you remember them or references past conversations, DO NOT say you can't remember. 
Instead, reference the information from past sessions naturally as if you remember the conversations.
PAST SESSION CONTEXT (Reference only if directly relevant):
{context}
Guidelines:
1. Use context only when directly relevant to the user's question
2. Show familiarity with previously discussed topics if mentioned or asked
3. Focus primarily on current conversation"""
        except Exception as e:
          print(f"Error handling context retrieval: {e}")
          
        
        self.add_message("user", user_message)
        
        if not self.manage_context_length():
          # Restore original system prompt before returning
          if 'original_system_prompt' in locals():
            self.system_prompt = original_system_prompt
          return "SESSION_END: I apologize, but our conversation has reached its length limit. Let's start a new chat to continue."
        
        messages = self.format_conversation()
        
        response = self.generate_response(messages)
        
        if 'original_system_prompt' in locals():
          self.system_prompt = original_system_prompt
        
        if self.is_context_exceeding_capacity(response):
          self.add_message("assistant", "I understand your message.")
          return "SESSION_END: I apologize, but our conversation has reached its length limit. Let's start a new chat to continue."
        # Add response to history (existing code)
        self.add_message("assistant", response)
        if self.is_context_exceeding_capacity():
          return f"{response}\n\nSESSION_END: I apologize, but we've reached the conversation limit. Let's start a new chat to continue."
        return response
def run_therapy_chat():
    """Run the therapy chatbot using OpenAI API."""
    # Check for API key
    api_key ="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"
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