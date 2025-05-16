# Emotional Intelligence Therapy Chatbot  
**A hybrid RAG + CAG framework for emotionally supportive multi-turn conversations**

## Project Summary
This project investigates whether combining **Retrieval-Augmented Generation (RAG)** and **Cache-Augmented Generation (CAG)** can improve emotional intelligence, personalization, and memory in therapy-style chatbot systems. We build and compare two architectures:
- **GPT-4o-mini (finetuned on a therapy dataset using OpenAI API)**
- **LLaMA 3.2 1B Instruct (unfinetuned open-source model)**


## Dataset
- Used the [PHR Mental Therapy Conversational Dataset](https://huggingface.co/datasets/vibhorag101/phr-mental-therapy-dataset-conversational-format)
- Multi-turn dialogues (~84k total)
- We sampled **1,200 conversations** (80% train / 20% test)
- Conversations were cleaned, normalized, and formatted as JSONL


## Architecture Overview
- **CAG**: Preloads therapeutic knowledge such as CBT practices, suicide protocols, and disclaimers
  - GPT-4o: Uses static prompt injection ("Poor Man’s Cache")
  - LLaMA: Uses true KV cache preloading
- **RAG**: Retrieves previous conversation summaries from **ChromaDB** using sentence embeddings


## Evaluation
### Semantic Similarity
- Measured cosine similarity between generated and gold responses using MiniLM sentence embeddings
- GPT-4o-mini (finetuned) improved average similarity compared to baseline.

### Human Evaluation
- Domain expert rated outputs on empathy, coherence, and relevance.
- CAG + RAG models had a comparitively better performance in emotional continuity.


## Code
**🧠Fine-tuning (GPT-4o)**

- `data_preparation.py`: Prepares datasets for training.
- `fileupload.py`: Handles file uploads to OpenAI API.
- `ft_job.py`: Launches fine-tuning jobs.
- `ft_status.py`: Checks job status.
- `ft_test.py`: Tests the fine-tuned models.

**📊Evaluation**

- `baselinegpt_eval.py`: Evaluates base GPT model through semantic similarity.
- `finetunedgpt_eval.py`: Evaluates fine-tuned model through semantic similarity.

**🦙LLaMA Integration**

- `Copy_of_FINAL_LLAMAcode.ipynb`: Notebook for running LLaMA-based models including baseline model as well as model with RAG and CAG integration.
- `mistral_cag_knowledge.md`: Knowledge base for the CAG module.
- `mistral_cag_knowledge_small (2)_llama_1b_kv_cache.pt`: Values of KV cache for the knowledge base.

**🤖OpenAI**

- `openai_conv_history.py`: Tracks conversational interactions using our finutuned model GPT 4o-mini.
- `openai_rag_cag.py`: RAG pipeline with CAG documents.
- `openai_with_CAG_and_chroma_storage.py`: Enhanced CAG with vector storage using ChromaDB.
- `mistral_cag_knowledge.md`: Knowledge base for CAG implementation
