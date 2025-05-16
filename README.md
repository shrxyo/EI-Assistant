# Emotional Intelligence Assistant using a Hybrid Augmented Generation Framework

## Key Components

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
