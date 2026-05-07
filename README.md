# Natural-Language-Processing

# Vietnamese NLP Final Project

This repository contains the final project for the **Natural Language Processing** course.  
The project focuses on building a **Vietnamese question answering system** using **Retrieval-Augmented Generation (RAG)** combined with **fine-tuning** of a large language model.

## Repository Link

GitHub: https://github.com/DarkDipp/Natural-Language-Processing

## Project Requirements

The project follows the assignment requirements:

- Choose one domain-specific Vietnamese knowledge source
- Build a knowledge base from collected documents
- Create at least 300 QA pairs for fine-tuning
- Prepare a manually created test set
- Fine-tune an LLM using **LoRA / QLoRA**
- Build a **RAG pipeline** with chunking, embeddings, and vector retrieval
- Compare 4 configurations:
  - **A**: Base LLM, no RAG
  - **B**: Base LLM, with RAG
  - **C**: Fine-tuned LLM, no RAG
  - **D**: Fine-tuned LLM, with RAG
- Evaluate using:
  - BLEU
  - ROUGE-L
  - BERTScore
  - Recall@5
  - Human evaluation

## Dataset Source

The dataset was built from Vietnamese domain-specific materials, including:

- Self-collected knowledge documents
- Manually written QA pairs
- Public Vietnamese text resources
- Domain-specific reference materials

The dataset is organized into:

- Training set
- Validation set
- Test set

## Project Content

This project includes:

- Data preprocessing
- Data visualization and analysis
- RAG pipeline
- Fine-tuning workflow
- Model comparison
- Evaluation metrics
- Final conclusion and references

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- PEFT
- LoRA / QLoRA
- BitsAndBytes
- ChromaDB / FAISS
- SentenceTransformers
- Google Colab

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/DarkDipp/Natural-Language-Processing.git
cd Natural-Language-Processing
