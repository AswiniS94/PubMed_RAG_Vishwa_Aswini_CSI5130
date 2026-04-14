# Domain-Specialized Embedding Learning for Medical Retrieval-Augmented Generation (RAG)

## Team members
Vishwa Gosavi and Aswini Sivakumar

### For grading:
[Video of presentation](https://drive.google.com/file/d/16lDLcY5c6GlzpO0cJrL6Mm0MHxACUi4N/view?usp=sharing)

The following link has the notebook with outputs:
- [Finetune-MiniLM](https://colab.research.google.com/drive/1FvtCNoNvAgxeN23G68hrW7iGLrPzsF-8#scrollTo=YfCSVPQhDY3J)
- [RAG](https://colab.research.google.com/drive/13iWNEzLZxeeGRNARqIbaSWxyrBWK13gm)

## 1. Introduction
Large Language Models (LLMs) have demonstrated strong performance in natural language understanding and generation tasks. However, when applied to high-stakes domains such as medicine, these models often produce hallucinated or factually incorrect responses. This limitation poses significant risks in biomedical question answering, where accuracy, evidence grounding, and citation reliability are critical.

Retrieval-Augmented Generation (RAG) has emerged as a practical approach to mitigate hallucination by conditioning model outputs on retrieved external knowledge. In medical applications, RAG systems typically retrieve biomedical literature (e.g., PubMed abstracts) and provide context to an LLM before generating an answer. While this improves factual grounding, the effectiveness of RAG is highly dependent on the quality of the retrieval component.

In turn, the quality of retrieval depends mainly on the embedding model used. Generic embedding models are trained on broad web corpora and are not optimized for biomedical terminology, domain-specific abbreviations, or clinical semantic relationships.

This project investigates whether domain-specialized embedding learning can significantly improve retrieval accuracy and answer faithfulness in a medical RAG system. Specifically, we evaluate multiple embedding training strategies, including generic embeddings, biomedical pre-trained models, and contrastively fine-tuned domain embeddings within a controlled RAG architecture. The goal is to quantify how embedding specialization impacts retrieval metrics and downstream question answering performance.


## 2. Research Question
**Can domain-specific embedding training improve retrieval quality and reduce hallucinations in medical Retrieval-Augmented Generation (RAG) systems?**


## 3.Problem Statement
Large Language Models (LLMs) do not have up-to-date knowledge and often produce hallucinated or inaccurate answers in the medical domain. Retrieval-Augmented Generation (RAG) can mitigate this by grounding answers in external knowledge, but the retrieval quality depends heavily on the embedding model. Generic embeddings are not optimized for biomedical terminology, which can reduce retrieval relevance and answer faithfulness.


## 4. Project Objective
This project aims to design and evaluate a medical RAG system grounded in PubMed abstracts and systematically compare two embedding training strategies:
1. A baseline embedding model (general-purpose sentence embedding).
2. A domain-specialized embedding model trained or fine-tuned on biomedical literature.

The primary objective is to measure whether domain adaptation in the embedding layer leads to:
- Improved retrieval relevance (Recall@K, MRR@10)
- Higher citation alignment with retrieved documents
- Improved factual consistency in generated answers (Yes/No/Maybe accuracy)
- Reduced hallucination rate in medical question answering
By isolating the embedding model as the key experimental variable while keeping the retrieval database, language model, and prompting strategy constant, this study evaluates the direct impact of domain-specific embedding learning on end-to-end RAG performance.


## 5. System Architecture
### Proposed Method:
We will build a medical RAG system with the following pipeline:
![System Diagram](RAG_pipeline_image.png)

### System Pipeline

1. PubMed abstracts are embedded using the embedding model.
2. Embeddings and metadata are stored dynamically in the Weaviate cloud vector database.
3. User queries are embedded and used to retrieve the most relevant documents via Hybrid search.
4. Retrieved biomedical evidence is passed to the LLM via LangChain.
5. The LLM generates a strictly constrained, evidence-grounded medical answer (Yes/No/Maybe).

## 6. Dataset, Models, and Tools

This project leverages open-source datasets, embedding models, language models, and vector databases to build the medical RAG system.

### Dataset

| Dataset | Split Used | Description | Source |
|-------|------------|-------------|--------|
| PubMedQA | pqa_unlabeled | Sampled 10,000 records to extract question-context pairs for **Contrastive Fine-tuning** | [qiaojin/PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) |
| PubMedQA | pqa_labeled | Used to populate the Vector DB and evaluate the end-to-end RAG pipeline (1000 expert-labeled questions) | [qiaojin/PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA) |

### Embedding Model

| Model | Description | Source |
|------|-------------|--------|
| all-MiniLM-L12-v2 | Baseline general-purpose sentence embedding models. | [sentence-transformers/all-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2) |
| mini-pubmedqa-finetuned | Our **domain-specialized model**, fine-tuned via contrastive learning (InfoNCE) on 10,000 PubMedQA pairs. | [AswiniSivakumar/mini-pubmedqa-finetuned](https://huggingface.co/AswiniSivakumar/mini-pubmedqa-finetuned) |

This *all-MiniLM-L12-v2* model will be used as:
- **Baseline embedding model**
- **Fine-tuning starting point for domain-specialized embeddings**

### Language Model

The model will be used to interpret retrieved contexts and strictly generate "Yes", "No", or "Maybe" answers for the user query:

| Model | Description | Source |
|------|-------------|--------|
| Llama-3.2-1B-Instruct | Lightweight instruction-tuned LLM optimal for Colab T4 inference. | [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) |


### Infrastructure and Tools

| Tool | Purpose | 
|-----|---------|
| Weaviate Cloud | Vector database for index biomedical documents and perform semantic search over embeddings. |
| SentenceTransformers | Framework used for the *MultipleNegativesRankingLoss* contrastive fine-tuning.|
| LangChain | Framework to construct the RAG retrieval chain, text-splitters, and prompting logic.| 
| Gradio | Interactive user interface allowing users to ask queries, select models dynamically, and view retrieved evidence.|
| Google Colab | Cloud-based Jupyter notebook environment to perform model fine-tuning and LLM inference using *T4 GPU runtime* |

### Implementation and Evaluation Details

#### **Fine-Tuning Phase:**
- **Objective:** Teach a base MiniLM model to map medical questions closer to their corresponding medical abstracts in vector space
- **Methodology:** Used *MultipleNegativesRankingLoss* with a batch size of 32 (utilizing in-batch negatives)
- **Result:** Slight improvement in *Accuracy@1* and *MRR@10* over the generic pre-trained baseline.

![System Diagram](Finetune_process_image.png)

#### **RAG Construction Phase:**
- **Database:** Implemented dynamic weaviate collections based on the selected embedding model.
- **LLM Optimization:** Deployed models using HuggingFace pipelines with *torch.float16* and *Scaled Dot Product Attention (SDPA)* for optimal Google Colab T4 compatibility.

#### **Evaluation Strategy:**
- **Embedding Evaluation:** Calculated *Recall@5*, *Recall@10*, and *MRR@10* using *InformationRetreivalEvaluator* on the test split.
- **End-to-End RAG Evaluation:** Batched inference over the 1000 questions in *pqa_labed* dataset, using *regex* cleaning to compare the generated answers against the ground-truth labels and calculate category-specific (Yes/No/Maybe) accuracy percentages.

