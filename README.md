# Explainable RAG System for Policy QA

## Overview
This project implements a retrieval-augmented generation (RAG) system that answers questions strictly from provided policy documents. The system is designed to prevent hallucinations, fail safely, and expose source evidence for every answer.

## Why this matters
Large Language Models can produce confident but incorrect answers. In regulated environments such as finance or enterprise SaaS, this creates risk. This project demonstrates how to constrain LLM behavior using retrieval, grounding, and system design rather than prompt tuning alone.

## Key features
- Document-grounded answers using semantic retrieval  
- Safe failure mode (“I don’t know”) when evidence is missing  
- Source attribution for explainability and auditability  
- Minimal disclosure to reduce information leakage  

## Architecture
1. Policy documents are split and embedded  
2. Top-k semantic retrieval selects relevant chunks  
3. Retrieved context is injected into a constrained prompt  
4. The model answers only from provided evidence  

## Example output
Answer:
Remote work must be approved by the employee's manager.

Sources:

policy.txt: "Remote work must be approved by the employee's manager."

policy.txt: "Employees may work remotely up to three days per week."

## Tech stack
- Python  
- Hugging Face Transformers  
- Sentence Transformers  
- FAISS  
- FLAN-T5  

## Design considerations
- The system prioritizes safe failure (“I don’t know”) over speculative answers  
- Retrieval quality is favored over model size to ensure predictable behavior  
- Responses are intentionally minimal to reduce over-disclosure risk
