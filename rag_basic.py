from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load policy document
with open("policy.txt", "r") as f:
    policy_text = f.read()

chunks = [
    line.strip()
    for line in policy_text.split("\n")
    if line.strip()
]

# Embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedder.encode(chunks)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Question
question = "What is the company policy on remote work?"

# Retrieve top-k relevant chunks
question_embedding = embedder.encode([question])
scores, indices = index.search(np.array(question_embedding), k=3)

retrieved_chunks = [chunks[i] for i in indices[0]]
context = "\n".join(retrieved_chunks)

# Load FLAN-T5 properly
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def answer_question(question, context):
    prompt = f"""
Answer the question using ONLY the information below.
If the answer is not in the text, say "I don't know".

Context:
{context}

Question:
{question}
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Generate answer
answer = answer_question(question, context)

print("Answer:")
print(answer)

print("\nSources:")
for chunk in retrieved_chunks:
    print(f"- policy.txt: \"{chunk}\"")
