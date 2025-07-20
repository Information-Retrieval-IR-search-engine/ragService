from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
import faiss
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENROUTER_API_KEY = "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"  # Replace with env var or config in real deployment
model = SentenceTransformer(f'quora_mpnet_v2_tuned_v3')
index = faiss.read_index(f"quora_faiss_index_v3.index")
docs = joblib.load(f"all_quora_doc_texts.joblib")
@app.post("/rag")
def rag_answer_api(query: str = Form(...), dataset: str = Form(...)):
    
    query_embedding = model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_embedding)



    D, I = index.search(query_embedding, 5)
    retrieved_docs = [docs[i] for i in I[0]]

    context = "\n".join(f"- {doc}" for doc in retrieved_docs)

    prompt = f"""You are a helpful assistant. Use only the context below to answer the question.

Context:
{context}

Question: {query}

Instructions:
- Be concise and specific.
- Base your answer only on the provided context.
- Do not repeat phrases or include irrelevant sentences.
- Answer in 1–2 complete sentences.
- If the answer is unknown, say "I don't know based on the context."

Answer:"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "mistralai/mixtral-8x7b-instruct",
        "messages": [{"role": "user", "content": prompt}]
    }

    response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=body)
    answer = response.json()["choices"][0]["message"]["content"].strip()
    return {"answer": answer}
