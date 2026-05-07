import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from Build_Index import build_index
import torch


class RAGEngine:
    def __init__(self):

        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Loading vector database if it was created previously 
        if os.path.exists("company_index.faiss"):
            self.index = faiss.read_index("company_index.faiss")
            with open("chunks.pkl", "rb") as f:
                self.chunks = pickle.load(f)
        else:
            index_path = os.getenv("INDEX_DIR")
            if not index_path:
                raise ValueError("INDEX_DIR missing")
            self.index, self.chunks = build_index(index_path)

        # Building text Generator
        device = 0 if torch.cuda.is_available() else -1
        self.generator = pipeline(
            "text-generation",
            model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            device=device
        )
     
    def embed_query(self, text):
        embedding = self.model.encode([text], convert_to_numpy=True)
        faiss.normalize_L2(embedding)
        return embedding

    def search(self, query, top_k=3):
        query_embedding = self.embed_query(query)
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i in indices[0]:
            if i != -1:
                results.append(self.chunks[i])

        return results

    def generate_answer(self, query):
        retrieved_chunks = self.search(query)

        
        if not retrieved_chunks:
            return {
                "answer": "I don't have enough information.",
                "sources": []
            }

        context = "\n\n".join(retrieved_chunks)

        prompt = f"""
You are a strict AI assistant.

Rules:
- Use ONLY the provided context.
- If answer is not in context, say: "I don't have enough information."
- Do NOT guess.
- Answer in 2 sentences max.
- No repetition.

Context:
{context}

Question:
{query}

Answer:
"""

        result = self.generator(
            prompt,
            max_new_tokens=60,
            do_sample=True,
            temperature=0.7,
            repetition_penalty=1.2
        )

        answer = result[0]["generated_text"]

        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1]

        return {
            "answer": answer.strip(),
            "sources": retrieved_chunks
        }
