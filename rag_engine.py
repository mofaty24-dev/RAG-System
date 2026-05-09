import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from Build_Index import build_index
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class RAGEngine:
    def __init__(self):

        # embedding model
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # load or build index
        if os.path.exists("company_index.faiss"):
            self.index = faiss.read_index("company_index.faiss")

            with open("chunks.pkl", "rb") as f:
                self.chunks = pickle.load(f)

        else:
            index_path = os.getenv("INDEX_DIR")

            if not index_path:
                raise ValueError("INDEX_DIR missing")

            self.index, self.chunks = build_index(index_path)

        # Azure OpenAI client
        self.client = OpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            base_url=f"{os.getenv('AZURE_OPENAI_ENDPOINT')}/openai/v1/"
        )

        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")

    def embed_query(self, text):
        embedding = self.model.encode([text], convert_to_numpy=True)
        faiss.normalize_L2(embedding)
        return embedding

    def search(self, query, top_k=3):

        query_embedding = self.embed_query(query)

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        chunks_text = [chunk["text"] for chunk in self.chunks]
        chunks_file = [chunk["source_name"] for chunk in self.chunks]

        for i in indices[0]:
            if i != -1:
                results.append({"text": chunks_text[i], "source":chunks_file[i]})

        return results

    def generate_answer(self, query):

        retrieved_chunks = self.search(query)

        if not retrieved_chunks:
            return {
                "answer": "I don't have enough information.",
                "sources": []
        }
        context = "\n\n".join(f"[{chunk['source']}]: {chunk['text']}" for chunk in retrieved_chunks)

        prompt = f"""
        You are a strict AI assistant.

        Rules:
        - Use ONLY the provided context.
        - If answer is not in context, say: "I don't have enough information."
        - Do NOT guess.
        - Keep answers concise.
        - No repetition.
        

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

        response = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict RAG assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.2,
            max_tokens=120
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources": retrieved_chunks
        }
