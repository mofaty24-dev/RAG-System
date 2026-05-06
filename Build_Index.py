import faiss
import pickle
from pypdf import PdfReader
import os
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")


def read_pdf(path):
    reader = PdfReader(path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def chunk_text(text, chunk_size=120, overlap=30):
    words = text.split()
    chunks = []

    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += (chunk_size - overlap)

    return chunks


def document_read(path):
    all_text = ""

    for file in os.listdir(path):
        if file.endswith(".pdf"):
            all_text += read_pdf(os.path.join(path, file))

    return all_text


def embed_chunks(chunks):
    embeddings = model.encode(chunks, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    return embeddings


def build_index(folder_path):
    text = document_read(folder_path)
    chunks = chunk_text(text)
    embeddings = embed_chunks(chunks)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, "company_index.faiss")

    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    return index, chunks