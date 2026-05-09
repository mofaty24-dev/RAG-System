import faiss
import pickle
from pypdf import PdfReader
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

model = SentenceTransformer("all-MiniLM-L6-v2")
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")


def read_pdf(path):
    reader = PdfReader(path)
    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text


def chunk_text(text,source_name, chunk_size=256, overlap=50 ):
    tokenized_text = tokenizer.encode(text , add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokenized_text):
        end = start + chunk_size
        tokenized_chunk = tokenized_text[start:end]
        chunk = tokenizer.decode(tokenized_chunk , skip_special_tokens=True)
        chunks.append({"text":chunk ,"source_name":source_name})
        start += (chunk_size - overlap)

    return chunks


def document_read(path):
    all_chunks = []

    for file in os.listdir(path):
        if file.endswith(".pdf"):
            all_text = read_pdf(os.path.join(path, file))
            chunks = chunk_text(text=all_text, source_name=file)
            all_chunks.extend(chunks)
    return all_chunks


def embed_chunks(chunks):
    text = [chunk["text"] for chunk in chunks]
    embedding = model.encode(text,convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(embedding)
    return embedding


def build_index(folder_path):
    chunks = document_read(folder_path)
    embeddings = embed_chunks(chunks)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, "company_index.faiss")

    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    return index, chunks
