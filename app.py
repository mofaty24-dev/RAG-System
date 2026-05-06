from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rag_engine import RAGEngine

app = FastAPI()
engine = RAGEngine()


class Question(BaseModel):
    question: str


@app.get("/", response_class=HTMLResponse)
def home():
    with open("Simple UI.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/ask")
def ask_question(q: Question):
    result = engine.generate_answer(q.question)
    return result