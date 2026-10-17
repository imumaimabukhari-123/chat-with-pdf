"""
Chat with your PDF: upload a PDF and ask it questions (a small RAG system).

How it works:
  1. The browser reads the PDF (pdf.js) and splits the text into chunks,
     remembering the page number of each chunk.
  2. /api/embed turns the chunks into embeddings (Gemini, free tier).
     The browser keeps them in memory, so nothing is stored on the server.
  3. For each question the browser finds the best chunks with hybrid search
     (vector similarity + keyword search) and sends them to /api/ask.
  4. /api/ask asks Gemini to answer ONLY from those chunks and cite pages.

Run:   python app.py
Open:  http://127.0.0.1:8002

API:
  POST /api/embed   {texts: [...], task: "document" | "query"}  -> {vectors: [[...]]}
  POST /api/ask     {question, sources: [{page, text}], history} -> {answer}
"""
import os
import time
from collections import defaultdict, deque
from typing import Literal

import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
PORT = int(os.getenv("PORT", "8002"))
CHAT_MODEL = "gemini-flash-lite-latest"
EMBED_MODEL = "gemini-embedding-001"
EMBED_DIMENSIONS = 768
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

SYSTEM_PROMPT = """You answer questions about a document the user uploaded.

Rules:
- Use ONLY the sources below. Never use outside knowledge or guess.
- After each fact, cite the page number shown above the passage it came from, like this: [p. 4]
- If the sources don't contain the answer, say exactly:
  "I couldn't find that in this document." and nothing else.
- Be clear and short. Use bullet points for lists. Use **bold** for key terms.
- Reply in the same language as the question."""

app = FastAPI(title="Chat with your PDF")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ---------------- rate limits (keeps the free Gemini quota safe) ----------------

LIMITS = {"embed": (80, 3600), "ask": (40, 3600)}  # (requests, per seconds) per visitor
_hits: dict = defaultdict(deque)


def check_rate_limit(request: Request, kind: str) -> None:
    max_hits, window = LIMITS[kind]
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?")
    hits = _hits[(kind, ip.split(",")[0].strip())]
    now = time.time()
    while hits and now - hits[0] > window:
        hits.popleft()
    if len(hits) >= max_hits:
        raise HTTPException(429, "You've reached the hourly limit of this free demo. Please try again later.")
    hits.append(now)


# ---------------- Gemini calls ----------------

def gemini_post(url: str, payload: dict, timeout: int = 45) -> dict:
    """POST to Gemini, waiting and retrying when the free quota is busy (429)."""
    if not GEMINI_API_KEY:
        raise HTTPException(500, "GEMINI_API_KEY is missing. Add it to the .env file.")
    # The OpenAI-compatible chat endpoint wants a Bearer token; the native endpoints want x-goog-api-key.
    auth = {"Authorization": f"Bearer {GEMINI_API_KEY}"} if "/openai/" in url else {"x-goog-api-key": GEMINI_API_KEY}
    headers = {**auth, "Content-Type": "application/json"}
    for wait in (2, 5, 10, 0):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        except requests.RequestException:
            raise HTTPException(502, "Couldn't reach the AI service. Please try again.")
        if resp.status_code == 429 and wait:
            time.sleep(wait)
            continue
        if resp.status_code == 429:
            raise HTTPException(429, "The free AI quota is busy right now. Please wait a minute and try again.")
        if not resp.ok:
            raise HTTPException(502, f"The AI service returned an error ({resp.status_code}).")
        return resp.json()


def embed(texts: list[str], task: str) -> list[list[float]]:
    task_type = "RETRIEVAL_QUERY" if task == "query" else "RETRIEVAL_DOCUMENT"
    data = gemini_post(
        f"{GEMINI_BASE}/models/{EMBED_MODEL}:batchEmbedContents",
        {"requests": [
            {
                "model": f"models/{EMBED_MODEL}",
                "content": {"parts": [{"text": t}]},
                "taskType": task_type,
                "outputDimensionality": EMBED_DIMENSIONS,
            }
            for t in texts
        ]},
    )
    # Round the numbers to keep the response small; it doesn't change the search results.
    return [[round(v, 5) for v in e["values"]] for e in data["embeddings"]]


def answer(question: str, sources: list["Source"], history: list["Turn"]) -> str:
    # Label passages by page only: numbering them "Source 1, 2…" made the AI cite those numbers as pages.
    passages = "\n\n".join(f"--- Page {s.page} ---\n{s.text}" for s in sources)
    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}\n\nSources:\n\n{passages}"}]
    messages += [{"role": t.role, "content": t.content} for t in history]
    messages.append({"role": "user", "content": question})
    data = gemini_post(
        f"{GEMINI_BASE}/openai/chat/completions",
        {"model": CHAT_MODEL, "messages": messages, "temperature": 0.2},
    )
    return (data["choices"][0]["message"]["content"] or "").strip()


# ---------------- website + API ----------------

class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=100)
    task: Literal["document", "query"] = "document"


class Source(BaseModel):
    page: int
    text: str = Field(max_length=4000)


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    sources: list[Source] = Field(max_length=8)
    history: list[Turn] = Field(default=[], max_length=6)


@app.get("/")
def home():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/api/embed")
def api_embed(body: EmbedRequest, request: Request):
    check_rate_limit(request, "embed")
    texts = [t[:4000] for t in body.texts]
    return {"vectors": embed(texts, body.task)}


@app.post("/api/ask")
def api_ask(body: AskRequest, request: Request):
    check_rate_limit(request, "ask")
    if not body.sources:
        return {"answer": "I couldn't find that in this document."}
    return {"answer": answer(body.question, body.sources, body.history)}


if __name__ == "__main__":
    print(f"Open http://127.0.0.1:{PORT}")
    uvicorn.run(app, host="127.0.0.1", port=PORT)
