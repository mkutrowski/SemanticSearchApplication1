import io
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer
import fitz
import docx




app = FastAPI(title="Semantic Search Application 1")





EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    metric: str = "cosine"

class SearchResultChunk(BaseModel):
    chunk: str
    score: float
    index: int

class SearchResponse(BaseModel):
    results: List[SearchResultChunk]

# Stan w pamięci (na potrzeby demo – po restarcie znika)
current_chunks: List[str] = []
current_embeddings: Optional[np.ndarray] = None
current_filename: Optional[str] = None

# ====== FUNKCJE POMOCNICZE ======

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text.append(page.get_text())
    return "\n".join(text)

def extract_text_from_docx(file_bytes: bytes) -> str:
    # python-docx wymaga ścieżki lub pliku; używamy BytesIO
    file_stream = io.BytesIO(file_bytes)
    document = docx.Document(file_stream)
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)

def chunk_text(text: str, max_chars: int = 500, overlap: int = 100) -> List[str]:
    """
    Bardzo prosty chunker po znakach, z nakładką (overlap).
    Można to później ulepszyć (np. dzielenie po zdaniach).
    """
    text = text.replace("\r\n", "\n")
    chunks = []
    start = 0
    length = len(text)
    print("chunkowanie")
    while start < length:

        end = min(start + max_chars, length)
        chunk = text[start:end].strip()
        print("start", start, "end",end,'length',length)
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break

        start = end - overlap  # overlap

        if start < 0:
            start = 0

    return chunks

def compute_embeddings(chunks: List[str]) -> np.ndarray:
    emb = embedding_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    return emb

# Metody prównań

# Zwykły iloczyn skalarny
def product_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    print("Iloczyn skalarny")
    print(np.dot(a, b))
    return np.dot(a, b)

#metoda cosinusów - unormowany iloczyn skalarny

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:

    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-10)
    b_norm = b / (np.linalg.norm(b) + 1e-10)
    return np.dot(a_norm, b_norm)

# Euklidesowa
def euclid_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a - b
    dist = np.linalg.norm(diff, axis=1)
    return -dist


# API

@app.get("/")
async def root():
    # serwujemy prosty frontend
    return FileResponse("static/index.html")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global current_chunks, current_embeddings, current_filename

    filename = file.filename
    print(filename)  # Do wycięcia
    if not filename:
        raise HTTPException(status_code=400, detail="Brak nazwy pliku.")

    ext = filename.split(".")[-1].lower()
    if ext not in ["pdf", "docx"]:
        raise HTTPException(status_code=400, detail="Obsługiwane są tylko pliki PDF i DOCX.")

    file_bytes = await file.read()

    # Ekstrakcja tekstu
    try:
        if ext == "pdf":
            text = extract_text_from_pdf(file_bytes)
        else:  # docx
            text = extract_text_from_docx(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Nie udało się odczytać pliku: {e}")
    print("Po try pdf/doc")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Plik nie zawiera tekstu lub nie udało się go odczytać.")
    print("Udało się odczytać")
    print(text)
    print("zaczynam chunkować")
    # Chunkowanie
    chunks = chunk_text(text, max_chars=500, overlap=100)

    if not chunks:
        raise HTTPException(status_code=400, detail="Nie udało się podzielić dokumentu na fragmenty.")

    # Embeddingi
    embeddings = compute_embeddings(chunks)

    # Zapis w stanie globalnym
    current_chunks = chunks
    current_embeddings = embeddings
    current_filename = filename

    return {
        "message": "Plik załadowany i zindeksowany.",
        "filename": filename,
        "num_chunks": len(chunks),
    }

# /search - wysyła query i mertodę porównania embedingów , zwraca najlepiej pasujące chunki, score i
@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    if current_embeddings is None or not current_chunks:
        raise HTTPException(status_code=400, detail="Najpierw załaduj dokument (endpoint /upload).")

    query_emb = embedding_model.encode(request.query, convert_to_numpy=True)
    metric = request.metric
    print(metric)
    if metric == "cosine": # Metoda cosinusów
        print(metric)
        scores = cosine_similarity(current_embeddings, query_emb)
    elif metric == "scalar": # iloczyn skalarny
        print(metric)
        scores = product_similarity(current_embeddings, query_emb)
    elif metric == "euclid": #odległość euklidesowa
        print(metric)
        scores = euclid_similarity(current_embeddings, query_emb)

    # Wybór top_k chunków  posortowanych malejąco po score
    top_k = min(request.top_k, len(current_chunks))
    top_indices = np.argsort(-scores)[:top_k]

    results = []
    for idx in top_indices:
        results.append(
            SearchResultChunk(
                chunk=current_chunks[int(idx)],
                score=float(scores[int(idx)]),
                index=int(idx),
            )
        )

    return SearchResponse(results=results)