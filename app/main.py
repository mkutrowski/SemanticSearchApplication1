import io
import torch
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz
import docx




app = FastAPI(title="Semantic Search Application 1")

# Wyb or modelu embeddingowego
DEFAULT_MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(DEFAULT_MODEL_NAME)
current_model_name: str = DEFAULT_MODEL_NAME

#Defi nicje
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    metric: str = "cosine"

class SearchResultChunk(BaseModel):
    chunk: str
    score: float
    index: int

class IndexRequest(BaseModel):
    model_name: str = DEFAULT_MODEL_NAME
    max_chars: int = 500
    overlap: int = 100

class SearchResponse(BaseModel):
    results: List[SearchResultChunk]


current_chunks: List[str] = []
current_embeddings: Optional[torch.Tensor] = None
current_filename: Optional[str] = None
current_raw_text: Optional[str] = None


# Definicje funkcji

def extract_text_from_pdf(file_bytes: bytes) -> str:
    text = []
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text.append(page.get_text())
    return "\n".join(text)

def extract_text_from_docx(file_bytes: bytes) -> str:
    file_stream = io.BytesIO(file_bytes)
    document = docx.Document(file_stream)
    paragraphs = [p.text for p in document.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def get_text_chunks(text: str, max_chars: int , overlap: int ) -> List[str]:
    text_splitter=RecursiveCharacterTextSplitter(
        separators=["\n",". "],
        chunk_size=max_chars,
        chunk_overlap=overlap
    )
    chunks = text_splitter.split_text(text)
    return chunks

def compute_embeddings(chunks: List[str]):
    emb = embedding_model.encode(chunks, convert_to_tensor=True, show_progress_bar=False)
    return emb

# Wybór funkcji scoringowej
def get_score_function(metric: str):
    if metric == "cosine":
        return util.cos_sim
    elif metric == "dot":
        return util.dot_score
    elif metric == "l2":
        return util.euclidean_sim
    elif metric == "l1":
        return util.manhattan_sim
    else:
        raise HTTPException(status_code=400, detail="Niepoprtawna metryka")


# =================== API =================================
# zwraca GUI
@app.get("/")
async def root():
       return FileResponse("static/index.html")

# =================POST/upload - pobiera plik, dzieli na fragmentuy i indeksuje==============
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global current_raw_text, current_filename, current_chunks,  current_embeddings
    print("upload")
    filename = file.filename
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

    if not text.strip():
        raise HTTPException(status_code=400, detail="Plik nie zawiera tekstu lub nie udało się go odczytać.")


    # Zapis w zmiennych  globalnych
    current_raw_text = text
    current_filename = filename



    return {
        "message": "Plik został załadowany.",
        "filename": filename
    }

@app.post("/index")
async def index_document(request: IndexRequest):
    global current_embeddings,embedding_model, current_model_name, current_raw_text, current_chunks
    model_name=request.model_name
    max_chars = request.max_chars
    overlap = request.overlap

    if current_raw_text is None:
        raise HTTPException(status_code=400, detail="Najpierw załaduj dokument.")

    # Dzielenie dokumentu na chunki
    chunks = get_text_chunks(current_raw_text, max_chars, overlap)

    if not chunks:
        raise HTTPException(status_code=400, detail="Nie udało się podzielić dokumentu na fragmenty.")


    print("model select")
    print(model_name)
    if model_name != current_model_name:
        try:
            print(model_name)
            embedding_model = SentenceTransformer(model_name)
            current_model_name = model_name
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Nie udało się załadować modelu '{model_name}': {e}"
            )
    # Indeksacja chunków

    embeddings = compute_embeddings(chunks)

    current_chunks = chunks
    current_embeddings = embeddings

    return {
        "message": "Plik został podzielony i  zaindeksowany.",
        "model_name": current_model_name,
        "num_chunks": len(current_chunks)
    }

# ==============POST/search - pobiera pytanie  i metrykę. Zwraca najlepiej pasujące chunki=============
@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    if current_embeddings is None or not current_chunks:
        raise HTTPException(status_code=400, detail="Najpierw załaduj dokument.")

    query_emb = embedding_model.encode_query(request.query, convert_to_tensor=True)

    metric = request.metric # Pobranie metryki  z request
    score_function = get_score_function(metric) # Wybór funkcji  scoringowej dla  metryki"

    top_k = min(request.top_k, len(current_chunks)) # top_k nie może byc większa od liczby chunków

    hits = util.semantic_search(
        query_emb,
        current_embeddings,
        top_k=top_k,
        score_function=score_function
    )

    results = []
    for hit in hits[0]:
        idx = int(hit["corpus_id"])
        score = float(hit["score"])
        results.append(
            SearchResultChunk(
                chunk=current_chunks[idx],
                score=score,
                index=idx,
            )
        )

    return SearchResponse(results=results)