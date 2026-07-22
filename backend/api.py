import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import query_chroma_db_and_llama as query_module

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

@app.get("/")
async def root():
    return FileResponse(frontend_dir / "index.html")

class QueryRequest(BaseModel):
    prompt: str
    debug: bool = False

class QueryResponse(BaseModel):
    answer: str
    context: str

@app.on_event("startup")
async def startup_event():
    model_path = Path(os.environ.get("LLAMA_MODEL_PATH", "Meta-Llama-3-8B-Instruct.Q4_0.gguf"))
    db_path = Path(os.environ.get("CHROMA_DB_PATH", "data/db/"))
    '''
    if not model_path.exists():
        raise FileNotFoundError(
            f"LLAMA model file not found: {model_path}. "
            "Set LLAMA_MODEL_PATH to the correct .gguf file location or place the file in the app directory."
        )
    if not db_path.exists():
        raise FileNotFoundError(
            f"Chroma DB path not found: {db_path}. "
            "Set CHROMA_DB_PATH to the correct database folder."
        )
    '''
    query_module.load_resources(model_path=str(model_path), db_path=str(db_path), gpu=True)

@app.post("/api/query", response_model=QueryResponse)
async def api_query(request: QueryRequest):
    try:
        context = query_module.query_collection(request.prompt, debug=request.debug)
        answer = query_module.process_query(request.prompt, context)
        return QueryResponse(answer=answer, context=context)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
