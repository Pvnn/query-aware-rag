import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from src.rag_pipeline import QueryAwareRAG
from src.data.demo_loader import load_demo_datasets

load_dotenv()
token = os.getenv("HF_TOKEN")

app = FastAPI(title="Query-Aware RAG Compression API")

# Enable CORS for the frontend UI
app.add_middleware(
  CORSMiddleware,
  allow_origins=["*"], 
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

# Global variables to hold state
pipeline = None
demo_data = None
current_dataset_id = None

@app.on_event("startup")
def startup_event():
  global pipeline, demo_data
  # Initialize the heavy models once on startup
  pipeline = QueryAwareRAG(token=token)
  # Load available static queries and corpora
  demo_data = load_demo_datasets()
  print("✓ FastAPI Server Ready.")


class LoadDatasetRequest(BaseModel):
  dataset_id: str

class QueryRequest(BaseModel):
  query: str
  top_k: int = 4
  compare_original: bool = True
  use_coarse: bool = True
  use_fine: bool = True


@app.get("/datasets")
def get_datasets():
  """Returns available datasets and their selectable queries."""
  if not demo_data:
    raise HTTPException(status_code=500, detail="Demo data not loaded.")
  
  return {
    dataset_id: {
      "name": info["name"],
      "queries": info["queries"]
    }
    for dataset_id, info in demo_data.items()
  }

@app.post("/dataset/load")
def load_dataset(request: LoadDatasetRequest):
  """Indexes the documents of the selected dataset into the Retriever."""
  global current_dataset_id
  
  if request.dataset_id not in demo_data:
    raise HTTPException(status_code=404, detail="Dataset not found")
      
  if request.dataset_id == current_dataset_id:
    return {"status": "success", "message": f"Dataset {request.dataset_id} is already loaded."}
      
  docs = demo_data[request.dataset_id]["documents"]
  
  # Feed documents to the DenseRetriever
  pipeline.retriever.index_documents(docs)
  current_dataset_id = request.dataset_id
  
  return {
    "status": "success", 
    "message": f"Indexed {len(docs)} documents for {request.dataset_id}"
  }

@app.post("/query")
def run_query(request: QueryRequest):
  """Executes the end-to-end RAG pipeline."""
  if not current_dataset_id:
    raise HTTPException(status_code=400, detail="No dataset loaded. Call /dataset/load first.")
      
  try:
    result = pipeline.run(
      query=request.query,
      top_k=request.top_k,
      compare_original=request.compare_original,
      use_coarse=request.use_coarse,
      use_fine=request.use_fine
    )
    return result
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)