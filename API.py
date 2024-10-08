from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn
from api_rag import RAGPipelineSetup
from typing import List

app = FastAPI()

EMBEDDINGS_MODEL_NAME = "BAAI/bge-m3"
QDRANT_URL = "https://88bb6378-66e7-49db-a5de-6bb17f0d664a.europe-west3-0.gcp.cloud.qdrant.io:6333"
HUGGINGFACE_API_KEY = "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"
QDRANT_API_KEY = "cLGVHbp48h0CZayJIXdxVW-JJijODOKpBFlzIPm6nvLHxRE4B_nrFA"
GROQ_API_KEY= "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"
GROQ_API_KEY2= "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"
GROQ_API_KEY3= "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"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

DATABASE_TO_COLLECTION = {
    "Trường Đại học Khoa học Tự nhiên": "US_vectorDB",
    "Trường Đại học Công nghệ Thông tin": "UIT_vectorDB",
    "Trường Đại Học Khoa Học Xã Hội - Nhân Văn": "USSH_vectorDB",
    "Trường Đại Học Bách Khoa": "UT_vectorDB",
    "Trường Đại Học Quốc Tế": "IU_vectorDB",
    "Trường Đại Học Kinh tế - Luật": "UEL_vectorDB"
}

class PromptRequest(BaseModel):
    prompt: str
    database: str

@app.post("/api/query")
async def query(prompt_request: PromptRequest):
    selected_collection = DATABASE_TO_COLLECTION.get(prompt_request.database, "US_vectorDB")
    rag_setup = RAGPipelineSetup(
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        qdrant_collection_name=selected_collection,
        huggingface_api_key=HUGGINGFACE_API_KEY,
        embeddings_model_name=EMBEDDINGS_MODEL_NAME,
        groq_api_key=GROQ_API_KEY,
        groq_api_key2=GROQ_API_KEY2,
        groq_api_key3=GROQ_API_KEY3,
        reranker_model_name=RERANKER_MODEL_NAME 
    )

    rag_pipeline = rag_setup.rag(source=selected_collection)
    response = rag_pipeline({"question": prompt_request.prompt})
    return {"response": response["answer"]}


@app.head("/api/query")
async def query_head(request: Request):
    return {"status": "HEAD request successful"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
