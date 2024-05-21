from fastapi import FastAPI, File, UploadFile, HTTPException
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from typing import List
from fastapi.responses import JSONResponse, RedirectResponse
import whisper
import torch
from tempfile import NamedTemporaryFile

# Check if CUDA is available and set the device accordingly
torch.cuda.is_available()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load the Whisper model
model = whisper.load_model("base", fp16=False)

app = FastAPI(middleware[
    Middleware(CORSMiddleware, allow_origins=["*"],
                allow_methods=["*"],
                allow_headers=["*"])

])

@app.post("/whisper_local")
async def transcribe_audio(files: List[UploadFile] = File( ... )):
    if not files:
        raise HTTPException(status_code=400, detail="Only one file is allowed")
    
    results = []

    for file in files:
        with NamedTemporaryFile(delete=True) as temp:
            with open(temp.name, "wb") as temp_file:
                temp_file.write(file.file.read())

            result = model.transcribe(temp.name)

            results.append(
                {
                    "filename": file.filename,
                    "transcript": result["text"]
                }
            )

    return JSONResponse(content={'results': results})


@app.get("/", response_class=RedirectResponse)
async def redirect_to_docs():
    return "/docs"
