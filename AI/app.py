import os
import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
from pathlib import Path
import tempfile
import asyncio
from typing import Optional
import json
import base64

# Import AI components from existing app
from app_version_3 import AITherapyAssistant

app = FastAPI()

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_path = Path("static")
if static_path.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize AI Assistant
ai_assistant = AITherapyAssistant()

@app.post("/api/process-audio")
async def process_audio(audio: UploadFile = File(...)):
    try:
        # Create temp directory if it doesn't exist
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        
        # Save uploaded audio to temporary file
        temp_audio = temp_dir / f"temp_audio_{os.urandom(8).hex()}.wav"
        try:
            contents = await audio.read()
            with open(temp_audio, "wb") as f:
                f.write(contents)
            
            # Process audio asynchronously
            user_text, ai_text, ai_audio_path = await asyncio.to_thread(
                ai_assistant.process_user_input,
                str(temp_audio)
            )
            
            # Read the generated audio file
            with open(ai_audio_path, "rb") as audio_file:
                ai_audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
            
            # Clean up temporary files
            temp_audio.unlink()
            Path(ai_audio_path).unlink()
            
            return JSONResponse(content={
                "user_text": user_text,
                "ai_text": ai_text,
                "ai_audio": ai_audio_data
            })
        
        except Exception as e:
            # Clean up temp file in case of error
            if temp_audio.exists():
                temp_audio.unlink()
            raise e
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return {"status": "AI Assistant API is running"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True) 