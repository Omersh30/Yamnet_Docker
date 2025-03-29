from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import os
from datetime import datetime
import json

from ..audio_processing.processor import AudioProcessor
from ..llama_analysis.llama_analysis import LlamaAnalyzer

app = FastAPI(
    title="Yamnet Docker Analysis API",
    description="API for analyzing audio files using YAMNet and Llama models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize processors
audio_processor = AudioProcessor()
llama_analyzer = LlamaAnalyzer()

@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Analyze an audio file and return insights.
    """
    try:
        # Create a temporary file to store the uploaded audio
        temp_path = f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # Process the audio file
        audio_features = audio_processor.process_audio(temp_path)
        
        # Analyze with Llama
        analysis_results = llama_analyzer.analyze(audio_features)

        # Clean up temporary file
        os.remove(temp_path)

        return {
            "status": "success",
            "results": analysis_results,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint.
    """
    return {"status": "healthy"} 