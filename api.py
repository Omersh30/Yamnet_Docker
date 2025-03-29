from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import os
from test_model import analyze_audio
from llama_cpp import Llama
import json

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Llama model
llm = Llama(
    model_path="models/llama-2-7b-chat.gguf",
    n_ctx=2048,
    n_threads=4
)

def generate_insights(analysis_results):
    """Generate insights using Llama 2"""
    prompt = f"""
    Analyze this live performance data and provide insights:
    
    Duration: {analysis_results['total_duration']}
    Total Events: {analysis_results['total_events']}
    Crowd Energy: {analysis_results['crowd_energy']:.2f}
    
    Sentiment Breakdown:
    {json.dumps(analysis_results['sentiment_breakdown'], indent=2)}
    
    Peak Moments:
    {json.dumps(analysis_results['setlist_analysis']['Track']['peak_moments'], indent=2)}
    
    Please provide:
    1. Overall performance quality assessment
    2. Key crowd engagement moments and their significance
    3. Audience reaction patterns
    4. Areas for improvement
    5. Recommendations for future performances
    
    Format the response in a clear, structured way.
    """
    
    response = llm(
        prompt,
        max_tokens=1000,
        stop=["###"],
        echo=False
    )
    
    return response['choices'][0]['text'].strip()

@app.post("/analyze")
async def analyze_audio_file(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    try:
        # Run YAMNet analysis
        stats, results_df = analyze_audio(temp_file_path, ["Track"])
        
        # Generate insights using Llama 2
        insights = generate_insights(stats)
        
        # Combine results
        response = {
            "technical_analysis": stats,
            "llm_insights": insights
        }
        
        return response
        
    finally:
        # Clean up temporary file
        os.unlink(temp_file_path)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 