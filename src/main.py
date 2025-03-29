import os
from datetime import datetime
from audio_processing.processor import ConcertAudioProcessor
from yamnet_analysis.analyzer import SoundAnalyzer
from llama_analysis.llama_analysis import LlamaAnalyzer
from data_storage.storage import save_analysis

def setup_environment():
    """Ensure all required models and directories are set up."""
    from setup import setup_models
    setup_models()

def main():
    """Main function to run the concert analysis pipeline."""
    print("Starting concert analysis...")
    
    # Setup environment
    setup_environment()
    
    # Initialize components
    audio_processor = ConcertAudioProcessor()
    sound_analyzer = SoundAnalyzer()
    llama_analyzer = LlamaAnalyzer()
    
    # Process audio file
    audio_file = "/Users/omersh/Desktop/pf1999-04-15d1t01.mp3"
    print(f"\nProcessing audio file: {audio_file}")
    
    # Extract audio features
    print("Extracting audio features...")
    audio_features = audio_processor.process(audio_file)
    
    # Analyze sound and crowd reactions
    print("\nAnalyzing sound and crowd reactions...")
    sound_analysis = sound_analyzer.analyze(audio_features)
    
    # Generate insights with Llama
    print("\nGenerating insights with Llama...")
    llama_analysis = llama_analyzer.analyze(audio_features, sound_analysis)
    
    # Combine results
    results = {
        "metadata": {
            "analysis_id": f"pf1999_04_15_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "audio_file": audio_file
        },
        "audio_features": audio_features,
        "sound_analysis": sound_analysis,
        "llama_analysis": llama_analysis
    }
    
    # Save results
    save_analysis(results)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 