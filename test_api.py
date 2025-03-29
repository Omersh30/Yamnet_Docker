from test_model import analyze_audio
import json

def test_analysis():
    # Path to your test audio file
    audio_file_path = '/Users/omersh/Desktop/pf1999-04-15d1t01.mp3'
    
    try:
        # Run YAMNet analysis directly
        print("Running YAMNet analysis...")
        stats, results_df = analyze_audio(audio_file_path, ["Track"])
        
        # Print technical analysis
        print("\nTechnical Analysis:")
        print(f"Duration: {stats['total_duration']}")
        print(f"Total Events: {stats['total_events']}")
        print(f"Crowd Energy: {stats['crowd_energy']:.2f}")
        
        print("\nSentiment Breakdown:")
        for sentiment, percentage in stats['sentiment_breakdown'].items():
            print(f"{sentiment.capitalize()}: {percentage:.1f}%")
        
        print("\nPeak Moments:")
        for peak in stats['setlist_analysis']['Track']['peak_moments']:
            print(f"Time: {float(peak['timestamp']):.1f}s - {peak['reaction']} (Energy: {peak['energy_level']:.2f})")
        
        # Save results to JSON file
        with open('analysis_results.json', 'w') as f:
            json.dump(stats, f, indent=2)
        print("\nResults saved to 'analysis_results.json'")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_analysis() 