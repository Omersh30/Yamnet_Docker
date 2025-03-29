import os
import json
from datetime import datetime

def save_analysis(results):
    """Save analysis results to a JSON file."""
    # Create analysis_results directory if it doesn't exist
    os.makedirs('analysis_results', exist_ok=True)
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_id = results['metadata']['analysis_id']
    
    # Create filename
    filename = f"{analysis_id}_{timestamp}.json"
    filepath = os.path.join('analysis_results', filename)
    
    # Save results to JSON file
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")
    return filepath 