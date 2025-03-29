from llama_cpp import Llama
import json

def analyze_with_llama(results_file):
    # Load the analysis results
    with open(results_file, 'r') as f:
        analysis_results = json.load(f)
    
    # Initialize Llama model
    llm = Llama(
        model_path="models/llama-2-7b-chat.gguf",
        n_ctx=2048,
        n_threads=4
    )
    
    # System prompt
    system_prompt = """You are an expert concert performance analyst. Your analysis should be:
- Data-driven: Use specific numbers and timestamps
- Concise: Focus on key insights
- Actionable: Provide clear recommendations
- Structured: Use clear sections and bullet points"""
    
    # User prompt
    user_prompt = f"""Analyze this concert performance data:

KEY METRICS:
Duration: {analysis_results['total_duration']}
Events: {analysis_results['total_events']}
Energy: {analysis_results['crowd_energy']:.2f}

SENTIMENT:
{json.dumps(analysis_results['sentiment_breakdown'], indent=2)}

PEAK MOMENTS (timestamp, energy, reaction):
{json.dumps(analysis_results['setlist_analysis']['Track']['peak_moments'], indent=2)}

Provide a concise analysis in this format:

1. PERFORMANCE OVERVIEW
   - Key metrics and their significance
   - Notable patterns in the data

2. ENGAGEMENT ANALYSIS
   - Peak moments and their impact
   - Audience reaction patterns

3. RECOMMENDATIONS
   - 2-3 specific, actionable improvements
   - Focus on highest impact changes

Use specific numbers and timestamps to support your analysis."""
    
    # Generate insights
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = llm.create_chat_completion(
        messages=messages,
        max_tokens=2000,
        temperature=0.7,
        top_p=0.95
    )
    
    # Extract and save insights
    insights = response['choices'][0]['message']['content'].strip()
    with open('llama_insights.txt', 'w') as f:
        f.write(insights)
    
    print("\nLlama 2 Analysis:")
    print("=" * 50)
    print(insights)
    print("\nInsights saved to 'llama_insights.txt'")

if __name__ == "__main__":
    analyze_with_llama('analysis_results.json') 