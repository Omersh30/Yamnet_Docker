from llama_cpp import Llama
import json
import os
from typing import Dict, Any, List
import numpy as np
from datetime import datetime
from collections import defaultdict

class LlamaAnalyzer:
    def __init__(self, model_path: str = "models/llama-2-7b-chat.gguf"):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4
        )

    def _group_reactions(self, reactions: List[Dict], time_threshold: float = 0.5) -> List[Dict]:
        """Group reactions that occur within the time threshold."""
        if not reactions:
            return []
            
        # Sort reactions by start time
        sorted_reactions = sorted(reactions, key=lambda x: x['start_time'])
        groups = []
        current_group = [sorted_reactions[0]]
        
        for reaction in sorted_reactions[1:]:
            if reaction['start_time'] - current_group[-1]['end_time'] <= time_threshold:
                current_group.append(reaction)
            else:
                groups.append(self._merge_reaction_group(current_group))
                current_group = [reaction]
        
        if current_group:
            groups.append(self._merge_reaction_group(current_group))
            
        return groups

    def _merge_reaction_group(self, group: List[Dict]) -> Dict:
        """Merge a group of reactions into a single event."""
        return {
            'start_time': group[0]['start_time'],
            'end_time': group[-1]['end_time'],
            'duration': group[-1]['end_time'] - group[0]['start_time'],
            'average_score': np.mean([r.get('average_confidence', 0) for r in group]),
            'peak_score': max([r.get('average_confidence', 0) for r in group]),
            'reaction_type': group[0]['subcategory'],
            'count': len(group)
        }

    def _analyze_reaction_patterns(self, reactions: List[Dict]) -> Dict:
        """Analyze patterns in the reactions."""
        patterns = {
            'reaction_counts': defaultdict(int),
            'reaction_durations': defaultdict(list),
            'reaction_intensities': defaultdict(list),
            'time_distribution': defaultdict(int)
        }
        
        for reaction in reactions:
            reaction_type = reaction['subcategory']
            patterns['reaction_counts'][reaction_type] += 1
            patterns['reaction_durations'][reaction_type].append(reaction['duration'])
            patterns['reaction_intensities'][reaction_type].append(reaction['average_confidence'])
            
            # Group by time segments (e.g., every 60 seconds)
            time_segment = int(reaction['start_time'] / 60)
            patterns['time_distribution'][time_segment] += 1
            
        return patterns

    def analyze(self, audio_features: Dict, sound_analysis: Dict) -> Dict:
        # Calculate mean energy from energy envelope
        mean_energy = np.mean(audio_features['energy_envelope'])
        
        # Get and group reactions
        reactions = sound_analysis.get('reaction_segments', [])
        grouped_reactions = self._group_reactions(reactions)
        patterns = self._analyze_reaction_patterns(reactions)
        
        # Create the prompt for Llama
        prompt = f"""You are analyzing crowd reaction data from a live concert. Based on the following performance data and crowd reactions, provide a detailed analysis:

Performance Metrics:
- Duration: {audio_features['duration']:.2f} seconds
- Tempo: {audio_features['tempo']:.2f} BPM
- Mean Energy: {mean_energy:.4f}
- Dynamic Range: {audio_features['dynamic_range']:.4f}

Grouped Crowd Reactions:
{json.dumps(grouped_reactions, indent=2)}

Reaction Patterns:
{json.dumps(patterns, indent=2)}

Please provide a structured analysis with the following sections:

[REACTION ANALYSIS]
• Group events by type and timing
• Identify the most engaged moments
• Analyze reaction patterns and distribution

[ENGAGEMENT INSIGHTS]
• Most frequent reaction types
• Peak engagement moments
• Temporal patterns across the performance

[TECHNICAL ANALYSIS]
• Sound quality and balance
• Energy progression
• Performance dynamics

[RECOMMENDATIONS]
• Areas for improvement
• Engagement optimization suggestions

Format your response in clear sections with specific examples and data points."""

        # Get Llama's response
        response = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are an expert concert analyst specializing in crowd reaction analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.7
        )

        # Extract the response text
        analysis_text = response['choices'][0]['message']['content']

        # Create timestamp for file names
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save the full analysis text to a separate file
        analysis_file = f"analysis_results/llama_analysis_{timestamp}.txt"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write(analysis_text)
        print(f"\nLlama analysis saved to: {analysis_file}")

        # Create a summary version with key metrics and analysis
        summary = {
            'timestamp': timestamp,
            'performance_metrics': {
                'duration': audio_features['duration'],
                'tempo': audio_features['tempo'],
                'mean_energy': mean_energy,
                'dynamic_range': audio_features['dynamic_range']
            },
            'reaction_metrics': {
                'total_reactions': len(reactions),
                'grouped_reactions': len(grouped_reactions),
                'reaction_patterns': patterns
            },
            'analysis_text': analysis_text
        }

        # Save the summary to a separate JSON file
        summary_file = f"analysis_results/llama_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f"Analysis summary saved to: {summary_file}")

        return summary

    def _format_notable_reactions(self, reactions):
        """Format notable reactions for the prompt."""
        formatted_reactions = []
        for reaction in reactions[:13]:  # Limit to top 13 reactions
            formatted_reactions.append(
                f"• Time: {reaction['start_time']} - {reaction['end_time']} "
                f"(Duration: {reaction['duration']:.2f}s, Sentiment: {reaction['sentiment']}, "
                f"Type: {reaction['subcategory']}, Confidence: {reaction['average_confidence']:.4f})"
            )
        return "\n".join(formatted_reactions)

    def save_analysis(self, analysis: str, output_path: str):
        """Save the analysis to a text file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(analysis)
            print(f"Analysis saved to: {output_path}")
        except Exception as e:
            print(f"Error saving analysis: {str(e)}")

def analyze_with_llama(results):
    """Analyze results using Llama model."""
    # Initialize Llama model
    model_path = "models/llama-2-7b-chat.gguf"
    if not os.path.exists(model_path):
        print(f"Warning: Llama model not found at {model_path}")
        return
        
    llm = Llama(
        model_path=model_path,
        n_ctx=2048,
        n_threads=4
    )
    
    # Extract crowd reaction summary
    crowd_engagement = results['sound_analysis']['crowd_engagement']
    crowd_reactions = results['sound_analysis']['crowd_reactions']
    
    # Create a more focused prompt
    prompt = f"""Analyze this concert performance data with focus on crowd reactions and engagement:

Performance Overview:
- Duration: {results['audio_features']['duration']:.2f} seconds
- Tempo: {results['audio_features']['tempo']:.2f} BPM

Crowd Engagement Metrics:
- Number of Reactions: {crowd_engagement['reaction_count']}
- Average Reaction Duration: {crowd_engagement['average_reaction_duration']:.2f} seconds
- Maximum Reaction Intensity: {crowd_engagement['max_reaction_intensity']:.4f}
- Total Reaction Time: {crowd_engagement['total_reaction_time']:.2f} seconds

Notable Crowd Reactions:
{json.dumps(crowd_reactions[:3], indent=2)}  # Show first 3 reactions

Please analyze:
1. Overall Crowd Engagement
   - How engaged was the audience?
   - Were there any particularly strong reactions?

2. Reaction Patterns
   - Are reactions evenly distributed?
   - Any notable patterns in reaction timing?

3. Performance Impact
   - How did crowd reactions relate to the performance?
   - What moments generated the strongest responses?

4. Recommendations
   - What does this suggest about audience preferences?
   - How could crowd engagement be improved?

Format your response in clear sections with specific examples."""

    # Generate analysis with smaller chunk size
    try:
        response = llm(
            prompt,
            max_tokens=1000,
            stop=["###"],
            echo=False
        )
        
        # Extract and format the response
        analysis = response['choices'][0]['text'].strip()
        
        # Save insights to file with timestamp
        timestamp = results['metadata']['timestamp']
        output_file = f"analysis_results/llama_insights_{results['metadata']['analysis_id']}.txt"
        
        with open(output_file, "w") as f:
            f.write(f"Concert Analysis (Duration: {timestamp:.2f}s)\n")
            f.write("=" * 50 + "\n\n")
            f.write(analysis)
        
        print(f"\nAnalysis completed and saved to {output_file}")
        return analysis
        
    except Exception as e:
        print(f"Error during Llama analysis: {str(e)}")
        return None 