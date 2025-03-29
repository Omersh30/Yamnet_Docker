import numpy as np
import soundfile as sf
import tensorflow_hub as hub
import pandas as pd
from datetime import datetime, timedelta
from scipy.signal import find_peaks, savgol_filter
from collections import defaultdict
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Define audience reaction categories with more specific patterns
AUDIENCE_REACTIONS = {
    'positive': {
        'applause': ['Applause', 'Clapping', 'Hands'],
        'cheering': ['Cheering', 'Whooping', 'Whistling', 'Crowd', 'Cheer'],
        'singing': ['Singing', 'Choir', 'Chorus', 'Vocal', 'Voice', 'Song'],
        'laughter': ['Laughter', 'Giggle', 'Chuckle', 'Laugh'],
        'excitement': ['Yell', 'Shout', 'Scream', 'Roar'],
        'engagement': ['Chatter', 'Conversation', 'Talk']
    },
    'negative': {
        'disapproval': ['Booing', 'Hissing', 'Boo'],
        'silence': ['Silence', 'Quiet'],
        'disruption': ['Cough', 'Sneeze', 'Noise', 'Disturbance'],
        'boredom': ['Yawn', 'Sigh', 'Groan']
    },
    'music': {
        'instruments': ['Guitar', 'Piano', 'Drum', 'Bass', 'Music', 'Musical'],
        'performance': ['Concert', 'Performance', 'Band', 'Orchestra'],
        'effects': ['Echo', 'Reverb', 'Acoustic']
    }
}

class SetlistAnalyzer:
    def __init__(self, setlist):
        self.setlist = setlist
        self.track_metrics = defaultdict(lambda: {
            'start_time': None,
            'end_time': None,
            'duration': 0,
            'reactions': defaultdict(list),
            'energy_levels': [],
            'peak_moments': [],
            'engagement_score': 0
        })
    
    def add_track_analysis(self, track_name, start_time, end_time, reactions_df):
        """Add analysis results for a specific track"""
        track_data = self.track_metrics[track_name]
        track_data['start_time'] = start_time
        track_data['end_time'] = end_time
        track_data['duration'] = end_time - start_time
        
        # Filter reactions for this track
        track_reactions = reactions_df[
            (reactions_df['timestamp'] >= start_time) & 
            (reactions_df['timestamp'] <= end_time)
        ]
        
        # Calculate track-specific metrics
        for _, reaction in track_reactions.iterrows():
            track_data['reactions'][reaction['sentiment']].append({
                'timestamp': reaction['timestamp'],
                'score': reaction['score'],
                'subcategory': reaction['subcategory']
            })
            if reaction['sentiment'] == 'positive':
                track_data['energy_levels'].append(reaction['score'])
        
        # Calculate engagement score
        positive_reactions = len(track_data['reactions']['positive'])
        total_reactions = sum(len(reactions) for reactions in track_data['reactions'].values())
        track_data['engagement_score'] = positive_reactions / total_reactions if total_reactions > 0 else 0
        
        # Find peak moments
        if track_data['energy_levels']:
            peaks, _ = find_peaks(track_data['energy_levels'], height=0.3, distance=20)
            for peak in peaks:
                track_data['peak_moments'].append({
                    'timestamp': track_reactions.iloc[peak]['timestamp'],
                    'energy_level': track_data['energy_levels'][peak],
                    'reaction': track_reactions.iloc[peak]['subcategory']
                })

def visualize_track_performance(track_metrics, output_file='track_performance.png'):
    """Create visualization of track performance"""
    plt.figure(figsize=(15, 10))
    
    # Plot energy levels over time
    plt.subplot(2, 1, 1)
    for track, metrics in track_metrics.items():
        if metrics['energy_levels']:
            times = [m['timestamp'] - metrics['start_time'] for m in metrics['reactions']['positive']]
            plt.plot(times, metrics['energy_levels'], label=track, alpha=0.7)
    
    plt.title('Track Energy Levels Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Energy Level')
    plt.legend()
    
    # Plot engagement scores
    plt.subplot(2, 1, 2)
    engagement_scores = [metrics['engagement_score'] for metrics in track_metrics.values()]
    track_names = list(track_metrics.keys())
    plt.bar(track_names, engagement_scores)
    plt.title('Track Engagement Scores')
    plt.xlabel('Track')
    plt.ylabel('Engagement Score')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

def get_reaction_category(class_name):
    """Map a class name to its reaction category and subcategory with improved matching"""
    class_name_lower = class_name.lower()
    
    for sentiment, categories in AUDIENCE_REACTIONS.items():
        for subcategory, reactions in categories.items():
            if any(reaction.lower() in class_name_lower for reaction in reactions):
                return sentiment, subcategory
    return 'other', 'unknown'

def format_timestamp(seconds):
    """Convert seconds to MM:SS format"""
    return str(timedelta(seconds=round(seconds)))[2:]

def smooth_scores(scores, window_length=5):
    """Apply Savitzky-Golay filter to smooth scores"""
    if len(scores) < window_length:
        return scores
    return savgol_filter(scores, window_length, 2)

def find_reaction_segments(results_df, min_duration=0.5, confidence_threshold=0.6):
    """Find continuous segments of reactions"""
    segments = []
    current_segment = None
    
    for _, row in results_df.iterrows():
        if row['score'] >= confidence_threshold and row['sentiment'] in ['positive', 'negative']:
            if current_segment is None:
                current_segment = {
                    'start_time': row['timestamp'],
                    'sentiment': row['sentiment'],
                    'subcategory': row['subcategory'],
                    'scores': [row['score']]
                }
            else:
                current_segment['scores'].append(row['score'])
        elif current_segment is not None:
            duration = row['timestamp'] - current_segment['start_time']
            if duration >= min_duration:
                segments.append({
                    'start_time': format_timestamp(current_segment['start_time']),
                    'end_time': format_timestamp(row['timestamp']),
                    'duration': duration,
                    'sentiment': current_segment['sentiment'],
                    'subcategory': current_segment['subcategory'],
                    'average_confidence': np.mean(current_segment['scores'])
                })
            current_segment = None
    
    return segments

def calculate_crowd_energy(results_df, window_size=50):
    """Calculate crowd energy with temporal analysis"""
    if results_df.empty:
        return 0.0, []
    
    # Weight different types of positive reactions
    weights = {
        'applause': 0.3,
        'cheering': 0.4,
        'singing': 0.2,
        'laughter': 0.1,
        'excitement': 0.5
    }
    
    # Calculate rolling energy
    positive_events = results_df[results_df['sentiment'] == 'positive'].copy()
    if positive_events.empty:
        return 0.0, []
    
    positive_events['energy'] = 0.0
    for subcategory, weight in weights.items():
        mask = positive_events['subcategory'] == subcategory
        positive_events.loc[mask, 'energy'] = positive_events.loc[mask, 'score'] * weight
    
    # Calculate rolling average
    rolling_energy = positive_events['energy'].rolling(window=window_size, min_periods=1).mean()
    
    # Find energy peaks
    peaks, _ = find_peaks(rolling_energy, height=0.3, distance=20)
    peak_moments = []
    
    for peak in peaks:
        peak_moments.append({
            'timestamp': format_timestamp(positive_events.iloc[peak]['timestamp']),
            'energy_level': rolling_energy[peak],
            'reaction': positive_events.iloc[peak]['subcategory']
        })
    
    return rolling_energy.mean(), peak_moments

def analyze_audio(file_path, setlist=None):
    # Load YAMNet model
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    
    # Load class names
    labels_path = yamnet_model.class_map_path().numpy().decode('utf-8')
    class_names = [line.split(',')[-1] for line in tf.io.gfile.GFile(labels_path).read().splitlines()]
    
    # Read and process audio
    waveform, sr = sf.read(file_path)
    if sr != 16000:
        print(f"Converting audio from {sr}Hz to 16000Hz...")
        import subprocess
        import tempfile
        import os
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, "input.wav")
            output_path = os.path.join(tmp_dir, "output.wav")
            sf.write(input_path, waveform, sr)
            
            subprocess.run([
                "ffmpeg", "-i", input_path,
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", output_path
            ], check=True)
            
            waveform, sr = sf.read(output_path)
    
    # Convert to float32 and handle stereo
    waveform = waveform.astype(np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    
    # Run inference
    print("Running YAMNet inference...")
    scores, _, _ = yamnet_model(waveform)
    scores = scores.numpy()
    
    # Smooth scores
    smoothed_scores = np.apply_along_axis(smooth_scores, 0, scores)
    top_classes = np.argmax(smoothed_scores, axis=1)
    
    # Create results DataFrame
    results_df = pd.DataFrame([
        {
            "timestamp": i * 0.1,
            "class": class_names[c],
            "score": float(smoothed_scores[i][c])
        }
        for i, c in enumerate(top_classes)
    ])
    
    # Add formatted timestamps and categories
    results_df['formatted_time'] = results_df['timestamp'].apply(format_timestamp)
    results_df[['sentiment', 'subcategory']] = pd.DataFrame(
        results_df['class'].apply(get_reaction_category).tolist(),
        index=results_df.index
    )
    
    # Find reaction segments
    reaction_segments = find_reaction_segments(results_df)
    
    # Calculate crowd energy and find peak moments
    crowd_energy, energy_peaks = calculate_crowd_energy(results_df)
    
    # Calculate temporal statistics
    time_windows = pd.cut(results_df['timestamp'], 
                         bins=10, 
                         labels=[f"Segment {i+1}" for i in range(10)])
    
    temporal_analysis = results_df.groupby(time_windows).agg({
        'sentiment': lambda x: x.value_counts().to_dict(),
        'score': ['mean', 'max']
    }).reset_index()
    
    # Convert temporal analysis to serializable format
    temporal_analysis_dict = []
    for _, row in temporal_analysis.iterrows():
        sentiment_counts = {}
        for sentiment_type, count in row['sentiment'].items():
            if isinstance(count, dict):
                sentiment_counts.update({str(k): int(v) for k, v in count.items()})
            else:
                sentiment_counts[str(sentiment_type)] = int(count)
        
        segment_dict = {
            'segment': str(row['timestamp']),
            'sentiment_counts': sentiment_counts,
            'score_mean': float(row['score']['mean']),
            'score_max': float(row['score']['max'])
        }
        temporal_analysis_dict.append(segment_dict)
    
    # Calculate overall statistics
    stats = {
        "total_duration": format_timestamp(len(waveform) / sr),
        "total_events": len(results_df),
        "crowd_energy": float(crowd_energy),
        "energy_peaks": [
            {
                'timestamp': str(peak['timestamp']),
                'energy_level': float(peak['energy_level']),
                'reaction': str(peak['reaction'])
            }
            for peak in energy_peaks
        ],
        "reaction_segments": [
            {
                'start_time': str(segment['start_time']),
                'end_time': str(segment['end_time']),
                'duration': float(segment['duration']),
                'sentiment': str(segment['sentiment']),
                'subcategory': str(segment['subcategory']),
                'average_confidence': float(segment['average_confidence'])
            }
            for segment in reaction_segments
        ],
        "sentiment_breakdown": {
            str(sentiment): float(len(results_df[results_df['sentiment'] == sentiment]) / len(results_df) * 100)
            for sentiment in AUDIENCE_REACTIONS.keys()
        },
        "temporal_analysis": temporal_analysis_dict
    }
    
    # Initialize setlist analyzer if provided
    setlist_analyzer = None
    if setlist:
        setlist_analyzer = SetlistAnalyzer(setlist)
        current_track = setlist[0]
        track_start_time = 0
        
        for i, row in results_df.iterrows():
            # Check if we've moved to the next track
            if i > 0 and i % 100 == 0:  # Check every 10 seconds
                next_track_index = setlist.index(current_track) + 1
                if next_track_index < len(setlist):
                    next_track = setlist[next_track_index]
                    # Add analysis for current track
                    setlist_analyzer.add_track_analysis(
                        current_track,
                        track_start_time,
                        row['timestamp'],
                        results_df
                    )
                    current_track = next_track
                    track_start_time = row['timestamp']
        
        # Add analysis for the last track
        setlist_analyzer.add_track_analysis(
            current_track,
            track_start_time,
            results_df['timestamp'].max(),
            results_df
        )
        
        # Create visualization
        visualize_track_performance(setlist_analyzer.track_metrics)
    
    # Add setlist analysis to stats
    if setlist_analyzer:
        stats['setlist_analysis'] = {
            track: {
                'duration': float(metrics['duration']),
                'engagement_score': float(metrics['engagement_score']),
                'peak_moments': [
                    {
                        'timestamp': str(peak['timestamp']),
                        'energy_level': float(peak['energy_level']),
                        'reaction': str(peak['reaction'])
                    }
                    for peak in metrics['peak_moments']
                ],
                'reaction_counts': {
                    str(sentiment): len(reactions)
                    for sentiment, reactions in metrics['reactions'].items()
                }
            }
            for track, metrics in setlist_analyzer.track_metrics.items()
        }
    
    return stats, results_df

def main():
    # Path to your audio file
    audio_file_path = '/Users/omersh/Desktop/pf1999-04-15d1t01.mp3'
    
    print(f"Analyzing audio file: {audio_file_path}")
    stats, results_df = analyze_audio(audio_file_path, ["Track"])
    
    # Print results
    print("\nAnalysis Results:")
    print(f"Total Duration: {stats['total_duration']}")
    print(f"Total Events Detected: {stats['total_events']}")
    print(f"Overall Crowd Energy Level: {stats['crowd_energy']:.2f}")
    
    print("\nSentiment Breakdown:")
    for sentiment, percentage in stats['sentiment_breakdown'].items():
        print(f"{sentiment.capitalize()}: {percentage:.1f}%")
    
    if 'setlist_analysis' in stats:
        print("\nTrack Analysis:")
        for track, metrics in stats['setlist_analysis'].items():
            print(f"\nDuration: {metrics['duration']:.1f}s")
            print(f"Engagement Score: {metrics['engagement_score']:.2f}")
            print("Reaction Counts:")
            for sentiment, count in metrics['reaction_counts'].items():
                print(f"  {sentiment.capitalize()}: {count}")
            print("\nPeak Moments:")
            for peak in metrics['peak_moments']:
                print(f"  Time: {float(peak['timestamp']):.1f}s - {peak['reaction']} (Energy: {peak['energy_level']:.2f})")
    
    # Save detailed results to CSV and JSON
    results_df.to_csv('analysis_results.csv', index=False)
    with open('analysis_results.json', 'w') as f:
        json.dump(stats, f, indent=2)
    print("\nDetailed results saved to 'analysis_results.csv' and 'analysis_results.json'")
    print("Track performance visualization saved to 'track_performance.png'")

if __name__ == "__main__":
    main() 