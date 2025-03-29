from flask import Flask, request, jsonify
import os
import tempfile
import subprocess
import numpy as np
import soundfile as sf
import tensorflow_hub as hub
from datetime import datetime, timedelta
import pandas as pd
from scipy.signal import find_peaks
from collections import defaultdict
import gunicorn
from flask_cors import CORS
import threading
import queue
import time

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# Global variables for real-time processing
processing_queue = queue.Queue()
is_processing = False

# Add CORS support
CORS(app)

# Add a health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "CrowdVibe AI",
        "timestamp": datetime.now().isoformat()
    })

# Add a welcome endpoint
@app.route('/', methods=['GET'])
def welcome():
    return jsonify({
        "name": "CrowdVibe AI",
        "version": "1.0.0",
        "description": "AI-powered audience reaction analysis for live performances",
        "endpoints": {
            "health": "/health",
            "upload": "/upload"
        }
    })

# Load YAMNet model and class names
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
import tensorflow as tf

# Load class names from YAMNet model assets and extract only the display names
labels_path = yamnet_model.class_map_path().numpy().decode('utf-8')
class_names = [line.split(',')[-1] for line in tf.io.gfile.GFile(labels_path).read().splitlines()]

# Enhanced audience reaction categories with more specific classifications
AUDIENCE_REACTIONS = {
    'positive': {
        'applause': ['Applause', 'Clapping'],
        'cheering': ['Cheering', 'Whooping', 'Whistling'],
        'singing': ['Singing', 'Choir', 'Chorus'],
        'laughter': ['Laughter', 'Giggle', 'Chuckle'],
        'excitement': ['Yell', 'Shout', 'Scream']
    },
    'negative': {
        'disapproval': ['Booing', 'Hissing'],
        'silence': ['Silence', 'Noise'],
        'disruption': ['Cough', 'Sneeze', 'Crowd noise']
    },
    'neutral': {
        'music': ['Music', 'Song', 'Instrumental'],
        'speech': ['Speech', 'Talking', 'Conversation'],
        'ambient': ['Background noise', 'Room noise']
    }
}

def get_reaction_category(class_name):
    for sentiment, categories in AUDIENCE_REACTIONS.items():
        for subcategory, reactions in categories.items():
            if class_name in reactions:
                return sentiment, subcategory
    return 'other', 'unknown'

def format_timestamp(seconds):
    """Convert seconds to MM:SS format"""
    return str(timedelta(seconds=round(seconds)))[2:]

def calculate_engagement_metrics(results_df, waveform, sr):
    """Calculate comprehensive engagement metrics"""
    total_duration = len(waveform) / sr
    
    # Basic metrics
    metrics = {
        "total_duration": format_timestamp(total_duration),
        "average_confidence": results_df['score'].mean(),
        "total_events": len(results_df)
    }
    
    # Sentiment-based metrics
    for sentiment in AUDIENCE_REACTIONS.keys():
        sentiment_events = results_df[results_df['sentiment'] == sentiment]
        metrics[f"{sentiment}_reaction_percentage"] = (len(sentiment_events) / len(results_df)) * 100
        metrics[f"{sentiment}_average_confidence"] = sentiment_events['score'].mean() if not sentiment_events.empty else 0
    
    # Subcategory metrics
    subcategory_counts = defaultdict(int)
    subcategory_confidences = defaultdict(list)
    
    for _, row in results_df.iterrows():
        subcategory_counts[row['subcategory']] += 1
        subcategory_confidences[row['subcategory']].append(row['score'])
    
    metrics['subcategory_counts'] = dict(subcategory_counts)
    metrics['subcategory_average_confidences'] = {
        subcat: np.mean(confidences) 
        for subcat, confidences in subcategory_confidences.items()
    }
    
    # Engagement intensity analysis
    confidence_scores = results_df['score'].values
    peaks, _ = find_peaks(confidence_scores, height=0.7, distance=10)
    metrics['high_engagement_moments'] = len(peaks)
    
    # Time-based analysis
    time_windows = pd.cut(results_df['timestamp'], bins=5)
    window_engagement = results_df.groupby(time_windows)['score'].mean()
    metrics['engagement_trend'] = window_engagement.to_dict()
    
    # Add crowd energy level
    metrics['crowd_energy'] = calculate_crowd_energy(results_df)
    
    return metrics

def calculate_crowd_energy(results_df):
    """Calculate overall crowd energy level based on positive reactions"""
    positive_events = results_df[results_df['sentiment'] == 'positive']
    if positive_events.empty:
        return 0.0
    
    # Weight different types of positive reactions
    weights = {
        'applause': 0.3,
        'cheering': 0.4,
        'singing': 0.2,
        'laughter': 0.1,
        'excitement': 0.5
    }
    
    energy = 0.0
    total_weight = 0.0
    
    for subcategory, weight in weights.items():
        category_events = positive_events[positive_events['subcategory'] == subcategory]
        if not category_events.empty:
            energy += category_events['score'].mean() * weight
            total_weight += weight
    
    return energy / total_weight if total_weight > 0 else 0.0

def analyze_audio(file_path):
    waveform, sr = sf.read(file_path)
    if sr != 16000:
        raise ValueError("Audio sample rate must be 16 kHz")
    waveform = waveform.astype(np.float32)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    scores, _, _ = yamnet_model(waveform)
    scores = scores.numpy()
    top_classes = np.argmax(scores, axis=1)

    # Create a DataFrame for easier analysis
    results_df = pd.DataFrame([
        {
            "timestamp": i * 0.1,
            "class": class_names[c],
            "score": float(scores[i][c])
        }
        for i, c in enumerate(top_classes)
    ])

    # Add sentiment and subcategory information
    results_df[['sentiment', 'subcategory']] = results_df['class'].apply(
        lambda x: pd.Series(get_reaction_category(x))
    )
    results_df['formatted_time'] = results_df['timestamp'].apply(format_timestamp)

    # Calculate comprehensive engagement metrics
    engagement_metrics = calculate_engagement_metrics(results_df, waveform, sr)

    # Group events by reaction type and find significant moments
    significant_events = []
    for sentiment, categories in AUDIENCE_REACTIONS.items():
        for subcategory, reactions in categories.items():
            category_events = results_df[
                (results_df['sentiment'] == sentiment) & 
                (results_df['subcategory'] == subcategory)
            ]
            if not category_events.empty:
                high_confidence_events = category_events[category_events['score'] > 0.7]
                if not high_confidence_events.empty:
                    significant_events.append({
                        "category": f"{sentiment}_{subcategory}",
                        "count": len(high_confidence_events),
                        "average_confidence": high_confidence_events['score'].mean(),
                        "examples": high_confidence_events.head(3).to_dict('records')
                    })

    # Add performance highlights
    highlights = identify_performance_highlights(results_df, engagement_metrics)

    return {
        "engagement_metrics": engagement_metrics,
        "significant_events": significant_events,
        "detailed_timeline": results_df.to_dict('records'),
        "highlights": highlights
    }

def identify_performance_highlights(results_df, metrics):
    """Identify key moments in the performance"""
    highlights = []
    
    # Find peak engagement moments
    confidence_scores = results_df['score'].values
    peaks, _ = find_peaks(confidence_scores, height=0.8, distance=20)
    
    for peak in peaks:
        if results_df.iloc[peak]['sentiment'] == 'positive':
            highlights.append({
                "type": "peak_engagement",
                "timestamp": results_df.iloc[peak]['formatted_time'],
                "reaction": results_df.iloc[peak]['class'],
                "confidence": float(results_df.iloc[peak]['score'])
            })
    
    # Add crowd energy highlights
    if metrics['crowd_energy'] > 0.8:
        highlights.append({
            "type": "high_energy",
            "description": "Exceptional crowd energy throughout the performance",
            "energy_level": float(metrics['crowd_energy'])
        })
    
    return highlights

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, file.filename)
            file.save(input_path)

            output_path = os.path.join(tmp_dir, "converted.wav")
            subprocess.run([
                "ffmpeg", "-i", input_path,
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", output_path
            ], check=True)

            results = analyze_audio(output_path)
            return jsonify({
                "message": "File analyzed successfully",
                "results": results,
                "timestamp": datetime.now().isoformat()
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Use gunicorn in production
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
