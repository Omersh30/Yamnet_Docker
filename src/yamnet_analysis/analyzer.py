import numpy as np
from abc import ABC, abstractmethod
import os
import librosa
from typing import Dict, Any, List

class SoundAnalyzer(ABC):
    @abstractmethod
    def analyze(self, audio_data):
        pass

class SimpleAnalyzer:
    def __init__(self):
        self.window_size = 1000  # Window size for energy calculation
        self.crowd_freq_range = (200, 2000)  # Frequency range for crowd noise (Hz)
        self.energy_threshold = 0.001  # Threshold for detecting reactions
        self.min_reaction_duration = 0.5  # Minimum duration for a reaction in seconds
        self.reaction_gap_threshold = 2  # Maximum gap between peaks to consider them part of the same reaction
        self.smoothing_window = 5  # Window size for energy smoothing

    def analyze(self, audio_data, sample_rate):
        """Analyze audio data for crowd reactions and energy levels."""
        # Calculate number of windows
        n_samples = len(audio_data)
        n_windows = n_samples // self.window_size
        if n_windows == 0:
            n_windows = 1
        
        # Pad audio data if needed
        if n_samples % self.window_size != 0:
            pad_length = self.window_size - (n_samples % self.window_size)
            audio_padded = np.pad(audio_data, (0, pad_length))
        else:
            audio_padded = audio_data

        # Reshape into windows
        energy_windows = audio_padded[:n_windows * self.window_size].reshape(-1, self.window_size)
        
        # Calculate energy levels for each window
        energy_levels = np.mean(np.abs(energy_windows), axis=1)
        
        # Apply smoothing to reduce noise
        energy_levels = self._smooth_energy(energy_levels)
        
        # Calculate average and max energy
        average_energy = np.mean(energy_levels)
        max_energy = np.max(energy_levels)
        
        # Find peaks (potential crowd reactions)
        peak_threshold = max(self.energy_threshold, average_energy * 1.5)
        peaks = np.where(energy_levels > peak_threshold)[0]
        
        # Group consecutive peaks into reactions
        reactions = self._group_reactions(peaks, energy_levels, sample_rate)
        
        # Calculate crowd engagement metrics
        engagement_metrics = self._calculate_engagement_metrics(reactions)
        
        # Find significant moments
        significant_moments = self._find_significant_moments(energy_levels, sample_rate)
        
        return {
            "duration": float(n_samples / sample_rate),
            "energy_levels": energy_levels.tolist(),
            "peak_moments": [float(p * self.window_size / sample_rate) for p in peaks],
            "average_energy": float(average_energy),
            "max_energy": float(max_energy),
            "crowd_reactions": reactions,
            "crowd_engagement": engagement_metrics,
            "significant_moments": significant_moments
        }
    
    def _smooth_energy(self, energy_levels):
        """Apply smoothing to energy levels to reduce noise."""
        kernel = np.ones(self.smoothing_window) / self.smoothing_window
        return np.convolve(energy_levels, kernel, mode='same')
    
    def _group_reactions(self, peaks, energy_levels, sample_rate):
        """Group consecutive peaks into crowd reactions."""
        reactions = []
        if len(peaks) > 0:
            reaction_start = peaks[0]
            current_reaction = [peaks[0]]
            
            for i in range(1, len(peaks)):
                if peaks[i] - peaks[i-1] <= self.reaction_gap_threshold:
                    current_reaction.append(peaks[i])
                else:
                    # End current reaction
                    reaction_end = peaks[i-1]
                    duration = (reaction_end - reaction_start + 1) * (self.window_size / sample_rate)
                    if duration >= self.min_reaction_duration:
                        intensity = np.mean(energy_levels[current_reaction])
                        reactions.append({
                            "start_time": float(reaction_start * self.window_size / sample_rate),
                            "end_time": float(reaction_end * self.window_size / sample_rate),
                            "duration": float(duration),
                            "intensity": float(intensity),
                            "peak_count": len(current_reaction)
                        })
                    # Start new reaction
                    reaction_start = peaks[i]
                    current_reaction = [peaks[i]]
            
            # Handle last reaction
            reaction_end = peaks[-1]
            duration = (reaction_end - reaction_start + 1) * (self.window_size / sample_rate)
            if duration >= self.min_reaction_duration:
                intensity = np.mean(energy_levels[current_reaction])
                reactions.append({
                    "start_time": float(reaction_start * self.window_size / sample_rate),
                    "end_time": float(reaction_end * self.window_size / sample_rate),
                    "duration": float(duration),
                    "intensity": float(intensity),
                    "peak_count": len(current_reaction)
                })
        
        return reactions
    
    def _calculate_engagement_metrics(self, reactions):
        """Calculate comprehensive crowd engagement metrics."""
        if not reactions:
            return {
                "reaction_count": 0,
                "average_reaction_duration": 0,
                "max_reaction_intensity": 0,
                "total_reaction_time": 0.0,
                "reaction_density": 0.0,
                "average_peaks_per_reaction": 0
            }
        
        total_reaction_time = sum(r["duration"] for r in reactions)
        avg_reaction_duration = total_reaction_time / len(reactions)
        max_reaction_intensity = max((r["intensity"] for r in reactions), default=0)
        total_peaks = sum(r["peak_count"] for r in reactions)
        
        return {
            "reaction_count": len(reactions),
            "average_reaction_duration": float(avg_reaction_duration),
            "max_reaction_intensity": float(max_reaction_intensity),
            "total_reaction_time": float(total_reaction_time),
            "reaction_density": float(len(reactions) / (reactions[-1]["end_time"] - reactions[0]["start_time"])),
            "average_peaks_per_reaction": float(total_peaks / len(reactions))
        }
    
    def _find_significant_moments(self, energy_levels, sample_rate):
        """Find significant moments in the performance."""
        # Find local maxima that are significantly above average
        mean_energy = np.mean(energy_levels)
        std_energy = np.std(energy_levels)
        threshold = mean_energy + 2 * std_energy
        
        significant_indices = np.where(energy_levels > threshold)[0]
        
        return [{
            "timestamp": float(idx * self.window_size / sample_rate),
            "energy_level": float(energy_levels[idx]),
            "significance": float((energy_levels[idx] - mean_energy) / std_energy)
        } for idx in significant_indices]

class EnsembleAnalyzer:
    def __init__(self):
        self.analyzer = SimpleAnalyzer()
        
    def analyze(self, audio_data):
        """Analyze audio using the simple analyzer."""
        return self.analyzer.analyze(audio_data)

class SoundAnalyzer:
    def __init__(self):
        self.crowd_freq_range = (100, 1000)  # Hz range for crowd noise
        self.reaction_threshold = 0.7
        self.min_reaction_duration = 0.5  # seconds
        self.max_reaction_duration = 10.0  # seconds

    def analyze(self, audio_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze audio features to detect crowd reactions and compute engagement metrics.
        
        Args:
            audio_features: Dictionary containing audio data and features
            
        Returns:
            Dictionary containing analysis results
        """
        # Convert audio to numpy array if needed
        audio = np.array(audio_features['audio'])
        sample_rate = audio_features['sample_rate']
        
        # Calculate spectrogram
        spec = librosa.stft(audio)
        spec_db = librosa.amplitude_to_db(np.abs(spec))
        
        # Detect crowd reactions
        reactions = self._detect_crowd_reactions(spec_db, audio, sample_rate)
        
        # Calculate engagement metrics
        metrics = self._calculate_engagement_metrics(reactions)
        
        return {
            'crowd_reactions': reactions,
            'engagement_metrics': metrics
        }

    def _detect_crowd_reactions(self, spectrogram: np.ndarray, audio: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """
        Detect crowd reactions in the audio using spectral analysis.
        
        Args:
            spectrogram: The spectrogram of the audio
            audio: The raw audio signal
            sample_rate: The sample rate of the audio
            
        Returns:
            List of detected reactions with their properties
        """
        # Parameters for analysis
        hop_length = 512  # Standard hop length for STFT
        
        # Find frequency range for crowd noise (typically 1-4 kHz)
        freq_bins = librosa.fft_frequencies(sr=sample_rate, n_fft=2048)
        crowd_range = (freq_bins >= self.crowd_freq_range[0]) & (freq_bins <= self.crowd_freq_range[1])
        
        # Calculate energy in crowd frequency range
        crowd_energy = np.mean(spectrogram[crowd_range], axis=0)
        
        # Find segments with high energy (potential reactions)
        threshold = np.mean(crowd_energy) + self.reaction_threshold * np.std(crowd_energy)
        is_reaction = crowd_energy > threshold
        
        # Find continuous segments
        reaction_segments = []
        start = None
        
        for i in range(len(is_reaction)):
            if is_reaction[i] and start is None:
                start = i
            elif not is_reaction[i] and start is not None:
                duration = (i - start) * hop_length / sample_rate
                if self.min_reaction_duration <= duration <= self.max_reaction_duration:
                    reaction_segments.append({
                        'start_time': start * hop_length / sample_rate,
                        'end_time': i * hop_length / sample_rate,
                        'duration': duration,
                        'intensity': float(np.mean(crowd_energy[start:i]))
                    })
                start = None
        
        return reaction_segments

    def _calculate_engagement_metrics(self, reactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate engagement metrics based on detected crowd reactions.
        
        Args:
            reactions: List of detected crowd reactions
            
        Returns:
            Dictionary containing engagement metrics
        """
        if not reactions:
            return {
                'total_reactions': 0,
                'total_reaction_time': 0.0,
                'average_reaction_duration': 0.0,
                'average_reaction_intensity': 0.0,
                'reaction_density': 0.0
            }
        
        total_duration = sum(r['duration'] for r in reactions)
        avg_duration = total_duration / len(reactions)
        avg_intensity = np.mean([r['intensity'] for r in reactions])
        
        # Calculate reaction density (reactions per minute)
        if reactions:
            time_span = reactions[-1]['end_time'] - reactions[0]['start_time']
            reaction_density = len(reactions) / (time_span / 60) if time_span > 0 else 0
        else:
            reaction_density = 0
        
        return {
            'total_reactions': len(reactions),
            'total_reaction_time': total_duration,
            'average_reaction_duration': avg_duration,
            'average_reaction_intensity': float(avg_intensity),
            'reaction_density': float(reaction_density)  # reactions per minute
        } 