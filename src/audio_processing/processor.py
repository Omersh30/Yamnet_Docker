import librosa
import numpy as np
import soundfile as sf
import os

class ConcertAudioProcessor:
    def __init__(self):
        self.target_sr = 16000  # Standard sample rate for audio processing
        self.hop_length = 512   # Standard hop length for feature extraction
        self.downsample_factor = 10  # Downsample by a factor of 10 to reduce data size
        self.frame_size = 2048  # Frame size for spectral analysis
        self.n_mels = 128  # Number of mel bands for spectral analysis

    def process(self, audio_path):
        """Process audio file and extract features."""
        # Load audio at target sample rate
        audio, sr = librosa.load(audio_path, sr=self.target_sr)
        
        # Calculate basic features
        duration = librosa.get_duration(y=audio, sr=sr)
        tempo = librosa.beat.tempo(y=audio, sr=sr)[0]
        
        # Calculate spectral features
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            n_fft=self.frame_size
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Calculate onset strength
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        
        # Calculate dynamic range
        dynamic_range = np.max(np.abs(audio)) - np.min(np.abs(audio))
        
        # Downsample audio for storage
        frame_size = self.downsample_factor
        n_frames = len(audio) // frame_size
        audio_frames = audio[:n_frames * frame_size].reshape(-1, frame_size)
        audio_downsampled = np.mean(audio_frames, axis=1)
        
        # Calculate energy envelope
        energy_envelope = librosa.feature.rms(y=audio, frame_length=self.frame_size, hop_length=self.hop_length)[0]
        
        return {
            "audio": audio_downsampled.tolist(),
            "sample_rate": sr // self.downsample_factor,
            "duration": float(duration),
            "tempo": float(tempo),
            "dynamic_range": float(dynamic_range),
            "energy_envelope": energy_envelope.tolist(),
            "onset_strength": onset_env.tolist(),
            "mel_spectrogram": mel_spec_db.tolist()
        }

    def convert_to_wav(self, input_path, output_path=None):
        """Convert audio to WAV format with specified sample rate."""
        if output_path is None:
            output_path = os.path.splitext(input_path)[0] + '.wav'
        
        # Load audio file
        audio, sr = librosa.load(input_path, sr=self.target_sr)
        
        # Save as WAV
        sf.write(output_path, audio, self.target_sr)
        return output_path 