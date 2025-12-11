"""
Audio Analyzer Module for Instrument Ratio Estimation

This module provides functionality to analyze audio files and estimate
the percentage mix of different instruments using spectral analysis
and machine learning techniques.
"""

import numpy as np
import librosa
from scipy import signal
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import MinMaxScaler
import io


class InstrumentAnalyzer:
    """Analyzes audio files to estimate instrument mix percentages."""
    
    INSTRUMENT_CATEGORIES = {
        'vocals': {
            'freq_range': (80, 1100),
            'description': 'Human voice and vocal harmonics'
        },
        'drums': {
            'freq_range': (20, 250),
            'description': 'Percussion, kick, snare, hi-hats'
        },
        'bass': {
            'freq_range': (40, 300),
            'description': 'Bass guitar, synth bass, low-end'
        },
        'guitar': {
            'freq_range': (80, 1200),
            'description': 'Electric and acoustic guitar'
        },
        'piano': {
            'freq_range': (27, 4200),
            'description': 'Piano, keys, synthesizers'
        },
        'other': {
            'freq_range': (200, 8000),
            'description': 'Strings, brass, synths, effects'
        }
    }
    
    def __init__(self, sr=22050):
        """Initialize the analyzer with target sample rate."""
        self.sr = sr
        self.scaler = MinMaxScaler()
        
    def load_audio(self, audio_file):
        """
        Load audio from file or bytes.
        
        Args:
            audio_file: File path or file-like object
            
        Returns:
            tuple: (audio_data, sample_rate, duration)
        """
        if isinstance(audio_file, (str, bytes)):
            if isinstance(audio_file, bytes):
                audio_file = io.BytesIO(audio_file)
            y, sr = librosa.load(audio_file, sr=self.sr, mono=True)
        else:
            y, sr = librosa.load(audio_file, sr=self.sr, mono=True)
            
        duration = librosa.get_duration(y=y, sr=sr)
        return y, sr, duration
    
    def extract_spectral_features(self, y, sr):
        """
        Extract spectral features from audio signal.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            dict: Spectral features including STFT, mel spectrogram, etc.
        """
        stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        
        zero_crossings = librosa.feature.zero_crossing_rate(y)[0]
        
        rms = librosa.feature.rms(y=y)[0]
        
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        return {
            'stft': stft,
            'mel_spec': mel_spec,
            'mel_spec_db': mel_spec_db,
            'spectral_centroid': spectral_centroid,
            'spectral_rolloff': spectral_rolloff,
            'spectral_bandwidth': spectral_bandwidth,
            'zero_crossings': zero_crossings,
            'rms': rms,
            'mfccs': mfccs,
            'onset_env': onset_env,
            'chroma': chroma
        }
    
    def analyze_frequency_bands(self, stft, sr):
        """
        Analyze energy in different frequency bands.
        
        Args:
            stft: Short-time Fourier transform
            sr: Sample rate
            
        Returns:
            dict: Energy values for each frequency band
        """
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        
        band_energies = {}
        
        sub_bass_mask = (freqs >= 20) & (freqs < 60)
        bass_mask = (freqs >= 60) & (freqs < 250)
        low_mid_mask = (freqs >= 250) & (freqs < 500)
        mid_mask = (freqs >= 500) & (freqs < 2000)
        high_mid_mask = (freqs >= 2000) & (freqs < 4000)
        high_mask = (freqs >= 4000) & (freqs < 8000)
        brilliance_mask = (freqs >= 8000)
        
        band_energies['sub_bass'] = np.mean(stft[sub_bass_mask, :]) if np.any(sub_bass_mask) else 0
        band_energies['bass'] = np.mean(stft[bass_mask, :]) if np.any(bass_mask) else 0
        band_energies['low_mid'] = np.mean(stft[low_mid_mask, :]) if np.any(low_mid_mask) else 0
        band_energies['mid'] = np.mean(stft[mid_mask, :]) if np.any(mid_mask) else 0
        band_energies['high_mid'] = np.mean(stft[high_mid_mask, :]) if np.any(high_mid_mask) else 0
        band_energies['high'] = np.mean(stft[high_mask, :]) if np.any(high_mask) else 0
        band_energies['brilliance'] = np.mean(stft[brilliance_mask, :]) if np.any(brilliance_mask) else 0
        
        return band_energies
    
    def detect_percussion(self, y, sr, onset_env):
        """
        Detect percussion/drum presence using onset detection.
        
        Args:
            y: Audio time series
            sr: Sample rate
            onset_env: Onset envelope
            
        Returns:
            float: Percussion score (0-1)
        """
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        onset_peaks = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        
        onset_strength = np.mean(onset_env)
        onset_variance = np.var(onset_env)
        
        if len(onset_peaks) > 0:
            regularity = 1.0 / (1.0 + np.std(np.diff(onset_peaks)) / (np.mean(np.diff(onset_peaks)) + 1e-6))
        else:
            regularity = 0.0
            
        percussion_score = min(1.0, (onset_strength * 0.4 + onset_variance * 0.3 + regularity * 0.3))
        
        return percussion_score, tempo
    
    def detect_vocals(self, mfccs, spectral_centroid, spectral_bandwidth):
        """
        Detect vocal presence using MFCC and spectral features.
        
        Args:
            mfccs: MFCC coefficients
            spectral_centroid: Spectral centroid values
            spectral_bandwidth: Spectral bandwidth values
            
        Returns:
            float: Vocal score (0-1)
        """
        mfcc_variance = np.mean(np.var(mfccs, axis=1))
        
        centroid_mean = np.mean(spectral_centroid)
        centroid_var = np.var(spectral_centroid)
        
        vocal_range = (centroid_mean > 500) and (centroid_mean < 3000)
        
        bandwidth_consistency = 1.0 / (1.0 + np.var(spectral_bandwidth) / (np.mean(spectral_bandwidth) + 1e-6))
        
        vocal_score = 0.0
        if vocal_range:
            vocal_score = min(1.0, mfcc_variance * 0.01 + bandwidth_consistency * 0.5 + 0.3)
        else:
            vocal_score = min(0.5, mfcc_variance * 0.005 + bandwidth_consistency * 0.2)
            
        return vocal_score
    
    def detect_bass(self, band_energies, spectral_centroid):
        """
        Detect bass instrument presence.
        
        Args:
            band_energies: Energy in different frequency bands
            spectral_centroid: Spectral centroid values
            
        Returns:
            float: Bass score (0-1)
        """
        low_freq_energy = band_energies['sub_bass'] + band_energies['bass']
        total_energy = sum(band_energies.values()) + 1e-6
        
        low_freq_ratio = low_freq_energy / total_energy
        
        low_centroid_frames = np.sum(spectral_centroid < 400) / len(spectral_centroid)
        
        bass_score = min(1.0, low_freq_ratio * 2.0 + low_centroid_frames * 0.3)
        
        return bass_score
    
    def detect_harmonic_instruments(self, chroma, band_energies, spectral_features):
        """
        Detect harmonic instruments (guitar, piano, strings).
        
        Args:
            chroma: Chromagram
            band_energies: Frequency band energies
            spectral_features: All spectral features
            
        Returns:
            dict: Scores for harmonic instruments
        """
        chroma_variance = np.mean(np.var(chroma, axis=1))
        chroma_peaks = np.sum(chroma > 0.5 * np.max(chroma)) / chroma.size
        
        mid_energy = band_energies['low_mid'] + band_energies['mid']
        high_mid_energy = band_energies['high_mid']
        total_energy = sum(band_energies.values()) + 1e-6
        
        mid_ratio = mid_energy / total_energy
        high_mid_ratio = high_mid_energy / total_energy
        
        spectral_centroid = spectral_features['spectral_centroid']
        centroid_mean = np.mean(spectral_centroid)
        
        guitar_score = min(1.0, mid_ratio * 1.5 + chroma_variance * 0.5)
        if centroid_mean < 800 or centroid_mean > 2500:
            guitar_score *= 0.7
            
        piano_score = min(1.0, (mid_ratio + high_mid_ratio) * 1.2 + chroma_peaks * 2.0)
        
        other_score = min(1.0, high_mid_ratio * 2.0 + band_energies['high'] / total_energy * 1.5)
        
        return {
            'guitar': guitar_score,
            'piano': piano_score,
            'other': other_score
        }
    
    def estimate_instrument_mix(self, audio_file):
        """
        Main method to estimate instrument mix percentages.
        
        Args:
            audio_file: Path or file-like object of audio file
            
        Returns:
            dict: Instrument percentages and analysis data
        """
        y, sr, duration = self.load_audio(audio_file)
        
        features = self.extract_spectral_features(y, sr)
        
        band_energies = self.analyze_frequency_bands(features['stft'], sr)
        
        percussion_score, tempo = self.detect_percussion(y, sr, features['onset_env'])
        
        vocal_score = self.detect_vocals(
            features['mfccs'],
            features['spectral_centroid'],
            features['spectral_bandwidth']
        )
        
        bass_score = self.detect_bass(band_energies, features['spectral_centroid'])
        
        harmonic_scores = self.detect_harmonic_instruments(
            features['chroma'],
            band_energies,
            features
        )
        
        raw_scores = {
            'vocals': vocal_score,
            'drums': percussion_score,
            'bass': bass_score,
            'guitar': harmonic_scores['guitar'],
            'piano': harmonic_scores['piano'],
            'other': harmonic_scores['other']
        }
        
        total_score = sum(raw_scores.values())
        if total_score > 0:
            percentages = {k: (v / total_score) * 100 for k, v in raw_scores.items()}
        else:
            percentages = {k: 100/len(raw_scores) for k in raw_scores.keys()}
        
        percentages = {k: round(v, 1) for k, v in percentages.items()}
        
        return {
            'percentages': percentages,
            'raw_scores': raw_scores,
            'tempo': float(tempo) if isinstance(tempo, np.ndarray) else tempo,
            'duration': duration,
            'sample_rate': sr,
            'audio_data': y,
            'spectral_features': features,
            'band_energies': band_energies
        }
    
    def get_waveform_data(self, y, sr, num_points=1000):
        """
        Get downsampled waveform data for visualization.
        
        Args:
            y: Audio time series
            sr: Sample rate
            num_points: Number of points for visualization
            
        Returns:
            tuple: (time_axis, amplitude_data)
        """
        if len(y) > num_points:
            factor = len(y) // num_points
            y_downsampled = y[::factor][:num_points]
        else:
            y_downsampled = y
            
        time_axis = np.linspace(0, len(y) / sr, len(y_downsampled))
        
        return time_axis, y_downsampled
    
    def get_spectrogram_data(self, features):
        """
        Get spectrogram data for visualization.
        
        Args:
            features: Spectral features dictionary
            
        Returns:
            numpy.ndarray: Mel spectrogram in dB
        """
        return features['mel_spec_db']
    
    def get_instrument_spectrograms(self, y, sr):
        """
        Generate spectrograms filtered for each instrument's frequency range.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            dict: Spectrograms for each instrument category
        """
        stft_complex = librosa.stft(y, n_fft=2048, hop_length=512)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        
        instrument_spectrograms = {}
        
        for instrument, config in self.INSTRUMENT_CATEGORIES.items():
            low_freq, high_freq = config['freq_range']
            
            mask = (freqs >= low_freq) & (freqs <= high_freq)
            
            filtered_stft = np.zeros_like(stft_complex)
            filtered_stft[mask, :] = stft_complex[mask, :]
            
            magnitude = np.abs(filtered_stft)
            spec_db = librosa.amplitude_to_db(magnitude, ref=np.max)
            
            instrument_spectrograms[instrument] = {
                'spectrogram': spec_db,
                'freq_range': (low_freq, high_freq),
                'freqs': freqs[mask] if np.any(mask) else np.array([]),
                'energy': np.mean(magnitude[mask, :]) if np.any(mask) else 0
            }
        
        return instrument_spectrograms
    
    def separate_instrument_audio(self, y, sr, instrument):
        """
        Separate audio for a specific instrument using bandpass filtering.
        
        Args:
            y: Audio time series
            sr: Sample rate
            instrument: Instrument name
            
        Returns:
            numpy.ndarray: Filtered audio for the instrument
        """
        if instrument not in self.INSTRUMENT_CATEGORIES:
            raise ValueError(f"Unknown instrument: {instrument}")
        
        low_freq, high_freq = self.INSTRUMENT_CATEGORIES[instrument]['freq_range']
        
        nyquist = sr / 2
        low = max(low_freq / nyquist, 0.001)
        high = min(high_freq / nyquist, 0.999)
        
        if low >= high:
            return np.zeros_like(y)
        
        order = 4
        sos = signal.butter(order, [low, high], btype='band', output='sos')
        filtered_audio = signal.sosfilt(sos, y)
        
        return filtered_audio
    
    def get_all_separated_tracks(self, y, sr):
        """
        Get all separated instrument tracks.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            dict: Separated audio for each instrument
        """
        separated = {}
        for instrument in self.INSTRUMENT_CATEGORIES.keys():
            separated[instrument] = self.separate_instrument_audio(y, sr, instrument)
        return separated
