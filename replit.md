# Instrument Ratio Estimation

## Overview
An ML-powered web application that analyzes audio files to estimate the percentage mix of different instruments in a song using spectral analysis and machine learning techniques.

## Current State
- **Status**: Feature Complete
- **Last Updated**: December 2, 2025

## Features

### Core Analysis
- Audio file upload (MP3, WAV, OGG, FLAC, M4A formats)
- Instrument detection for 6 categories: Vocals, Drums, Bass, Guitar, Piano, Other
- Tempo detection
- Sample rate analysis

### Visualizations
- Interactive pie chart and bar chart of instrument percentages
- Audio waveform visualization
- Mel spectrogram display
- Frequency band energy distribution
- Instrument-specific spectrograms (filtered by frequency range)

### Batch Processing
- Upload and analyze multiple songs at once
- Comparative view of all analyzed songs

### History & Comparison
- PostgreSQL database storage for all analyses
- History dashboard to view past analyses
- Compare up to 5 songs side-by-side
- Radar chart comparison of instrument profiles

### Export Features
- Download analysis report (text)
- Download JSON data
- Export separated instrument tracks as WAV files

### Resources & Documentation
- Downloadable source code (ZIP archive with all files)
- Individual file downloads
- Links to recommended audio datasets (MUSDB18, MedleyDB, FMA, MTG-Jamendo)
- Model performance documentation
- Methodology explanation (spectral analysis approach)
- Estimated detection confidence levels per instrument

## Project Architecture

### Files
- `app.py` - Main Streamlit application with UI, visualizations, and navigation
- `audio_analyzer.py` - Core audio analysis module with ML algorithms
- `models.py` - SQLAlchemy database models for persistence
- `.streamlit/config.toml` - Streamlit configuration

### Key Components

#### InstrumentAnalyzer Class (audio_analyzer.py)
- `load_audio()` - Loads and processes audio files
- `extract_spectral_features()` - Extracts STFT, mel spectrogram, MFCCs, etc.
- `analyze_frequency_bands()` - Analyzes energy across frequency bands
- `detect_percussion()` - Uses onset detection for drum identification
- `detect_vocals()` - MFCC-based vocal detection
- `detect_bass()` - Low-frequency analysis for bass detection
- `detect_harmonic_instruments()` - Chroma-based harmonic analysis
- `estimate_instrument_mix()` - Main analysis orchestrator
- `get_instrument_spectrograms()` - Generates filtered spectrograms per instrument
- `separate_instrument_audio()` - Bandpass filter for instrument isolation
- `get_all_separated_tracks()` - Exports all instrument tracks

#### Database Models (models.py)
- `AnalysisHistory` - Stores song analysis results with all instrument percentages
- Functions for CRUD operations on analysis history

### Technologies
- **Streamlit** - Web framework
- **librosa** - Audio processing and feature extraction
- **Plotly** - Interactive visualizations
- **scikit-learn** - ML preprocessing
- **scipy** - Signal processing and bandpass filtering
- **pydub** - Audio file handling
- **SQLAlchemy** - Database ORM
- **PostgreSQL** - Persistent storage
- **soundfile** - Audio export

## How It Works
1. Audio is loaded and converted to mono at 22050 Hz
2. Spectral features are extracted (STFT, MFCCs, chroma, etc.)
3. Multiple detection algorithms analyze different aspects:
   - Percussion: Onset detection, beat tracking, regularity analysis
   - Vocals: MFCC variance, spectral centroid in vocal range
   - Bass: Low-frequency energy ratio
   - Harmonic instruments: Chroma analysis, mid-frequency energy
4. Raw scores are normalized to percentages
5. Results are saved to PostgreSQL database

## Model Performance Note
This application uses **unsupervised spectral analysis** rather than a supervised ML classifier. Traditional metrics (accuracy, F1 score, precision, recall) require labeled ground truth data. The Resources page provides:
- Detailed methodology documentation
- Estimated confidence levels per instrument
- Links to labeled datasets for future supervised training
- Example code for calculating traditional ML metrics

## Pages
1. **Single Analysis** - Upload and analyze one song
2. **Batch Analysis** - Analyze multiple songs at once
3. **History** - View all past analyses
4. **Compare** - Side-by-side comparison of selected songs
5. **Resources** - Download code, access datasets, view model documentation

## Running the Application
```bash
streamlit run app.py --server.port 5000
```
