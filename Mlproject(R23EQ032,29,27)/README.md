# Instrument Ratio Estimation

An ML-powered web application that analyzes audio files to estimate the percentage mix of different instruments in a song.

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up PostgreSQL database and set DATABASE_URL environment variable

3. Run the application:
   ```bash
   streamlit run app.py --server.port 5000
   ```

## Features

- Audio file upload (MP3, WAV, OGG, FLAC, M4A)
- Instrument detection for 6 categories
- Interactive visualizations
- Batch processing
- History tracking and comparison
- Export separated instrument tracks

## How It Works

This application uses spectral analysis and signal processing techniques:
- STFT (Short-Time Fourier Transform) for frequency analysis
- MFCC extraction for vocal detection
- Onset detection for percussion identification
- Chroma features for harmonic instrument analysis
- Frequency band energy analysis

For detailed methodology, see the Resources page in the app.
