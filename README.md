# Deepfake Audio Detection API

A deep learning service that detects AI-generated fake audio by analyzing multiple audio representations simultaneously.

## How It Works

The model takes a WAV file and extracts three parallel features:

1. **Mel Spectrogram** — frequency content over time (128 mel bands)
2. **MFCCs** — 13 cepstral coefficients capturing timbral characteristics
3. **Raw Waveform** — normalized audio signal

All three are fed into a multi-input PyTorch model that outputs a probability score (0 = real, 1 = fake).

## Audio Processing Pipeline

- Converts stereo to mono
- Resamples to 16kHz
- Pads or truncates to 10 seconds
- Normalizes each feature independently

## Tech Stack

- **Model:** PyTorch (TorchScript)
- **Audio Processing:** TorchAudio
- **Serving:** BentoML
- **Containerization:** Docker (Debian, with FFmpeg + libsndfile)
- **Hardware:** GPU-accelerated inference

## Running the Service

```bash
# Install dependencies
pip install -r requirements.txt

# Serve locally
bentoml serve service:AudioClassificationService

# Build a Bento for deployment
bentoml build
```

## API Usage

Send a POST request with a WAV file to the `/predict` endpoint:

```bash
curl -X POST http://localhost:3000/predict \
  -F "file=@sample.wav"
```

**Response:**
```json
{
  "prediction": 1.0,
  "probability": 0.87
}
```

- `prediction`: 0 (real) or 1 (fake)
- `probability`: model confidence (0.0 to 1.0)
