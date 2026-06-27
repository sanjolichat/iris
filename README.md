# Iris — Environmental Intelligence Scanner
### FutureHacks 8 · Build the City of Tomorrow

> *See what your choices leave behind.*

Iris uses real webcam eye tracking to scan everyday products for their environmental impact. Look at something — Iris scans it.

## What it does

1. **Calibrate** — follow dots on screen with your eyes (takes ~45 seconds)
2. **Look** at any product on screen — a green reticle follows your gaze
3. **Dwell** for 1 second — Iris locks on and scans
4. **See** the environmental HUD: microplastic risk, recyclability, impact score, threat level

Today it's a Python app. Tomorrow it lives in AR glasses.

## Setup

```bash
# 1. Install dependencies
pip install eyetrax opencv-python numpy

# 2. Get a free Groq API key at groq.com (no credit card)

# 3. Run
python iris.py
# Enter your Groq key when prompted
```

Or just:
```bash
bash run.sh
```

## Controls

| Key     | Action              |
|---------|---------------------|
| `SPACE` | Dismiss current scan |
| `R`     | Recalibrate          |
| `ESC`   | Quit                 |

## Tech stack

- **EyeTrax** — webcam gaze estimation (MediaPipe + Ridge regression)
- **Groq + Llama 4 Scout** — environmental AI analysis
- **OpenCV** — HUD overlay rendering
- **Python** — glue

## The vision

The city of tomorrow needs citizens who can make informed environmental choices instantly — not after reading labels, not after Googling. Iris is the environmental intelligence layer: look at something, know its impact. Today it's on your laptop. Next it's in your glasses.
