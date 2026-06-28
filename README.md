# Iris — Environmental Intelligence Scanner
### FutureHacks 8 · Build the City of Tomorrow

> *See what your choices leave behind.*

Iris uses real webcam eye tracking to scan everyday products for their environmental impact. Look at something — Iris scans it. No clicks, no hovering. Just your gaze.

---

## What it does

1. **Calibrate** — follow dots on screen with your eyes (~45 seconds)
2. **Look** at any product — a reticle follows your gaze in real time
3. **Dwell** for 1 second — Iris locks on and triggers an AI scan
4. **See** the results: microplastic risk, recyclability, impact score, and threat level (LOW → CRITICAL)

Today it's a Python app. Tomorrow it lives in AR glasses.

---

## Setup

```bash
# 1. Install dependencies
pip install eyetrax --no-deps
pip install mediapipe opencv-python numpy requests websockets screeninfo

# 2. Get a free Groq API key at groq.com (no credit card needed)

# 3. Run
GROQ_API_KEY=your_key_here python3.11 iris.py

# 4. Open the scanner in Chrome
# Go to localhost:3000/web/scanner.html
# (run python3 -m http.server 3000 in a second terminal tab)
```

---

## Tech Stack

- **EyeTrax** — webcam gaze estimation via MediaPipe + dense grid calibration
- **Groq + Llama 4 Scout** — real-time environmental AI analysis
- **WebSocket** — streams gaze coordinates 30x/second to the browser frontend
- **OpenCV** — backend gaze rendering
- **HTML/CSS/JS + Spline** — frontend scanner UI

---

## The Vision

The city of tomorrow needs citizens who can make informed environmental choices instantly — not after reading labels, not after Googling. Iris is the environmental intelligence layer: look at something, know its impact. Today it's your laptop. Next it's your glasses.

---

Built by Sanjoli Chattopadhyay & Uma Manthina · FutureHacks 8
