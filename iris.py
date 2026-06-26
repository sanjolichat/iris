"""
Iris — Environmental Intelligence Scanner
FutureHacks 8: Build the City of Tomorrow

Uses real webcam eye tracking (EyeTrax) to detect where you're looking.
When you dwell on a product for ~1 second, Iris scans it and shows
its environmental impact in a futuristic HUD overlay.
"""

import cv2
import numpy as np
import threading
import time
import json
import os
import sys
import urllib.request
import urllib.parse
import urllib.error
import requests as _requests

from eyetrax import GazeEstimator, run_9_point_calibration

# ── Config ─────────────────────────────────────────────────────────────────────

API_KEY     = os.environ.get("GROQ_API_KEY", "")
MODEL       = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_URL    = "https://api.groq.com/openai/v1/chat/completions"
BG_PATH     = os.path.join(os.path.dirname(__file__), "background.png")
DWELL_SECS  = 1.0      # seconds of gaze dwell before triggering scan
SCAN_RADIUS = 80       # pixels — how close gaze must stay to count as dwell
WINDOW_W    = 1280
WINDOW_H    = 720

# ── Colour palette (BGR for OpenCV) ────────────────────────────────────────────
GREEN       = (0, 255, 136)
GREEN_DIM   = (0, 140, 70)
GREEN_FAINT = (0, 60, 30)
RED         = (60, 51, 255)
ORANGE      = (0, 170, 255)
BLACK       = (6, 13, 2)
WHITE       = (220, 245, 220)
PANEL_BG    = (6, 20, 10)

# ── Product zones (x1, y1, x2, y2, label) ──────────────────────────────────────
# These match the products drawn in background.py
PRODUCT_ZONES = [
    # Shelf 1 — clothing
    ( 80,  20, 210, 180, "H&M Polyester Jacket"),
    (240,  20, 370, 180, "ZARA Nylon Sweater"),
    (400,  20, 530, 180, "GAP Cotton T-Shirt"),
    (560,  20, 690, 180, "Nike Fleece Hoodie"),
    (720,  20, 850, 180, "H&M Acrylic Scarf"),
    (880,  20,1010, 180, "Lulu Spandex Leggings"),
    (1040, 20,1170, 180, "Levis Denim Jeans"),
    # Shelf 2 — grocery/plastic
    ( 80, 220, 180, 380, "Evian Plastic Bottle"),
    (210, 240, 310, 380, "Tide Detergent Pod"),
    (340, 230, 450, 380, "Heinz Ketchup Bottle"),
    (480, 250, 580, 380, "Glad Plastic Wrap"),
    (610, 225, 730, 380, "Pantene Shampoo Bottle"),
    (760, 240, 860, 380, "Kellogg Cereal Box"),
    (890, 230, 1000,380, "Chobani Yogurt Container"),
    (1030,235,1130, 380, "Lays Chips Bag"),
    # Shelf 3 — electronics/misc
    ( 80, 430, 200, 560, "OtterBox Phone Case"),
    (230, 440, 340, 560, "Anker USB Charger"),
    (370, 430, 500, 560, "Sony Foam Earpods"),
    (530, 440, 630, 560, "JanSport Nylon Backpack"),
    (660, 430, 780, 560, "Crocs Rubber Sandals"),
    (810, 435, 920, 560, "Ray-Ban Plastic Sunglasses"),
    (950, 430,1070, 560, "REI Synthetic Towel"),
    (1100,440,1200,560, "Coach PVC Wallet"),
]

# ── State ───────────────────────────────────────────────────────────────────────

class IrisState:
    def __init__(self):
        self.gaze_x = WINDOW_W // 2
        self.gaze_y = WINDOW_H // 2
        self.dwell_start   = None
        self.dwell_product = None
        self.scanning      = False
        self.scan_result   = None
        self.scan_product  = None
        self.hud_alpha     = 0.0      # fade in/out
        self.lock          = threading.Lock()

state = IrisState()

# ── Groq API ────────────────────────────────────────────────────────────────────

def call_groq(product_label):
    prompt = f"""You are Iris, an environmental AI scanner built into smart glasses.
The user is looking at: "{product_label}".

Analyze the environmental impact in 2-3 sentences. Focus on microplastics, wastewater compliance, and sustainability. Be specific and data-grounded.

Respond ONLY with this exact JSON, no other text:
{{
  "analysis": "2-3 sentence analysis here",
  "threat": "HIGH",
  "metric1_label": "Microplastic Risk",
  "metric1_value": "8.2/10",
  "metric2_label": "Recyclable",
  "metric2_value": "12%",
  "metric3_label": "Impact Score",
  "metric3_value": "8.4/10",
  "tags": ["SYNTHETIC POLYMER", "MICROFIBER", "NONCOMPLIANT"]
}}
Threat must be one of: CRITICAL, HIGH, MEDIUM, LOW."""

    try:
        resp = _requests.post(
            GROQ_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {API_KEY}"
            },
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 400,
                "temperature": 0.3
            },
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
        raw  = data["choices"][0]["message"]["content"]
        clean = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(clean)
    except Exception as e:
        return {
            "analysis": f"Scan error: {e}",
            "threat": "UNKNOWN",
            "metric1_label": "Error", "metric1_value": "—",
            "metric2_label": "Error", "metric2_value": "—",
            "metric3_label": "Error", "metric3_value": "—",
            "tags": ["SCAN FAILED"]
        }

def scan_async(product_label):
    result = call_groq(product_label)
    with state.lock:
        state.scan_result  = result
        state.scan_product = product_label
        state.scanning     = False

# ── Product detection ───────────────────────────────────────────────────────────

def product_at(x, y):
    for (x1, y1, x2, y2, label) in PRODUCT_ZONES:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return label
    return None

# ── Drawing helpers ─────────────────────────────────────────────────────────────

def draw_rounded_rect(img, x1, y1, x2, y2, color, thickness=1, r=8):
    cv2.rectangle(img, (x1+r, y1), (x2-r, y2), color, thickness)
    cv2.rectangle(img, (x1, y1+r), (x2, y2-r), color, thickness)
    cv2.ellipse(img, (x1+r, y1+r), (r,r), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-r, y1+r), (r,r), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1+r, y2-r), (r,r),  90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2-r, y2-r), (r,r),   0, 0, 90, color, thickness)

def fill_rounded_rect(img, x1, y1, x2, y2, color, r=8):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1+r), (x2, y2-r), color, -1)
    cv2.rectangle(overlay, (x1+r, y1), (x2-r, y2), color, -1)
    cv2.ellipse(overlay, (x1+r, y1+r), (r,r), 180, 0, 90, color, -1)
    cv2.ellipse(overlay, (x2-r, y1+r), (r,r), 270, 0, 90, color, -1)
    cv2.ellipse(overlay, (x1+r, y2-r), (r,r),  90, 0, 90, color, -1)
    cv2.ellipse(overlay, (x2-r, y2-r), (r,r),   0, 0, 90, color, -1)
    cv2.addWeighted(overlay, 0.92, img, 0.08, 0, img)

def put_text(img, text, x, y, color=WHITE, scale=0.45, thickness=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)

def put_text_mono(img, text, x, y, color=GREEN, scale=0.38, thickness=1):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN,
                scale * 2.2, color, thickness, cv2.LINE_AA)

# ── Gaze crosshair ──────────────────────────────────────────────────────────────

def draw_gaze(img, gx, gy, dwelling, progress):
    """Draw a futuristic gaze reticle with dwell progress ring."""
    r = 22
    # Outer faint circle
    cv2.circle(img, (gx, gy), r+4, GREEN_FAINT, 1)
    # Cross hairs
    cv2.line(img, (gx-r, gy), (gx-6, gy), GREEN_DIM, 1)
    cv2.line(img, (gx+6, gy), (gx+r, gy), GREEN_DIM, 1)
    cv2.line(img, (gx, gy-r), (gx, gy-6), GREEN_DIM, 1)
    cv2.line(img, (gx, gy+6), (gx, gy+r), GREEN_DIM, 1)
    # Center dot
    cv2.circle(img, (gx, gy), 3, GREEN, -1)

    if dwelling and progress > 0:
        angle = int(360 * progress)
        cv2.ellipse(img, (gx, gy), (r, r), -90, 0, angle, GREEN, 2)

# ── HUD panel ───────────────────────────────────────────────────────────────────

def draw_hud(img, result, product_label, scanning):
    """Draw the Iris environmental scan HUD in the bottom-right corner."""
    pw, ph = 380, 260
    px = WINDOW_W - pw - 20
    py = WINDOW_H - ph - 20

    # Panel background
    panel = np.zeros((ph, pw, 3), dtype=np.uint8)
    panel[:] = (6, 20, 10)

    # Border
    cv2.rectangle(panel, (0, 0), (pw-1, ph-1), (0, 100, 50), 1)
    cv2.line(panel, (0, 0), (30, 0), GREEN, 2)
    cv2.line(panel, (0, 0), (0, 30), GREEN, 2)
    cv2.line(panel, (pw-30, ph-1), (pw-1, ph-1), GREEN, 2)
    cv2.line(panel, (pw-1, ph-30), (pw-1, ph-1), GREEN, 2)

    # Header
    cv2.putText(panel, "IRIS", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, tuple(GREEN), 2, cv2.LINE_AA)
    cv2.circle(panel, (5, 18), 4, tuple(GREEN), -1)
    cv2.putText(panel, "ENVIRONMENTAL SCANNER", (45, 18),
                cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 140, 70), 1, cv2.LINE_AA)
    cv2.line(panel, (8, 34), (pw-8, 34), (0, 60, 30), 1)

    # Product name
    label_short = product_label[:42] + ("…" if len(product_label) > 42 else "")
    cv2.putText(panel, label_short.upper(), (10, 52),
                cv2.FONT_HERSHEY_PLAIN, 0.95, (0, 150, 80), 1, cv2.LINE_AA)

    if scanning:
        # Scanning animation
        dots = "." * (int(time.time() * 3) % 4)
        cv2.putText(panel, f"SCANNING{dots}", (10, 90),
                    cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 180, 90), 1, cv2.LINE_AA)
        cv2.putText(panel, "Analyzing environmental impact...", (10, 115),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 100, 50), 1, cv2.LINE_AA)
    elif result:
        threat = result.get("threat", "UNKNOWN")
        threat_color = {
            "CRITICAL": (50, 50, 255),
            "HIGH":     (50, 50, 255),
            "MEDIUM":   (0, 165, 255),
            "LOW":      tuple(GREEN),
        }.get(threat, (150, 150, 150))

        # Threat badge
        cv2.rectangle(panel, (pw-110, 38), (pw-8, 58), threat_color, 1)
        cv2.putText(panel, f"THREAT: {threat}", (pw-106, 52),
                    cv2.FONT_HERSHEY_PLAIN, 0.9, threat_color, 1, cv2.LINE_AA)

        # Analysis text (word wrap)
        analysis = result.get("analysis", "")
        words = analysis.split()
        lines, line = [], []
        for w in words:
            line.append(w)
            if len(" ".join(line)) > 52:
                lines.append(" ".join(line[:-1]))
                line = [w]
        if line:
            lines.append(" ".join(line))

        y = 72
        for ln in lines[:3]:
            cv2.putText(panel, ln, (10, y),
                        cv2.FONT_HERSHEY_PLAIN, 0.88, (0, 200, 100), 1, cv2.LINE_AA)
            y += 16

        # Metrics row
        metrics = [
            (result.get("metric1_label","—"), result.get("metric1_value","—")),
            (result.get("metric2_label","—"), result.get("metric2_value","—")),
            (result.get("metric3_label","—"), result.get("metric3_value","—")),
        ]
        mw = (pw - 24) // 3
        for i, (mlabel, mval) in enumerate(metrics):
            mx = 8 + i * (mw + 4)
            my = 140
            cv2.rectangle(panel, (mx, my), (mx+mw, my+50), (0, 40, 20), -1)
            cv2.rectangle(panel, (mx, my), (mx+mw, my+50), (0, 80, 40), 1)
            cv2.putText(panel, mlabel[:14].upper(), (mx+4, my+13),
                        cv2.FONT_HERSHEY_PLAIN, 0.72, (0, 100, 50), 1, cv2.LINE_AA)
            cv2.putText(panel, mval, (mx+4, my+38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, tuple(GREEN), 1, cv2.LINE_AA)

        # Tags
        tags = result.get("tags", [])
        tx = 8
        ty = 205
        for tag in tags[:4]:
            tw_px = len(tag) * 7 + 10
            cv2.rectangle(panel, (tx, ty-12), (tx+tw_px, ty+4), (0, 40, 20), -1)
            cv2.rectangle(panel, (tx, ty-12), (tx+tw_px, ty+4), (0, 80, 40), 1)
            cv2.putText(panel, tag, (tx+4, ty),
                        cv2.FONT_HERSHEY_PLAIN, 0.72, (0, 160, 80), 1, cv2.LINE_AA)
            tx += tw_px + 6
            if tx > pw - 80:
                break

        # Footer
        cv2.putText(panel, "FUTUREHACKS 8", (10, ph-10),
                    cv2.FONT_HERSHEY_PLAIN, 0.75, (0, 60, 30), 1, cv2.LINE_AA)

    # Blend panel onto frame
    alpha = 0.93
    roi = img[py:py+ph, px:px+pw]
    blended = cv2.addWeighted(panel, alpha, roi, 1-alpha, 0)
    img[py:py+ph, px:px+pw] = blended

# ── Status bar ──────────────────────────────────────────────────────────────────

def draw_status(img, product_under_gaze):
    cv2.rectangle(img, (0, 0), (WINDOW_W, 44), (4, 18, 10), -1)
    cv2.line(img, (0, 44), (WINDOW_W, 44), (0, 80, 40), 1)

    # Logo
    cv2.circle(img, (14, 22), 5, tuple(GREEN), -1)
    cv2.putText(img, "IRIS", (24, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, tuple(GREEN), 2, cv2.LINE_AA)

    # Status
    if product_under_gaze:
        msg = f"LOCKING ON: {product_under_gaze.upper()}"
        color = tuple(GREEN)
    else:
        msg = "SCANNING — look at any product"
        color = (0, 140, 70)

    cv2.putText(img, msg, (100, 28), cv2.FONT_HERSHEY_PLAIN,
                1.0, color, 1, cv2.LINE_AA)

    # Right side
    cv2.putText(img, "FUTUREHACKS 8  |  BUILD THE CITY OF TOMORROW",
                (WINDOW_W - 420, 28), cv2.FONT_HERSHEY_PLAIN,
                0.85, (0, 80, 40), 1, cv2.LINE_AA)

# ── Main ─────────────────────────────────────────────────────────────────────────

def main():
    global API_KEY

    # Get API key
    if not API_KEY:
        API_KEY = input("Enter your Groq API key (gsk_...): ").strip()
    if not API_KEY:
        print("No API key provided. Exiting.")
        sys.exit(1)

    # Load background
    bg = cv2.imread(BG_PATH)
    if bg is None:
        print(f"Background not found at {BG_PATH}")
        sys.exit(1)
    bg = cv2.resize(bg, (WINDOW_W, WINDOW_H))

    print("\n=== IRIS Eye-Tracking Scanner ===")
    print("You'll now calibrate the eye tracker.")
    print("Follow the green dots on screen with your eyes.")
    print("Press ESC at any time to quit.\n")

    # Calibrate EyeTrax
    estimator = GazeEstimator()
    run_9_point_calibration(estimator)
    print("\nCalibration complete! Starting Iris...\n")

    show_hud   = False
    scan_timer = None

    while True:
        frame = bg.copy()

        # Get current gaze
        with state.lock:
            gx = int(np.clip(state.gaze_x, 0, WINDOW_W - 1))
            gy = int(np.clip(state.gaze_y, 0, WINDOW_H - 1))
            scanning    = state.scanning
            scan_result = state.scan_result
            scan_prod   = state.scan_product

        # What product is the gaze on?
        product = product_at(gx, gy)

        # Dwell logic
        now = time.time()
        if product:
            if state.dwell_product != product:
                state.dwell_product = product
                state.dwell_start   = now
            elapsed  = now - (state.dwell_start or now)
            progress = min(elapsed / DWELL_SECS, 1.0)

            if elapsed >= DWELL_SECS and not scanning and not scan_result:
                # Trigger scan
                with state.lock:
                    state.scanning    = True
                    state.scan_result = None
                show_hud = True
                threading.Thread(
                    target=scan_async, args=(product,), daemon=True
                ).start()
        else:
            state.dwell_product = None
            state.dwell_start   = None
            progress = 0.0

        # Draw status bar
        draw_status(frame, product)

        # Draw gaze reticle
        draw_gaze(frame, gx, gy, product is not None, progress)

        # Draw HUD if active
        with state.lock:
            scanning    = state.scanning
            scan_result = state.scan_result
            scan_prod   = state.scan_product

        if show_hud:
            draw_hud(frame, scan_result, scan_prod or product or "Unknown", scanning)

        # Dismiss HUD on spacebar, reset on 'r'
        cv2.imshow("Iris — Environmental Scanner", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord(' '):
            show_hud = False
            with state.lock:
                state.scan_result  = None
                state.scan_product = None
                state.scanning     = False
            state.dwell_start   = None
            state.dwell_product = None
        elif key == ord('r'):
            # Re-calibrate
            print("Re-calibrating...")
            run_9_point_calibration(estimator)
            print("Done.")

        # Update gaze from EyeTrax
        cap = getattr(main, '_cap', None)
        # (gaze update happens in gaze thread below)

    cv2.destroyAllWindows()
    estimator.close()


def gaze_thread(estimator):
    """Continuously reads webcam and updates gaze coordinates."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        features, blink = estimator.extract_features(frame)
        if features is not None and not blink:
            try:
                x, y = estimator.predict([features])[0]
                with state.lock:
                    state.gaze_x = float(x)
                    state.gaze_y = float(y)
            except Exception:
                pass
    cap.release()


if __name__ == "__main__":
    # Calibrate first, then start gaze thread
    if not API_KEY:
        API_KEY = input("Groq API key (gsk_...): ").strip()

    print("\n=== IRIS Environmental Scanner ===")
    print("Calibrating eye tracker — follow the dots with your eyes.\n")

    estimator = GazeEstimator()
    run_9_point_calibration(estimator)
    print("\nCalibration complete! Starting scanner...\n")
    print("Controls:")
    print("  SPACE — dismiss current scan")
    print("  R     — recalibrate")
    print("  ESC   — quit\n")

    # Start gaze tracking in background
    t = threading.Thread(target=gaze_thread, args=(estimator,), daemon=True)
    t.start()

    # Load background
    bg = cv2.imread(BG_PATH)
    if bg is None:
        print(f"Background image not found: {BG_PATH}")
        sys.exit(1)
    bg = cv2.resize(bg, (WINDOW_W, WINDOW_H))

    show_hud = False

    while True:
        frame = bg.copy()

        with state.lock:
            gx = int(np.clip(state.gaze_x, 0, WINDOW_W - 1))
            gy = int(np.clip(state.gaze_y, 0, WINDOW_H - 1))
            scanning    = state.scanning
            scan_result = state.scan_result
            scan_prod   = state.scan_product

        product = product_at(gx, gy)
        now     = time.time()

        # Dwell logic
        if product:
            if state.dwell_product != product:
                state.dwell_product = product
                state.dwell_start   = now
            elapsed  = now - (state.dwell_start or now)
            progress = min(elapsed / DWELL_SECS, 1.0)

            if elapsed >= DWELL_SECS and not scanning and not (scan_result and scan_prod == product):
                with state.lock:
                    state.scanning    = True
                    state.scan_result = None
                    state.scan_product = product
                show_hud = True
                threading.Thread(
                    target=scan_async, args=(product,), daemon=True
                ).start()
        else:
            state.dwell_product = None
            state.dwell_start   = None
            progress = 0.0

        # Draw everything
        draw_status(frame, product)
        draw_gaze(frame, gx, gy, product is not None, progress)

        with state.lock:
            scanning    = state.scanning
            scan_result = state.scan_result
            scan_prod   = state.scan_product

        if show_hud:
            draw_hud(frame, scan_result, scan_prod or product or "Unknown", scanning)

        cv2.imshow("Iris — Environmental Scanner", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break
        elif key == ord(' '):
            show_hud = False
            with state.lock:
                state.scan_result  = None
                state.scan_product = None
                state.scanning     = False
            state.dwell_start   = None
            state.dwell_product = None
        elif key == ord('r'):
            print("Recalibrating...")
            run_9_point_calibration(estimator)
            print("Done.")

    cv2.destroyAllWindows()
    estimator.close()
