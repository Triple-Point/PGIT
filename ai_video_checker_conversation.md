# AI Video Checker – Project Planning Conversation

**Participants**: Mr Tree (BiteMe on GitHub), ChatGPT (OpenAI)  
**Date**: June 2025  
**Topic**: Designing a real-time AI app to detect inappropriate content in video files for use in a school classroom environment.

---

## Project Goals

- Audience: Children aged 10–12
- Detect inappropriate video content in real-time.
- Must run on a home PC.
- Input source: **Pre-recorded movie files** or **video streams**, NOT webcams.
- Must use pre-trained models to ensure students are never exposed to NSFW content.
- Entire stack must be **open-source** and **Python-based**.

---

## Final Script – `check_video.py`

```python
"""
check_video.py
--------------

This script scans a video file for inappropriate content (NSFW) using a pre-trained model.
It is designed for educational use in a school AI project for children aged 10–12.

⚠️ Important Notes:
- No inappropriate content is ever shown to students.
- The app analyzes video files (not webcam input) in real time or near-real time.
- It uses a pre-trained, open-source model (`open_nsfw2`) to classify frames.
- Every Nth frame is checked to reduce processing load.
- The video is not altered or saved—only warnings are printed to the console.

Created by BiteMe (https://github.com/BiteMe)
AI-assisted with ChatGPT by OpenAI
License: MIT or any compatible open-source license of your choice.

Dependencies (see requirements.txt):
- opencv-python
- torch
- torchvision
- open-nsfw2
- (optional) pytube for YouTube support (not used in this file)

Setup Instructions:
1. Install dependencies:
   $ pip install -r requirements.txt

2. Download the model weights:
   - Visit https://github.com/EBazarov/open_nsfw2
   - Download `nsfw_model.pt`
   - Save it as: `./open_nsfw2/nsfw_model.pt`

3. Place a sample video in the project directory, e.g., `sample_movie.mp4`.

Usage:
$ python check_video.py

You can quit during playback with the 'q' key.
"""

import cv2
import os
from nsfw_detector import predict

# -------------------- Settings --------------------

VIDEO_PATH = "sample_movie.mp4"  # Replace with the path to your video file
FRAME_SKIP = 30                  # Check every Nth frame (~1/sec for 30fps video)
TEMP_FRAME = "temp_frame.jpg"    # Temp file used to classify frames
THRESHOLD = 0.5                  # Probability above which we warn about NSFW

# -------------------- Load Model --------------------

print("[INFO] Loading NSFW detection model...")
model = predict.load_model("open_nsfw2/nsfw_model.pt")
print("[INFO] Model loaded.")

# -------------------- Open Video --------------------

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print(f"[ERROR] Cannot open video file: {VIDEO_PATH}")
    exit()

frame_count = 0
print(f"[INFO] Starting video analysis on '{VIDEO_PATH}'...")

# -------------------- Main Loop --------------------

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Only analyze every Nth frame
    if frame_count % FRAME_SKIP == 0:
        # Save current frame as an image
        cv2.imwrite(TEMP_FRAME, frame)

        # Classify the image
        result = predict.classify(model, [TEMP_FRAME])
        scores = result[TEMP_FRAME]

        # Combine NSFW categories
        score = scores.get("porn", 0) + scores.get("sexy", 0) + scores.get("hentai", 0)

        if score >= THRESHOLD:
            print(f"⚠️ NSFW content detected at frame {frame_count} (Score: {score:.2f})")

    # Optionally show the video (only for teacher use)
    cv2.imshow('Analyzing Video...', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------- Cleanup --------------------

cap.release()
cv2.destroyAllWindows()

if os.path.exists(TEMP_FRAME):
    os.remove(TEMP_FRAME)

print("[INFO] Video analysis complete.")
```

---

## Summary of Key Steps

1. Use OpenCV to read a pre-recorded video file.
2. Every 30th frame is saved temporarily and analyzed.
3. Use `open_nsfw2` (PyTorch-based) to classify the frame.
4. If NSFW probability exceeds threshold (0.5), print a warning.
5. Clean up all temporary files afterward.
6. No NSFW content is displayed or stored.

---

## License

MIT or any OSI-approved license of your choice.  
All components are open source.
