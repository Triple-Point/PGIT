"""
check_video.py

AI Video Content Checker using opennsfw2

This script processes a video file (from local disk or URL stream),
extracts frames at a defined interval, and uses a pre-trained NSFW
model (opennsfw2) to detect inappropriate content in real-time.

Author: Mr Tree
Project: PGIT (Parental Guidance Image Tester)
License: MIT
Date: 2025-06-07

Notes:
- Uses open-source TensorFlow-based NSFW detector 'opennsfw2'.
- Frames are sampled at 1 frame per second by default to balance speed and accuracy.
- NSFW score threshold is configurable (default 0.5).
- Video input can be a local file path or a stream URL.
- No webcam used, per project requirements.

Dependencies:
- opennsfw2 (install via `pip install opennsfw2`)
- opencv-python
- numpy

"""

import cv2
from PIL import Image
import opennsfw2 as n2
import sys


def check_video_nsfw(video_path, frame_rate=1, threshold=0.5):
    """
    Analyze a video file or stream for inappropriate content using opennsfw2.

    Args:
        video_path (str): Path or URL to the video file/stream.
        frame_rate (int): Number of frames per second to analyze.
        threshold (float): NSFW score threshold to flag inappropriate frames.

    Returns:
        List of tuples (timestamp_sec, nsfw_score) for frames exceeding threshold.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Warning: Unable to get FPS from video, assuming 25 FPS.")
        fps = 25

    frame_interval = int(fps / frame_rate)
    if frame_interval == 0:
        frame_interval = 1

    flagged_frames = []
    frame_idx = 0

    print(f"Processing video: {video_path}")
    print(f"Video FPS: {fps}, analyzing every {frame_interval} frames (~{frame_rate} fps)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # opennsfw2 expects RGB images
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # NSFW score (0.0 to 1.0) for the frame
            pil_img = Image.fromarray(rgb_frame)
            nsfw_score = n2.predict_image(pil_img)

            timestamp_sec = frame_idx / fps

            print(f"Time {timestamp_sec:.1f}s: NSFW score = {nsfw_score:.3f}")

            if nsfw_score >= threshold:
                flagged_frames.append((timestamp_sec, nsfw_score))

        frame_idx += 1

    cap.release()
    return flagged_frames


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_video.py <video_path_or_url> [frame_rate] [threshold]")
        print("Example: python check_video.py sample_movie.mp4 1 0.5")
        sys.exit(1)

    video_file = sys.argv[1]
    frame_rate = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    flagged = check_video_nsfw(video_file, frame_rate=frame_rate, threshold=threshold)

    if flagged:
        print("\n⚠️ Inappropriate content detected at these timestamps:")
        for t, score in flagged:
            print(f" - {t:.1f} seconds (NSFW score: {score:.3f})")
    else:
        print("\n✅ No inappropriate content detected above threshold.")
