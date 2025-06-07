#!/usr/bin/env python3
"""
check_video.py – PGIT: Parental Guidance Inspection Tool

Description:
    This script scans a video file for NSFW content using the opennsfw2 model.
    Any frame flagged as NSFW is blurred and saved to a new output video.
    All other frames are copied unchanged.

Attribution:
    Developed with Open Source libraries:
        - OpenCV for video processing
        - Pillow for image format conversion
        - opennsfw2 for NSFW detection (MIT license)

Author: BiteMe / richardkilgour
License: MIT
"""

import cv2
from PIL import Image
import opennsfw2 as n2
import argparse
import os
import time

# ---- Parameters ----
NSFW_THRESHOLD = 0.7  # Tune this value for stricter/looser filtering
FRAME_SKIP = 1        # Process every frame (increase to speed up, at the cost of lower accuracy)
BLUR_KERNEL = (101, 101)

PIXELATE = True

def pixelate(image, pixel_size=20):
    h, w = image.shape[:2]
    temp = cv2.resize(image, (w//pixel_size, h//pixel_size), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)


# ---- Main Function ----
def check_video_nsfw(input_path, output_path):
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {input_path}")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing video: {input_path}")
    print(f"Output will be saved to: {output_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps}")

    frame_idx = 0
    blurred_frames = 0
    blurring_start_time = None

    processing_start_time = None # Track total processing time
    last_log_second = -1  # Track last timestamp that was logged

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not processing_start_time:
            processing_start_time= time.time()  # Track total processing time

        if frame_idx % FRAME_SKIP == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_frame)

            nsfw_score = n2.predict_image(pil_img)
            if nsfw_score >= NSFW_THRESHOLD:
                if not blurring_start_time:
                    blurring_start_time = frame_idx
                if PIXELATE:
                    frame = pixelate(frame, pixel_size=30)
                else:  # Blur NSFW frame
                    frame = cv2.GaussianBlur(frame, BLUR_KERNEL, 0)
                blurred_frames += 1
            elif blurring_start_time:
                print(f"Blurred frames {blurring_start_time} through to {frame_idx}")
                blurring_start_time = None

        # End of 10-second segment: print duration
        if int(frame_idx) % (10 * fps) == 0:
            segment_end = time.time()
            print(f"Duration to process 10s of video: {segment_end - processing_start_time:.2f} seconds")
            processing_start_time = None

        out.write(frame)
        frame_idx += 1

        if frame_idx >= 3000:
            break

    cap.release()
    out.release()
    print(f"Done. Total frames processed: {frame_idx}. Blurred frames: {blurred_frames}.")

# ---- CLI ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check video for NSFW content and censor frames.")
    parser.add_argument("input", help="Input video file path")
    parser.add_argument("output", nargs="?", help="Output video file path (default: _cleaned.mp4)", default=None)

    args = parser.parse_args()
    input_file = args.input
    output_file = args.output or os.path.splitext(input_file)[0] + "_cleaned.mp4"

    check_video_nsfw(input_file, output_file)
