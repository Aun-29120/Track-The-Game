from __future__ import annotations
import cv2
import numpy as np


def read_all_frames(path: str) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def write_video(path: str, frames: list[np.ndarray], fps: float) -> None:
    if not frames:
        raise ValueError("No frames to write")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
