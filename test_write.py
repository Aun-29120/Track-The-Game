import cv2
import numpy as np
import time

frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(300)]
h, w = frames[0].shape[:2]

for codec in ['mp4v', 'avc1']:
    t0 = time.time()
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(f'test_{codec}.mp4', fourcc, 30.0, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    print(f"{codec}: {time.time() - t0:.2f}s")
