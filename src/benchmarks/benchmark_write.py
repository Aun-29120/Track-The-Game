import time
import cv2
import numpy as np

def benchmark_video_write():
    frames = [np.zeros((720, 1280, 3), dtype=np.uint8) for _ in range(150)]
    fps = 30.0
    h, w = frames[0].shape[:2]
    
    codecs = ["mp4v", "avc1"]
    for codec in codecs:
        t0 = time.time()
        fourcc = cv2.VideoWriter_fourcc(*codec)
        path = f"data/output/test_{codec}.mp4"
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        
        t1 = time.time()
        for f in frames:
            writer.write(f)
        t_loop = time.time()
        
        writer.release()
        t2 = time.time()
        print(f"Codec: {codec}")
        print(f"  Init: {t1 - t0:.3f}s")
        print(f"  Write loop: {t_loop - t1:.3f}s")
        print(f"  Release: {t2 - t_loop:.3f}s")
        print(f"  Total: {t2 - t0:.3f}s")
        
benchmark_video_write()
