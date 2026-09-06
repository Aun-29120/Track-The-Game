import cv2
import time
import sys

def benchmark_cv2(path):
    start = time.time()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Cannot open via cv2")
        return
    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        count += 1
    cap.release()
    elapsed = time.time() - start
    print(f"cv2 decode: {count} frames in {elapsed:.2f}s ({count/elapsed:.1f} fps)")

def benchmark_av(path):
    try:
        import av
    except ImportError:
        print("PyAV not installed yet.")
        return
    
    start = time.time()
    container = av.open(path)
    count = 0
    for frame in container.decode(video=0):
        img = frame.to_ndarray(format='bgr24')
        count += 1
    container.close()
    elapsed = time.time() - start
    print(f"PyAV decode: {count} frames in {elapsed:.2f}s ({count/elapsed:.1f} fps)")

if __name__ == "__main__":
    path = "data/clips/clip2.mp4"
    print(f"Benchmarking decode for {path} ...")
    benchmark_cv2(path)
    benchmark_av(path)
