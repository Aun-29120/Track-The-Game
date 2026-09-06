import av
import numpy as np
import time

def benchmark(w, h, n_frames=300):
    print(f"\n--- Benchmarking {w}x{h} (Aligned? {w%16==0 and h%16==0}) ---")
    frame_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    frame_bgr[:, :, 1] = 200  # Some color
    
    path = f"data/output/bench_{w}x{h}.mp4"
    container = av.open(path, mode='w')
    stream = container.add_stream('h264_videotoolbox', rate=30)
    stream.width = w
    stream.height = h
    stream.pix_fmt = 'yuv420p'
    
    t_convert = 0.0
    t_encode = 0.0
    
    t_start = time.time()
    for _ in range(n_frames):
        # 1. Swscale conversion
        t0 = time.time()
        av_frame = av.VideoFrame.from_ndarray(frame_bgr, format='bgr24')
        t1 = time.time()
        
        # 2. Hardware encode
        for packet in stream.encode(av_frame):
            container.mux(packet)
        t2 = time.time()
        
        t_convert += (t1 - t0)
        t_encode += (t2 - t1)
        
    # Flush
    for packet in stream.encode():
        container.mux(packet)
        
    container.close()
    
    t_total = time.time() - t_start
    print(f"Total time for {n_frames} frames: {t_total:.3f}s ({n_frames/t_total:.1f} fps)")
    print(f"  Swscale (BGR->YUV420p) time: {t_convert:.3f}s")
    print(f"  Hardware encode time:      {t_encode:.3f}s")

if __name__ == "__main__":
    benchmark(2106, 1180, 300)  # Original unaligned
    benchmark(2112, 1184, 300)  # 16-aligned padding
