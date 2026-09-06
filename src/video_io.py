from __future__ import annotations
import cv2
import numpy as np


def read_all_frames(path: str, target_fps: float = 30.0) -> tuple[list[np.ndarray], float]:
    import av
    import time
    import logging
    logger = logging.getLogger(__name__)
    
    # Detect display rotation: cv2 auto-applies rotation metadata, PyAV does not.
    # Compare cv2's first frame shape to determine if rotation is needed.
    cap = cv2.VideoCapture(path)
    ok, cv2_frame = cap.read()
    cap.release()
    cv2_h, cv2_w = cv2_frame.shape[:2] if ok else (0, 0)
    
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    
    source_frames = []
    source_times = []
    needs_rotation = False
    
    t_decode_start = time.time()
    for frame in container.decode(stream):
        t = frame.time
        if t is None:
            continue
            
        img = frame.to_ndarray(format='bgr24')
        
        # On first frame, detect if rotation is needed
        if not source_frames and ok:
            raw_h, raw_w = img.shape[:2]
            if raw_h == cv2_w and raw_w == cv2_h:
                needs_rotation = True
        
        if needs_rotation:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        source_frames.append(img)
        source_times.append(t)
        
    container.close()
    t_decode_end = time.time()
    
    if not source_frames:
        raise ValueError(f"No valid frames found in {path}")
    
    logger.info(f"PyAV decode: {len(source_frames)} source frames in {t_decode_end - t_decode_start:.2f}s"
                f"{' (rotated)' if needs_rotation else ''}")
        
    start_time = source_times[0]
    end_time = source_times[-1]
    duration = end_time - start_time
    
    num_output_frames = int(round(duration * target_fps)) + 1
    
    t_resample_start = time.time()
    resampled_frames = []
    source_idx = 0
    num_source = len(source_frames)
    
    last_used_idx = -1
    for i in range(num_output_frames):
        target_t = start_time + i / target_fps
        
        while source_idx < num_source - 1:
            dist_current = abs(source_times[source_idx] - target_t)
            dist_next = abs(source_times[source_idx + 1] - target_t)
            if dist_next <= dist_current:
                source_idx += 1
            else:
                break
                
        if source_idx == last_used_idx:
            # Duplicate frame, we MUST copy so renderer doesn't draw twice on the same memory
            resampled_frames.append(source_frames[source_idx].copy())
        else:
            # First time using this frame, we can just take it
            resampled_frames.append(source_frames[source_idx])
            last_used_idx = source_idx
            
    # Free the unused source frames explicitly to drop peak memory
    source_frames.clear()
    t_resample_end = time.time()
    
    logger.info(f"VFR resample: {num_source} -> {num_output_frames} frames at {target_fps}fps "
                f"in {t_resample_end - t_resample_start:.2f}s (duration={duration:.2f}s)")
        
    return resampled_frames, target_fps


def write_video_stream(path: str, frames: list[np.ndarray], dense_map: dict, geo, fps: float) -> None:
    from renderer import render_frame
    import time
    import logging
    import av
    
    logger = logging.getLogger(__name__)
    
    if not frames:
        raise ValueError("No frames to write")
    h, w = frames[0].shape[:2]
    
    t_render_total = 0.0
    t_encode_total = 0.0
    
    container = av.open(path, mode='w')
    try:
        stream = container.add_stream('h264_videotoolbox', rate=int(round(fps)))
        stream.width = w
        stream.height = h
        stream.pix_fmt = 'yuv420p'
    except Exception as e:
        container.close()
        raise RuntimeError(f"Failed to initialize h264_videotoolbox encoder via PyAV: {e}")
    
    try:
        for i in range(len(frames)):
            t0 = time.time()
            out_frame = render_frame(frames[i], geo, dense_map[i])
            t1 = time.time()
            
            # Let PyAV automatically convert from bgr24 to the stream's yuv420p
            av_frame = av.VideoFrame.from_ndarray(out_frame, format='bgr24')
            for packet in stream.encode(av_frame):
                container.mux(packet)
            
            t2 = time.time()
            
            t_render_total += (t1 - t0)
            t_encode_total += (t2 - t1)
            
        # Flush the encoder
        t1 = time.time()
        for packet in stream.encode():
            container.mux(packet)
        t_encode_total += (time.time() - t1)
        
    except Exception as e:
        raise RuntimeError(f"Error during video encoding: {e}")
    finally:
        container.close()
    
    logger.info(f"Render time: {t_render_total:.2f}s")
    logger.info(f"Video write/encode time: {t_encode_total:.2f}s")
