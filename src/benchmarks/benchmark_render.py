import time
import cv2
import numpy as np
import sys
import os
import json
sys.path.insert(0, os.path.abspath("src"))
from video_io import read_all_frames
from renderer import render_frame
from config import CFG
from ruler_overlay import RulerGeometry
from interpolator import interpolate_pipeline

frames, fps = read_all_frames("data/clips/clip2.mp4")
h, w = frames[0].shape[:2]
geo = RulerGeometry(w, h, CFG.ruler_thickness_px, CFG.ruler_scale_max)
print(f"Frame shape: {h}x{w}")

with open("data/output/clip2_dynamic_boundary_raw.json", "r") as f:
    raw_results = json.load(f)
from schema import parse_vlm_response_verbose
kf_results = {}
for kf_str, raw_dict in raw_results.items():
    parsed, _ = parse_vlm_response_verbose(json.dumps(raw_dict))
    if parsed: kf_results[int(kf_str)] = parsed

dense_map = interpolate_pipeline(kf_results, len(frames), geo)

# Benchmark Rendering
t0 = time.time()
annotated = []
for i in range(len(frames)):
    out_frame = render_frame(frames[i].copy(), geo, dense_map[i])
    annotated.append(out_frame)
t_render = time.time() - t0
print(f"Render time: {t_render:.3f}s for {len(frames)} frames")

# Benchmark Writing
t1 = time.time()
fourcc = cv2.VideoWriter_fourcc(*"avc1")
writer = cv2.VideoWriter("data/output/test_avc1.mp4", fourcc, fps, (w, h))
for f in annotated:
    writer.write(f)
writer.release()
t_write = time.time() - t1
print(f"Write time (avc1): {t_write:.3f}s")

t1 = time.time()
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter("data/output/test_mp4v.mp4", fourcc, fps, (w, h))
for f in annotated:
    writer.write(f)
writer.release()
t_write = time.time() - t1
print(f"Write time (mp4v): {t_write:.3f}s")

