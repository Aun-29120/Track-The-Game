import time
import cv2
import numpy as np
import sys
import os
import json
import resource

sys.path.insert(0, os.path.abspath("src"))
from video_io import read_all_frames
from renderer import render_frame
from config import CFG
from ruler_overlay import RulerGeometry
from interpolator import interpolate_pipeline

def get_memory():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)

frames, fps = read_all_frames("data/clips/clip2.mp4")
h, w = frames[0].shape[:2]
geo = RulerGeometry(w, h, CFG.ruler_thickness_px, CFG.ruler_scale_max)

with open("data/output/clip2_dynamic_boundary_raw.json", "r") as f:
    raw_results = json.load(f)
from schema import parse_vlm_response_verbose
kf_results = {}
for kf_str, raw_dict in raw_results.items():
    parsed, _ = parse_vlm_response_verbose(json.dumps(raw_dict))
    if parsed: kf_results[int(kf_str)] = parsed

dense_map = interpolate_pipeline(kf_results, len(frames), geo)

print("\n--- OLD APPROACH (Buffered) ---")
t0 = time.time()
annotated_frames = []
t_render_start = time.time()
for i in range(len(frames)):
    out_frame = frames[i].copy()
    snapshot = dense_map.get(i, {"players": [], "ball": None})
    out_frame = render_frame(out_frame, geo, snapshot)
    annotated_frames.append(out_frame)
t_render = time.time() - t_render_start

t_write_start = time.time()
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter("data/output/test_old.mp4", fourcc, fps, (w, h))
for f in annotated_frames:
    writer.write(f)
writer.release()
t_write = time.time() - t_write_start

print(f"Peak Memory: {get_memory():.1f} MB")
print(f"Render time: {t_render:.3f}s")
print(f"Write time: {t_write:.3f}s")
print(f"Total Old: {t_render + t_write:.3f}s")
