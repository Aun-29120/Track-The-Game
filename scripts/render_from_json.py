import sys
import os
import json
import traceback
sys.path.insert(0, os.path.abspath("src"))

from pipeline import read_all_frames, write_video
from schema import parse_vlm_response_verbose
from interpolator import interpolate_pipeline
from renderer import render_frame
from config import CFG
from ruler_overlay import RulerGeometry

print("Starting scripts/render_from_json.py", flush=True)

try:
    with open("data/output/clip2_dynamic_boundary_raw.json", "r") as f:
        raw_results = json.load(f)
        
    kf_results = {}
    for kf_str, raw_dict in raw_results.items():
        parsed, _ = parse_vlm_response_verbose(json.dumps(raw_dict))
        if parsed is not None:
            kf_results[int(kf_str)] = parsed

    frames, fps = read_all_frames("data/clips/clip2.mp4")
    print(f"Read {len(frames)} frames.", flush=True)
    
    h, w = frames[0].shape[:2]
    geo = RulerGeometry(w, h, CFG.ruler_thickness_px, CFG.ruler_scale_max)
    
    print("Interpolating...", flush=True)
    dense_map = interpolate_pipeline(kf_results, len(frames), geo)
    
    annotated = []
    print("Rendering...", flush=True)
    for i in range(len(frames)):
        out_frame = render_frame(frames[i].copy(), geo, dense_map[i])
        annotated.append(out_frame)
        
    path = "data/output/clip2_dynamic_boundary.mp4"
    print(f"Writing {path}...", flush=True)
    write_video(path, annotated, fps)
    print("DONE!", flush=True)
    print(f"File size: {os.path.getsize(path)}", flush=True)

except Exception as e:
    traceback.print_exc()
    sys.exit(1)
