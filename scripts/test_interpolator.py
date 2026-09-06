import sys
import os
import json
sys.path.insert(0, os.path.abspath("src"))

from schema import parse_vlm_response_verbose
from interpolator import interpolate_pipeline
from config import CFG
from ruler_overlay import RulerGeometry
import numpy as np

print("Starting interpolator test", flush=True)

with open("data/output/clip2_dynamic_boundary_raw.json", "r") as f:
    raw_results = json.load(f)

kf_results = {}
for kf_str, raw_dict in raw_results.items():
    parsed, _ = parse_vlm_response_verbose(json.dumps(raw_dict))
    if parsed is not None:
        kf_results[int(kf_str)] = parsed

print(f"Loaded {len(kf_results)} kfs", flush=True)

class MockGeo:
    def ruler_to_orig_px(self, x, y):
        return x, y

geo = MockGeo()
try:
    dense_map = interpolate_pipeline(kf_results, 381, geo)
    print("Done interpolating!", flush=True)
except Exception as e:
    import traceback
    traceback.print_exc()
