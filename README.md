# Track the Game

Annotates 30s of game footage: every player marked, teams distinguished by color, the ball highlighted, and whoever's on the ball marked differently — under $1 and 15–25s of processing per finished video.

## How it works

* **1. Input Ingestion & Sparse Keyframe Sampling:** Extracts frames at a sparse interval (`KEYFRAME_INTERVAL = 8` or `16`) to drastically reduce VLM API costs (keeping expenses under $1.00 per video) and slash total execution time.
* **2. Synthetic Ruler Grounding:** Overlays a dynamic 0–1000 integer coordinate grid (`src/ruler_overlay.py`) on selected keyframes, allowing the Vision-Language Model to read precise spatial coordinates instead of guessing raw pixels.
* **3. Asynchronous VLM Perception:** Dispatches ruled keyframe images concurrently via OpenRouter (`src/vlm_client.py`) with strict rate limiting, returning structured JSON (`src/schema.py`) containing raw bounding boxes, team kit classifications, and ball coordinates.
* **4. Multi-Gate Spatial & Pitch Filtering:** Eliminates false positives, stadium ad-board artifacts, and spectator ghosting via a strict 15-pixel edge margin and a dual-gate pitch horizon/HSV green-turf filter (`Hue: 35–85`) (`src/pipeline.py`).
* **5. Dynamic CIELAB Team Clustering & Identity Locking:** Avoids brittle static thresholds by accumulating upper-torso color samples over the first 45 frames, stripping out turf, converting to CIELAB space, and running unsupervised $k=2$ K-Means clustering (`src/team_identity.py`). Features a 5-frame lock streak to prevent mid-play team flipping and a Euclidean distance threshold ($\Delta E > 35.0$) to route referees into an "unknown" pool.
* **6. Hybrid Continuity (Kalman + KCF + Hungarian Matching):** Combines Hungarian assignment with per-player Kalman filters on keyframes (`src/tracker.py`), while Kernelized Correlation Filters (KCF) propagate player bounding boxes frame-by-frame between keyframes with high velocity stability.
* **7. Possession Logic & Rendering:** Calculates ball possession in native pixel space ($< 50\text{px}$) and renders clean FIFA-style ellipse markers directly onto original, unwarped video frames (`src/renderer.py`).

See `thinking_cap.txt` for the full design reasoning, options considered and rejected, and known limitations.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENROUTER_API_KEY

```

## Smoke-test the pipeline (no API cost)

Generates a synthetic clip and runs the full pipeline against it using a color-blob mock detector instead of a real VLM call — validates that tracking, KCF propagation, and rendering all work mechanically before spending real budget on real footage.

```bash
python scripts/make_test_clip.py --out data/clips/synthetic_test.mp4
USE_MOCK_VLM=true python scripts/run_pipeline.py \
    --in data/clips/synthetic_test.mp4 \
    --out data/output/synthetic_test_annotated.mp4

```

## Run on a real clip

```bash
python scripts/run_pipeline.py \
    --in data/clips/clip1.mp4 \
    --out data/output/clip1_annotated.mp4

```

Key tunables live in `src/config.py` / `.env`: `KEYFRAME_INTERVAL`, `VLM_MODEL`, tracker cost weights. These are the ablation variables referenced in the report.

## Project layout

```
src/
    config.py           all tunables in one place
    schema.py           VLM structured-output schema + validation
    ruler_overlay.py    grounding overlay for VLM keyframes
    vlm_client.py       async OpenRouter calls, concurrency-limited
    mock_vlm.py         TEST-ONLY color-blob stand-in, no real VLM
    tracker.py          Kalman filter + Hungarian assignment
    appearance.py       color histogram for occlusion tiebreaking
    team_identity.py    dynamic CIELAB K-Means clustering + streak locking
    renderer.py         Pillow/OpenCV drawing + ball-possession logic
    pipeline.py         orchestrates the above end to end
    video_io.py         read/write video
scripts/
    make_test_clip.py   synthetic clip generator
    run_pipeline.py     CLI entrypoint

```
