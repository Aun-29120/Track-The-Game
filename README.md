# Track the Game

Annotates 30s of game footage: every player marked, teams distinguished by color, the ball highlighted, and whoever's on the ball marked differently — under $1 and 15–25s of processing per finished video.

## How it works

* **Perception (VLM):** Sparse keyframes only (every Nth frame, not all 900), each grounded with a synthetic ruler overlay (Pillow) so the model reads coordinates off a scale instead of guessing. Returns player positions, team by kit color, and ball position as validated structured JSON.
* **Spatial & Pitch Filtering:** A strict 15-pixel edge margin drops detections near screen borders, paired with a dual-gate pitch horizon and green-turf HSV check (`Hue: 35–85`) to completely eliminate ad-board ghosting and false background rings.
* **Dynamic Team Classification:** Accumulates upper-torso color samples over the first 45 frames, strips out green turf using an HSV mask, converts crops to CIELAB space, and runs unsupervised $k=2$ K-Means clustering. Includes a 15-frame rolling majority vote, a 5-frame lock streak to prevent mid-play team flipping, and a Euclidean distance threshold ($\Delta E > 35.0$) to route referees and outliers into an "unknown" pool.
* **Continuity & Tracking:** A Kalman filter per player + Hungarian matching combined with **KCF (Kernelized Correlation Filter)** trackers to propagate player bounding boxes between keyframes with high velocity stability and robust handling of close tackles.
* **Possession & Rendering:** Native pixel-space distance checks ($< 50\text{px}$) determine ball possession, rendering clean FIFA-style ellipse markers directly onto the original, unwarped video frames.

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
