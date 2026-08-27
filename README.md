# Track the Game

Annotates 30s of game footage: every player marked, teams distinguished
by color, the ball highlighted, and whoever's on the ball marked
differently — under $1 and 15–25s of processing per finished video.

## How it works

- **Perception (VLM):** sparse keyframes only (every Nth frame, not all
  900), each grounded with a synthetic ruler overlay (Pillow) so the
  model reads coordinates off a scale instead of guessing. Returns
  player positions, team by kit color, and ball position as validated
  structured JSON.
- **Continuity (classical CV):** a Kalman filter per player predicts
  position frame-to-frame and corrects itself against VLM detections at
  keyframes; Hungarian assignment (IoU + color-histogram appearance cost)
  matches each keyframe's unlabeled detections to the right existing
  track; Lucas-Kanade optical flow carries positions through the frames
  between keyframes with zero extra VLM calls.
- **Rendering (Pillow):** all annotations are drawn on the original,
  ruler-free frames.

See `thinking_cap.txt` for the full design reasoning, options considered
and rejected, and known limitations.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENROUTER_API_KEY
```

## Smoke-test the pipeline (no API cost)

Generates a synthetic clip and runs the full pipeline against it using a
color-blob mock detector instead of a real VLM call — validates that
tracking, optical flow, and rendering all work mechanically before
spending real budget on real footage.

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

Key tunables live in `src/config.py` / `.env`: `KEYFRAME_INTERVAL`,
`VLM_MODEL`, tracker cost weights. These are the ablation variables
referenced in the report.

## Project layout

```
src/
  config.py        all tunables in one place
  schema.py         VLM structured-output schema + validation
  ruler_overlay.py   grounding overlay for VLM keyframes
  vlm_client.py       async OpenRouter calls, concurrency-limited
  mock_vlm.py          TEST-ONLY color-blob stand-in, no real VLM
  tracker.py            Kalman filter + Hungarian assignment
  appearance.py           color histogram for occlusion tiebreaking
  optical_flow.py          Lucas-Kanade propagation between keyframes
  renderer.py               Pillow drawing + ball-possession logic
  pipeline.py                orchestrates the above end to end
  video_io.py                 read/write video
scripts/
  make_test_clip.py   synthetic clip generator
  run_pipeline.py       CLI entrypoint
```
