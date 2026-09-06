# Track the Game

**Author:** Muhammad Aun Haider Bilgrami

A fully automated video processing pipeline that annotates 30-second sports clips with per-player team markers, a ball highlight, and a dynamic possession indicator. Developed for the Zeta Solutions Internship Programme, this system bypasses classical computer vision heuristics and relies entirely on Vision-Language Models (VLMs) for visual perception.

## System Architecture

The pipeline processes video in five sequential stages to balance speed, cost, and tracking accuracy:

* **Video Decoding**: Uses `PyAV` with multithreading (`stream.thread_type = "AUTO"`) for high-speed extraction, automatically correcting rotation metadata and resampling the footage to exactly 30.0 fps.
* **Spatial Grounding**: A synthetic 40-pixel white band is added to the top and left edges of the frame. A ruler scaling from 0 to 1000 is drawn onto these borders, allowing the VLM to read exact spatial coordinates rather than hallucinating pixel locations.
* **Sparse VLM Perception**: To maintain cost limits, only keyframes (every 8 or 16 frames) are analyzed. Images are downscaled to a maximum of 1280 pixels and dispatched concurrently (up to 64 simultaneous requests) via `httpx.AsyncClient` to the OpenRouter API.
* **Temporal Interpolation**: A Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) logically links independent keyframe detections into continuous tracks based on spatial proximity, jersey number rewards, and team color penalties. A monotonic spline (`PchipInterpolator`) smoothly calculates movement during the skipped frames.
* **Hardware Rendering**: OpenCV natively renders team-colored ellipses at the players' feet. Possession is assigned to the nearest player within a 50-pixel radius of the ball, indicated by an additional white ring. The final video is encoded using macOS Apple Silicon hardware acceleration (`h264_videotoolbox`).

## Setup & Installation

Create a `.env` file in the root directory to define your API credentials and configuration parameters:

```env
OPENROUTER_API_KEY="sk-or-v1-..."
VLM_MODEL="google/gemini-3.5-flash-lite"
MAX_CONCURRENT_VLM_CALLS=64
KEYFRAME_INTERVAL=8

```

*Note: To run a dry-test without spending real API budget, you can set `USE_MOCK_VLM=true` to test the pipeline logic via a synthetic color-blob CV script.*

## Usage

Execute the pipeline through the central command-line interface by providing the input and output file paths:

```bash
python scripts/run_pipeline.py --in data/clips/clip1.mp4 --out data/output/clip1_annotated.mp4

```

## Benchmarks & Performance

The system was evaluated against strict project constraints: maintaining a budget of under $1.00 per finished video while ensuring 100% data integrity and robust tracking across diverse clips.

| Clip | Frames Processed | Keyframes Sent | Keyframes Failed | Interval | Elapsed (s) | Cost ($) |
| --- | --- | --- | --- | --- | --- | --- |
| Clip 1 | 892 | 112 | 0 | 8 | 49.7 | 0.3307 |
| Clip 2 | 919 | 115 | 0 | 8 | 56.8 | 0.4499 |
| Clip 3 | 948 | 119 | 0 | 8 | 52.8 | 0.4323 |
| Clip 4 | 907 | 114 | 0 | 8 | 51.7 | 0.3854 |
| Clip 5 | 901 | 57 | 0 | 16 | 62.2 | 0.1773 |

The pipeline successfully meets all core operational goals:

* **Cost Efficiency**: Consistently operates well below the $1.00 target, averaging between $0.33 and $0.45 per video at interval 8, and dropping to $0.17 at interval 16.
* **High Reliability**: Achieved a 0% keyframe failure rate across all production runs, demonstrating robust schema validation, resilient error handling, and stable OpenRouter API concurrency.
* **Visual Precision**: Combines concurrent VLM perception with advanced temporal mathematics (Hungarian matching and monotonic spline interpolation) to eliminate identity flickering and deliver seamless FIFA-style match annotations.
