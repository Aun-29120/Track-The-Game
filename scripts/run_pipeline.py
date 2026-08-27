"""
Usage:
    python scripts/run_pipeline.py --in data/clips/clip1.mp4 --out data/output/clip1_annotated.mp4

Set USE_MOCK_VLM=true in .env to smoke-test against the synthetic clip
without spending real API budget. Set it false (default) for real runs.
"""
import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from pipeline import run_pipeline  # noqa: E402
from config import CFG  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", dest="output_path", required=True)
    args = parser.parse_args()

    if not CFG.use_mock_vlm and not CFG.openrouter_api_key:
        print("ERROR: OPENROUTER_API_KEY not set (and USE_MOCK_VLM is not true). "
              "Add your key to .env or set USE_MOCK_VLM=true to smoke-test first.")
        sys.exit(1)

    stats = run_pipeline(args.input_path, args.output_path)
    print(f"\nDone.")
    print(f"  frames processed:   {stats.n_frames}")
    print(f"  keyframes sent:     {stats.n_keyframes} (interval={CFG.keyframe_interval})")
    print(f"  keyframes failed:   {stats.n_keyframes_failed}")
    print(f"  elapsed:            {stats.elapsed_s:.1f}s  (target <{CFG.target_latency_s}s, "
          f"accept <{CFG.max_latency_s}s)")
    if stats.total_cost_usd is not None:
        flag = " (partial — some keyframes didn't report cost)" if stats.cost_is_partial else ""
        under = "under" if stats.total_cost_usd < CFG.target_cost_usd else "OVER"
        print(f"  cost this video:    ${stats.total_cost_usd:.4f}{flag}  ({under} ${CFG.target_cost_usd:.2f} target)")
    else:
        print(f"  cost this video:    unavailable (mock mode, or provider didn't report cost)")
    print(f"  output written to:  {args.output_path}")


if __name__ == "__main__":
    main()
