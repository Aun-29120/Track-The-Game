"""
Generates a short synthetic clip: colored circles moving on a green
"pitch," two teams by color, one ball, deliberately including a couple of
crossing paths so the occlusion/appearance-tiebreak logic actually gets
exercised. This is NOT a substitute for the five real clips the brief
requires -- it exists only to validate the pipeline mechanically
(does it run, does identity survive, does the output video look sane)
before spending API budget on real footage.

Usage:
    python scripts/make_test_clip.py --out data/clips/synthetic_test.mp4
"""
import argparse
import numpy as np
import cv2

W, H = 960, 540
FPS = 30
DURATION_S = 10  # shorter than the real 30s requirement, enough to smoke-test
N_FRAMES = FPS * DURATION_S

TEAM_A_BGR = (190, 210, 0)
TEAM_B_BGR = (60, 60, 230)
BALL_BGR = (30, 200, 250)
PITCH_BGR = (40, 110, 40)


def make_clip(out_path: str) -> None:
    rng = np.random.default_rng(7)

    n_a, n_b = 4, 4
    pos_a = rng.uniform([100, 100], [W - 100, H - 100], size=(n_a, 2))
    pos_b = rng.uniform([100, 100], [W - 100, H - 100], size=(n_b, 2))
    vel_a = rng.uniform(-2, 2, size=(n_a, 2))
    vel_b = rng.uniform(-2, 2, size=(n_b, 2))
    ball_pos = np.array([W / 2, H / 2], dtype=float)
    ball_vel = rng.uniform(-4, 4, size=2)

    # force one deliberate crossing between an A and a B player around the midpoint
    pos_a[0] = [200, H / 2 - 5]
    pos_b[0] = [W - 200, H / 2 + 5]
    vel_a[0] = [(W - 400) / (N_FRAMES * 0.5), 0]
    vel_b[0] = [-(W - 400) / (N_FRAMES * 0.5), 0]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, FPS, (W, H))

    for _frame_idx in range(N_FRAMES):
        img = np.full((H, W, 3), PITCH_BGR, dtype=np.uint8)

        for pos, vel in ((pos_a, vel_a), (pos_b, vel_b)):
            pos += vel
            for i in range(len(pos)):
                for d in range(2):
                    limit = W if d == 0 else H
                    if pos[i, d] < 20 or pos[i, d] > limit - 20:
                        vel[i, d] *= -1
                        pos[i, d] = np.clip(pos[i, d], 20, limit - 20)

        ball_pos += ball_vel
        for d in range(2):
            limit = W if d == 0 else H
            if ball_pos[d] < 15 or ball_pos[d] > limit - 15:
                ball_vel[d] *= -1
                ball_pos[d] = np.clip(ball_pos[d], 15, limit - 15)

        for p in pos_a:
            cv2.circle(img, (int(p[0]), int(p[1])), 14, TEAM_A_BGR, -1)
        for p in pos_b:
            cv2.circle(img, (int(p[0]), int(p[1])), 14, TEAM_B_BGR, -1)
        cv2.circle(img, (int(ball_pos[0]), int(ball_pos[1])), 7, BALL_BGR, -1)

        writer.write(img)

    writer.release()
    print(f"Wrote {N_FRAMES} frames ({DURATION_S}s @ {FPS}fps) to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/clips/synthetic_test.mp4")
    args = parser.parse_args()
    make_clip(args.out)
