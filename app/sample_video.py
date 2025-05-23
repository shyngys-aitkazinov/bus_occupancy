import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Sample N frames from a fixed‑camera video either\n"
        "• weighted by motion magnitude (MOG2)\n"
        "• or uniform‑in‑time probabilistic (one per chunk).",
    )
    p.add_argument("-v", "--video", required=True, help="Input video file path")
    p.add_argument("-o", "--output-dir", default="selected_frames", help="Folder for saved JPGs")
    p.add_argument("-n", "--num-frames", type=int, default=1000, help="Number of frames to sample")
    p.add_argument("--gap-sec", type=float, default=1.0, help="Minimum seconds between two saved frames")
    p.add_argument(
        "--strategy",
        choices=["motion", "uniform"],
        default="motion",
        help="Sampling strategy: motion‑weighted or uniform‑probabilistic",
    )
    # motion parameters
    p.add_argument(
        "--threshold",
        type=int,
        default=2000,
        help="Ignore frames with <= threshold motion pixels (motion strategy only)",
    )
    p.add_argument("--history", type=int, default=500, help="MOG2 history (motion strategy only)")
    p.add_argument("--var-th", type=int, default=25, help="MOG2 variance threshold (motion strategy only)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return p.parse_args()


# -------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------


def greedy_gap_select(candidates: list[int], k: int, gap: int) -> list[int]:
    """Greedy pick up to k indices from `candidates` (already scored/sorted)
    such that any two chosen indices differ by > `gap` frames."""
    chosen: list[int] = []
    for idx in candidates:
        if all(abs(idx - c) > gap for c in chosen):
            chosen.append(idx)
            if len(chosen) == k:
                break
    return sorted(chosen)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(args.video)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    gap_frames = int(args.gap_sec * fps)

    if args.strategy == "motion":
        fgbg = cv2.createBackgroundSubtractorMOG2(history=args.history, varThreshold=args.var_th, detectShadows=False)
        motion = np.zeros(total_frames, dtype=np.int32)
        print("[pass 1] measuring motion …")
        for i in tqdm(range(total_frames)):
            ret, frame = cap.read()
            if not ret:
                total_frames = i  # video shorter than metadata
                motion = motion[:total_frames].reshape(-1)
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            motion[i] = cv2.countNonZero(fgbg.apply(gray))
        cap.release()

        weights = np.where(motion > args.threshold, motion, 0)
        nonzero_idx = np.where(weights > 0)[0]
        if len(nonzero_idx) == 0:
            print("No significant motion detected; exiting.")
            return
        sorted_idx = nonzero_idx[np.argsort(weights[nonzero_idx])[::-1]]  # high→low
        selected = greedy_gap_select(sorted_idx.tolist(), args.num_frames, gap_frames)

    else:  # uniform strategy
        cap.release()
        chunk = total_frames / args.num_frames
        rand_offsets = np.random.uniform(0, chunk, size=args.num_frames)
        base_idx = (np.arange(args.num_frames) * chunk).astype(int)
        selected = (base_idx + rand_offsets).astype(int).tolist()
        selected = [min(idx, total_frames - 1) for idx in selected]  # clamp
        selected = greedy_gap_select(sorted(selected), args.num_frames, gap_frames)

    if not selected:
        print("No frames selected.")
        return
    print(f"[info] saving {len(selected)} / {args.num_frames} frames → {args.output_dir}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    save_iter = iter(selected)
    next_save = next(save_iter, None)

    for f_idx in tqdm(range(total_frames), desc="saving"):
        ret, frame = cap.read()
        if not ret or next_save is None:
            break
        if f_idx == next_save:
            out_path = Path(args.output_dir) / f"frame_{f_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            next_save = next(save_iter, None)

    cap.release()
    print("Sampling completed.")


if __name__ == "__main__":
    main()
