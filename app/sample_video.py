import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-v", "--video", required=True)
    p.add_argument("-o", "--output-dir", default="selected_frames")
    p.add_argument("-n", "--num-frames", type=int, default=1000)
    p.add_argument("-g", "--gap-sec", type=float, default=1.0)
    p.add_argument("--threshold", type=int, default=2000)
    p.add_argument("--history", type=int, default=500)
    p.add_argument("--var-th", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(args.video)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    gap_frames = int(args.gap_sec * fps)

    fgbg = cv2.createBackgroundSubtractorMOG2(history=args.history, varThreshold=args.var_th, detectShadows=False)

    motion_counts_list = []
    print("[pass 1] scanning video …")
    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(gray)
        motion_counts_list.append(int(cv2.countNonZero(fgmask)))

    cap.release()
    motion_counts = np.array(motion_counts_list, dtype=np.int32)
    weights = np.where(motion_counts > args.threshold, motion_counts, 0)

    # select indices
    candidates = np.where(weights > 0)[0]
    sorted_idx = candidates[np.argsort(weights[candidates])[::-1]]

    selected: list[int] = []
    for idx in sorted_idx:
        if all(abs(idx - s) > gap_frames for s in selected):
            selected.append(idx)
            if len(selected) == args.num_frames:
                break
    selected.sort()

    print(f"[info] saving {len(selected)} frames -> {args.output_dir}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # second pass: save
    cap = cv2.VideoCapture(args.video)
    save_iter = iter(selected)
    next_save = next(save_iter, None)

    counter = 0
    for f_idx in tqdm(range(total_frames), desc="saving"):
        ret, frame = cap.read()
        if not ret or next_save is None:
            break
        if f_idx == next_save:
            out = Path(args.output_dir) / f"frame_{counter}.jpg"
            cv2.imwrite(str(out), frame)
            next_save = next(save_iter, None)
            counter += 1

    cap.release()
    print("Sampling done.")


if __name__ == "__main__":
    main()
