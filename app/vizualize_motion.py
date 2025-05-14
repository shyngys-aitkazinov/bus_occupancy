#!/usr/bin/env python3
"""
Preview motion‑pixel count while scrolling through a video.
"""

import argparse
from pathlib import Path

import cv2


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=True)
    ap.add_argument("-t", "--threshold", type=int, default=5000)
    ap.add_argument("--gap", type=int, default=30)
    return ap.parse_args()


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open {args.video}")

    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)

    frame_idx, last_saved = 0, -args.gap

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(gray)
        motion_pixels = cv2.countNonZero(fgmask)

        # -------- visualisation --------
        # Convert mask to 3‑channel so we can draw coloured text
        mask_bgr = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)
        cv2.putText(
            mask_bgr,
            f"motion pixels: {motion_pixels}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        stacked = cv2.hconcat([frame, mask_bgr])
        cv2.imshow("frame | mask", stacked)

        # --- crucial: waitKey handles GUI refresh & key events ---
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):  # ESC or q to quit
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
