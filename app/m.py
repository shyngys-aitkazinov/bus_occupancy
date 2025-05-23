import argparse
from pathlib import Path

import cv2
import supervision as sv
from ultralytics import YOLO


def build_tracker(fps: int) -> sv.ByteTrack:
    """Return a ByteTrack instance (only supported tracker)."""
    return sv.ByteTrack(frame_rate=fps)


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="People entry/exit counter with YOLO + ByteTrack tracking & Supervision visualization.",
    )
    p.add_argument("-v", "--video", required=True)
    p.add_argument("-m", "--model", default="yolov8n.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--line", nargs=4, type=int, metavar=("x1", "y1", "x2", "y2"))
    p.add_argument("--start-sec", type=int, default=0)

    # visual/logic flags
    p.add_argument("--smooth", type=int, default=5, help="Detections smoother window length (0 disables)")
    p.add_argument(
        "--trace/--no-trace",
        dest="trace",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Draw trajectory tails for each track",
    )
    p.add_argument(
        "--label/--no-label",
        dest="label",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Overlay #id + conf text above boxes",
    )
    p.add_argument("--style", choices=["box", "round", "mask"], default="box", help="Bounding-box visual style")

    p.add_argument("--display/--no-display", dest="display", default=True, action=argparse.BooleanOptionalAction)
    p.add_argument("--quit-key", default="q", help="Key to exit GUI window")
    p.add_argument("-o", "--output", help="Path to save annotated video")
    return p.parse_args()


def main() -> None:
    args = get_args()

    # video info / fps
    video_info = sv.VideoInfo.from_video_path(args.video)
    cap = cv2.VideoCapture(args.video)
    start_frame = int(args.start_sec * video_info.fps)
    print(f"[INFO] seek to {args.start_sec:.2f}s")
    ret, first = cap.read()
    if not ret:
        raise RuntimeError("Cannot read first frame after seek")
    h, w = first.shape[:2]

    # components
    model = YOLO(args.model)
    tracker = build_tracker(video_info.fps)
    smoother = sv.DetectionsSmoother(length=args.smooth) if args.smooth > 0 else None

    # annotators
    if args.style == "round":
        box_annot = sv.RoundBoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.TRACK)
    elif args.style == "mask":
        box_annot = sv.ColorAnnotator(alpha_t=0.4, color_lookup=sv.ColorLookup.TRACK)
    else:
        box_annot = sv.BoxAnnotator(thickness=2, color_lookup=sv.ColorLookup.TRACK)

    label_annot = sv.LabelAnnotator(color=sv.Color.WHITE) if args.label else None
    trace_annot = sv.TraceAnnotator(color_lookup=sv.ColorLookup.TRACK) if args.trace else None

    # counting line
    if args.line:
        x1, y1, x2, y2 = args.line
        p1, p2 = sv.Point(x1, y1), sv.Point(x2, y2)
    else:
        p1, p2 = sv.Point(0, h // 2), sv.Point(w, h // 2)
    line_zone = sv.LineZone(start=p1, end=p2)
    line_draw = sv.LineZoneAnnotator(thickness=2)

    # writer
    writer = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        writer = sv.VideoSink(args.output, video_info)
        print(f"[INFO] writing → {args.output}")

    quit_key = ord(args.quit_key.lower())

    for frame in sv.get_video_frames_generator(args.video, start_seconds=args.start_sec):
        results = model(frame, imgsz=args.imgsz, conf=args.conf)
        det = sv.Detections.from_ultralytics(results[0])
        if det.class_id is not None:
            det = det[det.class_id == 0]  # person / head class

        det = tracker.update_with_detections(det)
        if smoother:
            det = smoother.update_with_detections(det)

        line_zone.trigger(det)

        # labels
        labels = [f"#{int(det.tracker_id[i])} {det.confidence[i]:.2f}" for i in range(len(det))] if args.label else None

        frame = box_annot.annotate(frame, det)
        if label_annot and labels:
            frame = label_annot.annotate(frame, det, labels=labels)
        if trace_annot:
            frame = trace_annot.annotate(frame, det)
        frame = line_draw.annotate(frame, line_zone)
        cv2.putText(frame, f"exits: {line_zone.out_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"entrances: {line_zone.in_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        if args.display:
            cv2.imshow("People Counter", frame)
            key = cv2.waitKey(1) & 0xFF
            if (
                key in (27, quit_key, quit_key - 32)
                or cv2.getWindowProperty("People Counter", cv2.WND_PROP_VISIBLE) < 1
            ):
                break
        if writer:
            writer.write_frame(frame)

    cap.release()
    if writer:
        writer.close()
    cv2.destroyAllWindows()

    print("\n=== Summary ===")
    print(f"Total exits: {line_zone.out_count} | Total entrances: {line_zone.in_count}")


if __name__ == "__main__":
    main()
