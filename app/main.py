import argparse
from pathlib import Path

import cv2
import supervision as sv
from ultralytics import YOLO


def build_tracker(tracker_name: str):
    """Return a tracker object given its name.
    Currently supports 'bytetrack' (via supervision) and 'sort' (via sort module).
    """
    tracker_name = tracker_name.lower()
    if tracker_name == "bytetrack":
        return sv.ByteTrack()
    elif tracker_name == "sort":
        try:
            from sort import Sort  # type: ignore
        except ImportError as e:
            raise ImportError("SORT tracker not found. Install via `pip install sort` or choose bytetrack.") from e
        return Sort()
    else:
        raise ValueError(f"Unsupported tracker: {tracker_name}. Choose 'bytetrack' or 'sort'.")


def main():
    parser = argparse.ArgumentParser(
        description="People entry/exit counter using YOLO detector and multi‑object tracking."
    )
    parser.add_argument("-v", "--video", required=True, help="Path to input video file")
    parser.add_argument("-m", "--model", default="yolov8n.pt", help="YOLO model weights (e.g. yolov8n.pt or custom)")
    parser.add_argument(
        "-t", "--tracker", default="bytetrack", choices=["bytetrack", "sort"], help="Tracking algorithm to use"
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size for YOLO")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for detections")
    parser.add_argument(
        "--line",
        nargs=4,
        type=int,
        metavar=("x1", "y1", "x2", "y2"),
        help="Custom line coordinates for counting (two points)",
    )
    parser.add_argument(
        "--start-sec", type=int, default=0, help="Start processing at N seconds into the video (default 0)"
    )
    parser.add_argument(
        "--display/--no-display",
        dest="display",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Show visualization window",
    )
    parser.add_argument("-o", "--output", help="Optional path to save annotated output video")

    args = parser.parse_args()

    # Load YOLO model
    print(f"[INFO] Loading YOLO model from {args.model} …")
    model = YOLO(args.model)

    # Build tracker
    print(f"[INFO] Initializing tracker: {args.tracker}")
    tracker = build_tracker(args.tracker)

    # Open video
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {args.video}")

    # Seek to starting second if requested
    if args.start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start_sec * 1000)
        print(f"[INFO] Starting processing at {args.start_sec:.2f}s …")

    # Read first frame (after seek)
    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to read from video at the requested start position.")

    frame_h, frame_w = first_frame.shape[:2]

    # Define counting line
    if args.line:
        (x1, y1, x2, y2) = args.line
        line_start, line_end = sv.Point(x=x1, y=y1), sv.Point(x=x2, y=y2)
    else:
        # Default: horizontal line in the middle
        line_start, line_end = sv.Point(x=0, y=frame_h // 2), sv.Point(x=frame_w, y=frame_h // 2)
        print("[INFO] No line coordinates supplied -  using horizontal mid-frame line as default.")

    line_zone = sv.LineZone(start=line_start, end=line_end)
    line_annotator = sv.LineZoneAnnotator(thickness=2)
    # Color each track with a unique color and show the id/class/confidence label
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        color_lookup=sv.ColorLookup.TRACK,  # colors by tracker_id, similar to demo screenshot
    )

    # Optional video writer
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(args.output), fourcc, cap.get(cv2.CAP_PROP_FPS), (frame_w, frame_h))
        print(f"[INFO] Saving annotated video to {args.output}")

    # Processing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Detection
        results = model(frame, imgsz=args.imgsz, conf=args.conf)
        detections = sv.Detections.from_ultralytics(results[0])

        # Filter person class for COCO‑pretrained model (class 0)
        if detections.class_id is not None:
            person_mask = detections.class_id == 1
            # Supervision Detections supports boolean indexing
            detections = detections[person_mask]

        # 2. Tracking
        if args.tracker == "bytetrack":
            tracked = tracker.update_with_detections(detections)
        else:  # sort returns ndarray [[x1,y1,x2,y2,track_id], ...]
            sort_tracks = tracker.update(detections.xyxy.cpu().numpy()) if len(detections) else []
            tracked = sv.Detections(
                xyxy=sort_tracks[:, :4] if len(sort_tracks) else [],
                tracker_id=sort_tracks[:, 4].astype(int) if len(sort_tracks) else [],
                confidence=detections.confidence[: len(sort_tracks)] if len(sort_tracks) else [],
            )

        # 3. Counting
        line_zone.trigger(tracked)

        # 4. Visualization
        # build per‑detection labels like "#7 person 0.88"
        labels = []
        for i in range(len(tracked)):
            tid = int(tracked.tracker_id[i]) if tracked.tracker_id is not None else i
            cls_id = int(tracked.class_id[i]) if tracked.class_id is not None else -1
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.0
            # Map class id to name – default to "obj" if unknown
            cls_name = model.names.get(cls_id, "obj") if hasattr(model, "names") else "obj"
            labels.append(f"#{tid} {cls_name} {conf:.2f}")

        # Draw boxes
        frame = box_annotator.annotate(scene=frame, detections=tracked)
        # Manually draw labels since older Supervision versions do not accept `labels=`
        for det_idx in range(len(tracked)):
            x1, y1, x2, y2 = tracked.xyxy[det_idx].astype(int)
            cv2.putText(
                frame,
                labels[det_idx],
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )
        frame = line_annotator.annotate(frame, line_zone)

        exits_text = f"exits: {line_zone.out_count}"
        enters_text = f"entrances: {line_zone.in_count}"
        cv2.putText(frame, exits_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, enters_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show or save
        if args.display:
            cv2.imshow("YOLO People Counter", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break
        if writer:
            writer.write(frame)

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # Final counts
    print("\n=== Summary ===")
    print(f"Total exits: {line_zone.out_count}")
    print(f"Total entrances: {line_zone.in_count}")


if __name__ == "__main__":
    main()
