#!/usr/bin/env python3
"""
Overlay mediapipe_holistic_minimal_27 pose keypoints on the corresponding MP4.

Usage:
    python scripts/visualize_pose.py --pkl path/to/pose.pkl --video path/to/video.mp4
    python scripts/visualize_pose.py --pkl path/to/pose.pkl --video path/to/video.mp4 --output out.mp4
    python scripts/visualize_pose.py --pkl path/to/pose.pkl --video path/to/video.mp4 --no-video
"""

import argparse
import pickle
import numpy as np
import cv2
from pathlib import Path

# ── Preset: indices into the full 75-point holistic skeleton ─────────────────
PRESET_INDICES = [0, 2, 5, 11, 12, 13, 14, 33, 37, 38, 41, 42, 45, 46,
                  49, 50, 53, 54, 58, 59, 62, 63, 66, 67, 70, 71, 74]

# Selected-space node labels (for reference)
# 0:nose  1:l_eye  2:r_eye  3:l_shoulder  4:r_shoulder  5:l_elbow  6:r_elbow
# 7:l_wrist  8-16:left hand fingers
# 17:r_wrist  18-26:right hand fingers

# ── Skeleton edges (in selected-space indices, from graph config) ─────────────
FACE_EDGES   = [(1, 0), (2, 0)]
BODY_EDGES   = [(0, 3), (0, 4), (3, 5), (4, 6), (5, 7), (6, 17)]
LHAND_EDGES  = [(7, 8), (7, 9), (9, 10), (7, 11), (11, 12),
                (7, 13), (13, 14), (7, 15), (15, 16)]
RHAND_EDGES  = [(17, 18), (17, 19), (19, 20), (17, 21), (21, 22),
                (17, 23), (23, 24), (17, 25), (25, 26)]

# BGR colours
COL_FACE  = (255, 220,   0)   # cyan-yellow
COL_BODY  = ( 50, 205,  50)   # lime green
COL_LHAND = (  0, 165, 255)   # orange
COL_RHAND = (  0,   0, 220)   # red
COL_KP    = (255, 255, 255)   # white keypoint dots

EDGE_GROUPS = [
    (FACE_EDGES,  COL_FACE),
    (BODY_EDGES,  COL_BODY),
    (LHAND_EDGES, COL_LHAND),
    (RHAND_EDGES, COL_RHAND),
]


def load_pkl(pkl_path: str):
    """Load pose pkl; returns keypoints (T, V_full, C) and confidences (T, V_full)."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    kps = np.array(data["keypoints"], dtype=np.float32)   # (T, V, C)
    conf = np.array(data["confidences"], dtype=np.float32) # (T, V)
    return kps, conf


def select_keypoints(kps: np.ndarray) -> np.ndarray:
    """Apply mediapipe_holistic_minimal_27 preset selection. (T, V_full, C) -> (T, 27, C)"""
    return kps[:, PRESET_INDICES, :]


def draw_frame(frame: np.ndarray, kps_frame: np.ndarray, conf_frame: np.ndarray) -> np.ndarray:
    """
    Draw skeleton overlay on a single BGR frame.

    Args:
        frame:      (H, W, 3) BGR image
        kps_frame:  (27, C) normalised keypoints for this frame [0, 1]
        conf_frame: (27,) confidence scores (from full set, already selected)
    """
    H, W = frame.shape[:2]
    out = frame.copy()

    # Convert normalised coords to pixels
    pts = np.stack([kps_frame[:, 0] * W, kps_frame[:, 1] * H], axis=1).astype(int)

    # Draw edges
    for edges, colour in EDGE_GROUPS:
        for a, b in edges:
            if conf_frame[a] > 0 and conf_frame[b] > 0:
                cv2.line(out, tuple(pts[a]), tuple(pts[b]), colour, 2, cv2.LINE_AA)

    # Draw keypoints
    for i, (x, y) in enumerate(pts):
        if conf_frame[i] > 0:
            cv2.circle(out, (x, y), 4, COL_KP, -1, cv2.LINE_AA)
            cv2.circle(out, (x, y), 4, (0, 0, 0), 1, cv2.LINE_AA)  # thin black outline

    return out


def process(pkl_path: str, video_path: str, output_path: str | None,
            no_video: bool, fps_override: float | None):

    # ── Load pose ─────────────────────────────────────────────────────────────
    kps_full, conf_full = load_pkl(pkl_path)
    T_pkl = kps_full.shape[0]
    kps27  = select_keypoints(kps_full)   # (T, 27, C)
    conf27 = conf_full[:, PRESET_INDICES] # (T, 27)

    if no_video:
        # Render on blank frames
        W, H = 640, 480
        fps = fps_override or 25.0
        frames = [np.zeros((H, W, 3), dtype=np.uint8)] * T_pkl
    else:
        # ── Load video ────────────────────────────────────────────────────────
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps   = fps_override or cap.get(cv2.CAP_PROP_FPS) or 25.0
        W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        T_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frames = []
        while True:
            ret, f = cap.read()
            if not ret:
                break
            frames.append(f)
        cap.release()

        if len(frames) != T_pkl:
            print(f"[warn] video has {len(frames)} frames, pkl has {T_pkl}. "
                  f"Aligning to min({len(frames)}, {T_pkl}).")
            T = min(len(frames), T_pkl)
            frames  = frames[:T]
            kps27   = kps27[:T]
            conf27  = conf27[:T]

    T = len(frames)

    if output_path:
        # ── Write annotated video ─────────────────────────────────────────────
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
        for t in range(T):
            ann = draw_frame(frames[t], kps27[t], conf27[t])
            writer.write(ann)
        writer.release()
        print(f"Saved to {output_path}")

    else:
        # ── Interactive display ───────────────────────────────────────────────
        delay = max(1, int(1000 / fps))
        print("Press [space] to pause/resume, [q] or [ESC] to quit, [←/→] to step frames.")
        t = 0
        paused = False
        while True:
            ann = draw_frame(frames[t], kps27[t], conf27[t])
            cv2.putText(ann, f"frame {t}/{T-1}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.imshow("Pose overlay", ann)

            key = cv2.waitKey(0 if paused else delay) & 0xFF
            if key in (ord('q'), 27):   # q or ESC
                break
            elif key == ord(' '):
                paused = not paused
            elif key == 81 or key == ord('a'):  # left arrow
                t = max(0, t - 1)
                paused = True
            elif key == 83 or key == ord('d'):  # right arrow
                t = min(T - 1, t + 1)
                paused = True
            elif not paused:
                t = (t + 1) % T

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Overlay mediapipe_holistic_minimal_27 pose on an MP4 video."
    )
    parser.add_argument("--pkl",      required=True,  help="Path to pose .pkl file")
    parser.add_argument("--video",    default=None,   help="Path to corresponding .mp4 (omit to render on black)")
    parser.add_argument("--output",   default=None,   help="Save annotated video to this path instead of displaying")
    parser.add_argument("--no-video", action="store_true",
                        help="Render keypoints on a blank frame (ignore --video)")
    parser.add_argument("--fps",      type=float, default=None,
                        help="Override FPS (default: read from video or 25)")
    args = parser.parse_args()

    if not args.no_video and args.video is None:
        parser.error("Provide --video or use --no-video to render on a blank background.")

    process(
        pkl_path=args.pkl,
        video_path=args.video,
        output_path=args.output,
        no_video=args.no_video,
        fps_override=args.fps,
    )


if __name__ == "__main__":
    main()
