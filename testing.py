import os
import cv2
import torch
import numpy as np
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from ByteTrack.yolox.tracking_utils.timer import Timer
from ByteTrack.yolox.exp import get_exp
from ByteTrack.yolox.utils import postprocess, fuse_model, get_model_info
from ByteTrack.yolox.data.data_augment import preproc

# -------------------------
# Tracker Args
# -------------------------
class TrackerArgs:
    track_thresh = 0.5
    match_thresh = 0.8
    track_buffer = 30
    mot20 = False

args = TrackerArgs()

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load YOLOX + ByteTrack pretrained
# -------------------------
exp_file = "ByteTrack/exps/example/mot/yolox_x_mix_det.py"
ckpt_path = "ByteTrack/pretrained/bytetrack_x_mot17.pth.tar"
exp = get_exp(exp_file, "bytetrack")  # exp_name can be any string
model = exp.get_model().to(device)
model.eval()
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model"])
fuse_model(model)
print("YOLOX + ByteTrack pretrained model loaded")

# -------------------------
# Initialize Tracker
# -------------------------
tracker = BYTETracker(args, frame_rate=30)
timer = Timer()

# -------------------------
# Counting setup
# -------------------------
counted_ids = set()
count_line_ratio = 0.5  # horizontal line at 50% of frame height

# -------------------------
# Run video/webcam
# -------------------------
def run_video(source=0, save_result=False):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Cannot open video source: {source}")
        return

    save_writer = None
    ret, frame = cap.read()
    if not ret:
        print(" Cannot read first frame")
        return
    frame_h, frame_w = frame.shape[:2]
    count_line_y = int(frame_h * count_line_ratio)

    if save_result:
        os.makedirs("results", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        save_writer = cv2.VideoWriter("results/output.mp4", fourcc, fps, (frame_w, frame_h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # -------------------------
        # Preprocess
        # -------------------------
        # -------------------------
# Preprocess
# -------------------------
        img, ratio = preproc(frame, exp.test_size, mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225))
        img = torch.from_numpy(img).unsqueeze(0).float().to(device)


        # -------------------------
        # Inference
        # -------------------------
        with torch.no_grad():
            outputs = model(img)
            outputs = postprocess(outputs, exp.num_classes, exp.test_conf, exp.nmsthre)

        # -------------------------
        # Update tracker
        # -------------------------
        online_targets = []
        if outputs[0] is not None:
            online_targets = tracker.update(outputs[0], [frame_h, frame_w], exp.test_size)

        # -------------------------
        # Draw and count
        # -------------------------
        cv2.line(frame, (0, count_line_y), (frame_w, count_line_y), (0,0,255), 2)
        for t in online_targets:
            tlwh = t.tlwh
            track_id = t.track_id
            x, y, w, h = map(int, tlwh)
            x2, y2 = x + w, y + h
            cv2.rectangle(frame, (x, y), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"ID:{track_id}", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

            # Count crossing line
            center_y = y + h//2
            if center_y >= count_line_y and track_id not in counted_ids:
                counted_ids.add(track_id)

        # Show total count and FPS
        cv2.putText(frame, f"Count: {len(counted_ids)}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        cv2.putText(frame, f"FPS: {1./max(1e-5, timer.average_time):.2f}", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

        if save_writer:
            save_writer.write(frame)
        cv2.imshow("YOLOX + ByteTrack Counting", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(f"Total Count: {len(counted_ids)}")
    cap.release()
    if save_writer:
        save_writer.release()
    cv2.destroyAllWindows()

# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    run_video(source=0, save_result=True)
