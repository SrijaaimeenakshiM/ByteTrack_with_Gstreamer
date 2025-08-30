import os
import cv2
import numpy as np
import math
import time
import torch
from dronekit import connect, VehicleMode, LocationGlobalRelative

# ByteTrack imports
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from ByteTrack.yolox.tracking_utils.timer import Timer
from ByteTrack.yolox.exp import get_exp
from ByteTrack.yolox.utils import postprocess, fuse_model, get_model_info
from ByteTrack.yolox.data.data_augment import preproc

# -------------------------
# Tracker Args
# -------------------------
class TrackerArgs:
    track_thresh = 0.3
    match_thresh = 0.9
    track_buffer = 60
    mot20 = False

args = TrackerArgs()

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Load YOLOX model (pretrained ByteTrack)
# -------------------------
exp_file = "ByteTrack/exps/example/mot/yolox_x_mix_det.py"  # your exp file
ckpt_path = "ByteTrack/pretrained/bytetrack_x_mot17.pth.tar"

from ByteTrack.yolox.exp import get_exp
exp = get_exp(exp_file, "bytetrack")
model = exp.get_model().to(device)
model.eval()

ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model"])
print("✅ YOLOX + ByteTrack pretrained model loaded")

# -------------------------
# Initialize ByteTrack
# -------------------------
tracker = BYTETracker(args, frame_rate=30)
timer = Timer()
object_counter = {}

# -------------------------
# Drone connection
# -------------------------
vehicle = connect("udp:127.0.0.1:14552", wait_ready=True)
vehicle.mode = VehicleMode("GUIDED")

# Camera specs
sensor_width_mm = 7.6
focal_length_mm = 4.6
altitude_m = 10

def arm_and_takeoff(target_altitude):
    print("Arming motors...")
    while not vehicle.is_armable:
        time.sleep(1)
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True
    while not vehicle.armed:
        time.sleep(1)
    print("Taking off!")
    vehicle.simple_takeoff(target_altitude)
    while True:
        current_alt = vehicle.location.global_relative_frame.alt
        if current_alt >= target_altitude * 0.95:
            break
        time.sleep(1)

# Convert pixel offset to GPS
def get_target_location(current_location, offset_x_m, offset_y_m, alt1=altitude_m):
    R = 6378137.0
    dLat = offset_y_m / R
    dLon = offset_x_m / (R * math.cos(math.pi * current_location.lat / 180.0))
    newlat = current_location.lat + (dLat * 180 / math.pi)
    newlon = current_location.lon + (dLon * 180 / math.pi)
    return LocationGlobalRelative(newlat, newlon, alt1)

def camera_to_uav(x_cam, y_cam):
    return -y_cam, x_cam

def uav_to_ne(x_uav, y_uav, yaw_rad):
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    north = x_uav * c - y_uav * s
    east = x_uav * s + y_uav * c
    return north, east

# -------------------------
# Start Drone + Camera
# -------------------------
gst_pipeline = (
    "rtspsrc location=rtsp://192.168.144.25:8554/main.264 latency=0 ! "
    "rtph264depay ! avdec_h264 ! videoconvert ! appsink"
)
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
arm_and_takeoff(altitude_m)

# -------------------------
# Main loop
# -------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_width_px = frame.shape[1]
    image_height_px = frame.shape[0]
    GSD = (sensor_width_mm * vehicle.location.global_relative_frame.alt) / (focal_length_mm * image_width_px)

    # -------------------------
    # Preprocess frame for YOLOX
    # -------------------------
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)

    # Convert detections to ByteTrack format
    detections = []
    if outputs[0] is not None:
        dets = outputs[0].detach().cpu().numpy()
        for det in dets:
            x1, y1, x2, y2, score, cls_id = det
            if int(cls_id) == 0:
                detections.append([x1, y1, x2, y2, score])
    detections = np.array(detections, dtype=np.float32)

    # Update tracker
    if detections.shape[0] > 0:
        timer.tic()
        online_targets = tracker.update(
            detections,
            [frame.shape[0], frame.shape[1]],
            [frame.shape[0], frame.shape[1]]
        )
        timer.toc()
    else:
        online_targets = []

    # Draw bounding boxes + IDs + total count
    for t in online_targets:
        tlwh = t.tlwh
        track_id = t.track_id
        vertical = tlwh[2] / tlwh[3] > 1.6
        if tlwh[2] * tlwh[3] > 20 and not vertical:
            x, y, w, h = map(int, tlwh)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
            cv2.putText(frame, f'ID:{track_id}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            if track_id not in object_counter:
                object_counter[track_id] = 1

    cv2.putText(frame, f"Total: {len(object_counter)}", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2)
    cv2.putText(frame, f"FPS: {1./max(1e-5, timer.average_time):.2f}", (20,80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)

    cv2.imshow("Drone Camera - Tracking", frame)

    # Navigate drone to first detected object (if any)
    if online_targets:
        t = online_targets[0]
        tlwh = t.tlwh
        cx = tlwh[0] + tlwh[2]/2
        cy = tlwh[1] + tlwh[3]/2
        offset_x_m = (cx - image_width_px/2) * GSD
        offset_y_m = (cy - image_height_px/2) * GSD
        x_uav, y_uav = camera_to_uav(offset_x_m, offset_y_m)
        yaw_rad = math.radians(vehicle.heading)
        north_offset, east_offset = uav_to_ne(x_uav, y_uav, yaw_rad)
        current_location = vehicle.location.global_relative_frame
        target_location = get_target_location(current_location, north_offset, east_offset)
        vehicle.simple_goto(target_location)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cv2.destroyAllWindows()
cap.release()
vehicle.close()
