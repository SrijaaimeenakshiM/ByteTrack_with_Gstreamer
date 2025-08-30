import os
import cv2
import numpy as np
import torch
import math
import time
from dronekit import connect, VehicleMode, LocationGlobalRelative
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from ByteTrack.yolox.tracking_utils.timer import Timer
from ByteTrack.yolox.models.yolox import YOLOX
from ByteTrack.yolox.exp import get_exp

# -------------------------
# Tracker args
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
# Load YOLOX model + pretrained weights
# -------------------------
exp_file = "ByteTrack/exps/example/mot/yolox_x_mix_det.py"  # replace with your exp file
ckpt_path = "ByteTrack/pretrained/bytetrack_x_mot17.pth.tar"

exp = get_exp(exp_file, "bytetrack")
model = exp.get_model().to(device)
model.eval()

ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model"])
print("YOLOX + ByteTrack pretrained model loaded")

# -------------------------
# Initialize ByteTrack
# -------------------------
tracker = BYTETracker(args, frame_rate=30)
timer = Timer()
object_counter = {}

# -------------------------
# Simulated drone connection (SITL)
# -------------------------
print("Connecting to SITL drone...")
vehicle = connect("udp:127.0.0.1:14552", wait_ready=True)
vehicle.mode = VehicleMode("GUIDED")
print("Drone connected")

# Drone altitude
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
        if vehicle.location.global_relative_frame.alt >= target_altitude * 0.95:
            break
        time.sleep(1)

# -------------------------
# Drone/GPS helpers
# -------------------------
def get_target_location(current_location, offset_x_m, offset_y_m, alt1):
    R = 6378137.0
    dLat = offset_y_m / R
    dLon = offset_x_m / (R * math.cos(math.pi * current_location.lat / 180.0))
    newlat = current_location.lat + (dLat * 180 / math.pi)
    newlon = current_location.lon + (dLon * 180 / math.pi)
    return LocationGlobalRelative(newlat, newlon, alt1)

def camera_to_uav(x_cam, y_cam):
    x_uav = -y_cam
    y_uav = x_cam
    return x_uav, y_uav

def uav_to_ne(x_uav, y_uav, yaw_rad):
    c = math.cos(yaw_rad)
    s = math.sin(yaw_rad)
    north = x_uav * c - y_uav * s
    east = x_uav * s + y_uav * c
    return north, east

# -------------------------
# Run video + tracking
# -------------------------
def run_video(source=0, save_result=False):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f" Cannot open video source: {source}")
        return

    save_writer = None
    if save_result:
        os.makedirs("results", exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        save_writer = cv2.VideoWriter("results/output.mp4", fourcc, fps, (w, h))

    arm_and_takeoff(altitude_m)
    print("Starting detection + tracking...")

    global object_counter

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)

        # YOLOX inference
        with torch.no_grad():
            outputs = model(img)

        detections = []
        if outputs[0] is not None:
            dets = outputs[0].detach().cpu().numpy()
            for det in dets:
                x1, y1, x2, y2, score, cls_id = det
                if int(cls_id) == 0:
                    detections.append([x1, y1, x2, y2, score])
        detections = np.array(detections, dtype=np.float32)

        # Update tracker
        online_targets = tracker.update(
            detections,
            [frame.shape[0], frame.shape[1]],
            [frame.shape[0], frame.shape[1]]
        ) if detections.shape[0] > 0 else []

        # Draw boxes + IDs
        for t in online_targets:
            tlwh = t.tlwh
            track_id = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > 20 and not vertical:
                if track_id not in object_counter:
                    object_counter[track_id] = 1
                x, y, w, h = map(int, tlwh)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(frame, f"ID:{track_id}", (x,y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)

        cv2.putText(frame, f"Total: {len(object_counter)}", (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255),2)
        cv2.putText(frame, f"FPS: {1./max(1e-5, timer.average_time):.2f}", (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0),2)

        if save_writer:
            save_writer.write(frame)
        cv2.imshow("YOLOX + ByteTrack + DroneSim", frame)

        # Simulated drone navigation (comment out if only testing tracking)
        if online_targets:
            target_center_x = int((online_targets[0].tlwh[0] + online_targets[0].tlwh[0]+online_targets[0].tlwh[2])/2)
            target_center_y = int((online_targets[0].tlwh[1] + online_targets[0].tlwh[1]+online_targets[0].tlwh[3])/2)
            offset_x_m = (target_center_x - frame.shape[1]//2) * 0.05
            offset_y_m = (target_center_y - frame.shape[0]//2) * 0.05
            x_uav, y_uav = camera_to_uav(offset_x_m, offset_y_m)
            north_offset, east_offset = uav_to_ne(x_uav, y_uav, math.radians(vehicle.heading))
            current_location = vehicle.location.global_relative_frame
            target_location = get_target_location(current_location, north_offset, east_offset, altitude_m)
            print(f"Drone moving to simulated target at lat:{target_location.lat} lon:{target_location.lon}")
            vehicle.simple_goto(target_location)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Final count:", len(object_counter))
    cap.release()
    if save_writer:
        save_writer.release()
    cv2.destroyAllWindows()
    vehicle.mode = VehicleMode("LAND")
    vehicle.close()

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    # source=0 for webcam, or "test_video.mp4" for video file
    run_video(source="test_video.mp4", save_result=True)
