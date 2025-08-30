print("code running...")
from dronekit import connect, VehicleMode, LocationGlobalRelative
import time
import math
import cv2
import torch
import numpy as np
from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from ByteTrack.yolox.tracking_utils.timer import Timer
from ByteTrack.yolox.models.yolox import YOLOX
from ByteTrack.yolox.exp import get_exp

# ------------------------- Drone Connection -------------------------
vehicle = connect("udp:127.0.0.1:14552", wait_ready=True)
print("Drone connected")
vehicle.mode = VehicleMode("GUIDED")

sensor_width_mm = 7.6
focal_length_mm = 4.6
altitude_m = 10

def arm_and_takeoff(target_altitude):
    while not vehicle.is_armable:
        time.sleep(1)
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True
    while not vehicle.armed:
        time.sleep(1)
    vehicle.simple_takeoff(target_altitude)
    while True:
        if vehicle.location.global_relative_frame.alt >= target_altitude * 0.95:
            break
        time.sleep(1)

# ------------------------- ByteTrack Setup -------------------------
class TrackerArgs:
    track_thresh = 0.3
    match_thresh = 0.9
    track_buffer = 60
    mot20 = False

args = TrackerArgs()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load YOLOX model
exp_file = "ByteTrack/exps/example/mot/yolox_x_mix_det.py"
ckpt_path = "ByteTrack/pretrained/bytetrack_x_mot17.pth.tar"
exp = get_exp(exp_file, "bytetrack")
model = exp.get_model().to(device)
model.eval()
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model"])
print("✅ YOLOX + ByteTrack model loaded")

tracker = BYTETracker(args, frame_rate=30)
timer = Timer()
object_counter = {}

# ------------------------- Drone Utility Functions -------------------------
def get_target_location(current_location, offset_x_m, offset_y_m, alt1=altitude_m):
    R = 6378137.0
    dLat = offset_y_m / R
    dLon = offset_x_m / (R * math.cos(math.pi * current_location.lat / 180.0))
    newlat = current_location.lat + (dLat * 180 / math.pi)
    newlon = current_location.lon + (dLon * 180 / math.pi)
    return LocationGlobalRelative(newlat, newlon, alt1)

def get_distance_meters(loc1, loc2):
    dlat = loc2.lat - loc1.lat
    dlon = loc2.lon - loc1.lon
    return math.sqrt((dlat * 1.113195e5)**2 + (dlon * 1.113195e5)**2)

def camera_to_uav(x_cam, y_cam):
    return -y_cam, x_cam

def uav_to_ne(x_uav, y_uav, yaw_rad):
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return x_uav*c - y_uav*s, x_uav*s + y_uav*c

# ------------------------- GStreamer Camera -------------------------
gst_pipeline = (
    "rtspsrc location=rtsp://192.168.144.25:8554/main.264 latency=0 ! "
    "rtph264depay ! avdec_h264 ! videoconvert ! appsink"
)
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
print("Camera connected")

# ------------------------- Takeoff -------------------------
arm_and_takeoff(altitude_m)

# ------------------------- Main Loop -------------------------
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # ------------------------- Preprocess for YOLOX -------------------------
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)

    # ------------------------- Convert detections to ByteTrack -------------------------
    detections = []
    if outputs[0] is not None:
        dets = outputs[0].detach().cpu().numpy()
        for det in dets:
            x1, y1, x2, y2, score, cls_id = det
            if int(cls_id) == 0:
                detections.append([x1, y1, x2, y2, score])
    detections = np.array(detections, dtype=np.float32)

    # ------------------------- Update Tracker -------------------------
    if len(detections) > 0:
        timer.tic()
        online_targets = tracker.update(
            detections, [frame.shape[0], frame.shape[1]], [frame.shape[0], frame.shape[1]]
        )
        timer.toc()
    else:
        online_targets = []

    # ------------------------- Track & Navigate -------------------------
    for t in online_targets:
        tlwh = t.tlwh
        track_id = t.track_id
        x, y, w, h = map(int, tlwh)
        object_centerx = x + w//2
        object_centery = y + h//2

        image_centerx = frame.shape[1] // 2
        image_centery = frame.shape[0] // 2

        # GSD meters per pixel
        GSD = (sensor_width_mm * vehicle.location.global_relative_frame.alt) / (focal_length_mm * frame.shape[1])
        offset_x_m = (object_centerx - image_centerx) * GSD
        offset_y_m = (object_centery - image_centery) * GSD

        x_uav, y_uav = camera_to_uav(offset_x_m, offset_y_m)
        yaw_rad = math.radians(vehicle.heading)
        north_offset, east_offset = uav_to_ne(x_uav, y_uav, yaw_rad)

        current_location = vehicle.location.global_relative_frame
        target_location = get_target_location(current_location, north_offset, east_offset)
        distance = get_distance_meters(current_location, target_location)

        print(f"ID {track_id} | Offset X={offset_x_m:.2f}, Y={offset_y_m:.2f} | Distance={distance:.2f}")
        vehicle.simple_goto(target_location)

    # ------------------------- Optional Display -------------------------
    cv2.imshow("Drone YOLOX+ByteTrack", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
vehicle.close()
