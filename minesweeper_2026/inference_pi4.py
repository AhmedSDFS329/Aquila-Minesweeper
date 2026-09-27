"""
IEEE Minesweepers 2026 — Surface Mine Detection on Raspberry Pi 4
TFLite INT8 inference with USB webcam
"""

import cv2
import numpy as np
import time
import json
from datetime import datetime

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter

try:
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Bool, Float32
    from geometry_msgs.msg import Point
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

# ── Config ──────────────────────────────────────────────
MODEL_PATH = "models/best_int8.tflite"
CONFIDENCE_THRESHOLD = 0.3
INPUT_SIZE = 320
CLASS_NAMES = ["mine"]
CAMERA_INDEX = 0

# GPIO pins (uncomment on Pi 4)
# import RPi.GPIO as GPIO
# LED_PIN = 17
# BUZZER_PIN = 27

# Known mine size for distance estimation (5cm cube)
REAL_MINE_SIZE_CM = 5.0
FOCAL_LENGTH_PX = None
CALIBRATION_FILE = "calibration.json"


# ── Detection Logger ────────────────────────────────────
class MineLogger:
    def __init__(self):
        self.detections = []

    def log(self, bbox, confidence, distance_cm=None):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "bbox": {"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]},
            "distance_cm": round(distance_cm, 1) if distance_cm else None,
        }
        self.detections.append(entry)
        return entry

    def save(self, path="mine_detections.json"):
        with open(path, "w") as f:
            json.dump(self.detections, f, indent=2)
        print(f"Saved {len(self.detections)} detections to {path}")


# ── GPIO Alert ──────────────────────────────────────────
def setup_gpio():
    """Call once at startup on Pi 4."""
    # GPIO.setmode(GPIO.BCM)
    # GPIO.setup(LED_PIN, GPIO.OUT)
    # GPIO.setup(BUZZER_PIN, GPIO.OUT)
    pass


def alert_mine_detected(on=True):
    """Trigger LED + buzzer when mine is detected."""
    # GPIO.output(LED_PIN, GPIO.HIGH if on else GPIO.LOW)
    # GPIO.output(BUZZER_PIN, GPIO.HIGH if on else GPIO.LOW)
    if on:
        print(">>> ALERT: MINE DETECTED! <<<")


# ── Distance Estimation ────────────────────────────────
def load_calibration():
    global FOCAL_LENGTH_PX
    try:
        with open(CALIBRATION_FILE, "r") as f:
            data = json.load(f)
            FOCAL_LENGTH_PX = data["focal_length_px"]
            print(f"Calibration loaded: focal_length={FOCAL_LENGTH_PX:.1f} px")
    except (FileNotFoundError, KeyError):
        pass


def save_calibration():
    with open(CALIBRATION_FILE, "w") as f:
        json.dump({"focal_length_px": FOCAL_LENGTH_PX}, f)
    print(f"Calibration saved to {CALIBRATION_FILE}")


def estimate_distance(bbox_size_px, frame_width):
    global FOCAL_LENGTH_PX
    if FOCAL_LENGTH_PX is None or bbox_size_px == 0:
        return None
    return (REAL_MINE_SIZE_CM * FOCAL_LENGTH_PX) / bbox_size_px


def calibrate_focal_length(bbox_size_px, known_distance_cm):
    global FOCAL_LENGTH_PX
    FOCAL_LENGTH_PX = (bbox_size_px * known_distance_cm) / REAL_MINE_SIZE_CM
    save_calibration()
    print(f"Focal length calibrated: {FOCAL_LENGTH_PX:.1f} px")
    return FOCAL_LENGTH_PX


# ── ROS 2 Publisher ────────────────────────────────────
class MinePublisher:
    def __init__(self):
        if not ROS_AVAILABLE:
            self.enabled = False
            return
        rclpy.init()
        self.node = rclpy.create_node('surface_detection_node')
        self.alert_pub = self.node.create_publisher(Bool, '/mine_detector/alert', 10)
        self.distance_pub = self.node.create_publisher(Float32, '/mine_detector/distance', 10)
        self.position_pub = self.node.create_publisher(Point, '/mine_detector/position', 10)
        self.enabled = True
        print("ROS 2 node started: surface_detection_node")

    def publish_detection(self, x1, y1, x2, y2, distance_cm):
        if not self.enabled:
            return
        alert_msg = Bool()
        alert_msg.data = True
        self.alert_pub.publish(alert_msg)

        if distance_cm is not None:
            dist_msg = Float32()
            dist_msg.data = float(distance_cm)
            self.distance_pub.publish(dist_msg)

        pos_msg = Point()
        pos_msg.x = float((x1 + x2) / 2)
        pos_msg.y = float((y1 + y2) / 2)
        pos_msg.z = float(distance_cm) if distance_cm else 0.0
        self.position_pub.publish(pos_msg)

    def publish_no_detection(self):
        if not self.enabled:
            return
        alert_msg = Bool()
        alert_msg.data = False
        self.alert_pub.publish(alert_msg)

    def shutdown(self):
        if self.enabled:
            self.node.destroy_node()
            rclpy.shutdown()


# ── TFLite Model ────────────────────────────────────────
class MineDetector:
    def __init__(self, model_path):
        self.interpreter = Interpreter(model_path=model_path, num_threads=4)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.input_shape = self.input_details[0]["shape"]
        self.input_dtype = self.input_details[0]["dtype"]

        if self.input_shape[1] == 3:
            self.nchw = True
            self.img_h = self.input_shape[2]
            self.img_w = self.input_shape[3]
        else:
            self.nchw = False
            self.img_h = self.input_shape[1]
            self.img_w = self.input_shape[2]

        # Check for quantization parameters
        self.input_scale = self.input_details[0].get("quantization_parameters", {}).get("scales", [1.0])
        self.input_zero = self.input_details[0].get("quantization_parameters", {}).get("zero_points", [0])

        print(f"Model loaded: input={self.input_shape}, dtype={self.input_dtype}")

    def preprocess(self, frame):
        """Resize with letterbox and prepare for TFLite."""
        h, w = frame.shape[:2]
        scale = min(self.img_w / w, self.img_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        pad_w, pad_h = (self.img_w - new_w) // 2, (self.img_h - new_h) // 2

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.img_h, self.img_w, 3), 114, dtype=np.uint8)
        canvas[pad_h : pad_h + new_h, pad_w : pad_w + new_w] = resized

        self.scale = scale
        self.pad_w = pad_w
        self.pad_h = pad_h
        self.orig_shape = (h, w)

        img = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        if self.input_dtype == np.float32:
            img = img.astype(np.float32) / 255.0
        elif self.input_dtype == np.int8:
            img = img.astype(np.int8)
        elif self.input_dtype == np.uint8:
            pass  # already uint8

        img = np.expand_dims(img, axis=0)

        if self.nchw:
            img = np.transpose(img, (0, 3, 1, 2))

        return img

    def detect(self, frame):
        """Run detection on a single frame. Returns list of (x1,y1,x2,y2,conf,cls)."""
        input_tensor = self.preprocess(frame)
        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details[0]["index"])

        # Dequantize if needed
        out_details = self.output_details[0]
        if out_details["dtype"] != np.float32:
            quant = out_details.get("quantization_parameters", {})
            scales = quant.get("scales", [1.0])
            zeros = quant.get("zero_points", [0])
            if len(scales) > 0:
                output = (output.astype(np.float32) - zeros[0]) * scales[0]

        self._debug_output(output)
        return self._postprocess(output)

    def _debug_output(self, output):
        """Print once to diagnose output format."""
        if not hasattr(self, '_debugged'):
            self._debugged = True
            print(f"Output shape: {output.shape}, dtype: {output.dtype}")
            print(f"Output min: {output.min():.4f}, max: {output.max():.4f}")
            flat = output[0]
            if flat.shape[0] < flat.shape[1]:
                flat = flat.T
            best_idx = np.argmax(flat[:, 4:].max(axis=1))
            row = flat[best_idx]
            print(f"Best detection raw: cx={row[0]:.4f} cy={row[1]:.4f} w={row[2]:.4f} h={row[3]:.4f} score={row[4]:.4f}")

    def _postprocess(self, output):
        """Parse YOLOv8 TFLite output → detections in original image coords."""
        # Output shape: [1, 5, 2100] — (batch, features, detections)
        # Features: cx, cy, w, h, conf (normalized 0-1)
        output = output[0]  # remove batch dim

        if output.shape[0] < output.shape[1]:
            # Shape is [5, 2100] → transpose to [2100, 5]
            output = output.T

        detections = []
        for row in output:
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            scores = row[4:]
            conf = float(np.max(scores))
            cls_id = int(np.argmax(scores))

            if conf < CONFIDENCE_THRESHOLD:
                continue

            # Coordinates are normalized (0-1) — scale to input size first
            cx *= self.img_w
            cy *= self.img_h
            w *= self.img_w
            h *= self.img_h

            # Convert from model coords to original image coords
            x1 = (cx - w / 2 - self.pad_w) / self.scale
            y1 = (cy - h / 2 - self.pad_h) / self.scale
            x2 = (cx + w / 2 - self.pad_w) / self.scale
            y2 = (cy + h / 2 - self.pad_h) / self.scale

            orig_h, orig_w = self.orig_shape
            x1 = max(0, min(int(x1), orig_w))
            y1 = max(0, min(int(y1), orig_h))
            x2 = max(0, min(int(x2), orig_w))
            y2 = max(0, min(int(y2), orig_h))

            detections.append((x1, y1, x2, y2, conf, cls_id))

        # NMS
        if detections:
            detections = self._nms(detections, iou_threshold=0.45)

        return detections

    def _nms(self, detections, iou_threshold=0.45):
        """Simple NMS."""
        detections.sort(key=lambda d: d[4], reverse=True)
        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)
            detections = [
                d for d in detections if self._iou(best, d) < iou_threshold
            ]
        return keep

    def _iou(self, a, b):
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0


# ── Drawing ─────────────────────────────────────────────
def draw_detections(frame, detections, fps=0):
    """Draw bounding boxes, labels, and FPS on frame."""
    for x1, y1, x2, y2, conf, cls_id in detections:
        color = (0, 0, 255)  # red
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Distance estimation
        bbox_size = max(x2 - x1, y2 - y1)
        dist = estimate_distance(bbox_size, frame.shape[1])
        label = "MINE"
        if dist:
            label += f" {dist:.0f}cm"

        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(
            frame,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0] + 5, y1),
            color,
            -1,
        )
        cv2.putText(
            frame, label, (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )

    # FPS counter
    cv2.putText(
        frame, f"FPS: {fps:.1f}", (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
    )
    return frame


# ── Main Loop ───────────────────────────────────────────
def main():
    print("=== IEEE Minesweepers — Mine Detection ===")

    # Init
    detector = MineDetector(MODEL_PATH)
    logger = MineLogger()
    ros_pub = MinePublisher()
    setup_gpio()
    load_calibration()

    # Camera
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {CAMERA_INDEX}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Camera opened. Press 'q' to quit, 'c' to calibrate distance.")
    print(f"Model: {MODEL_PATH}")
    print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")

    fps = 0
    mine_alert_active = False
    frame_count = 0
    detections = []

    try:
        while True:
            t_start = time.time()

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # Detect every 2nd frame to save CPU
            if frame_count % 2 == 0:
                detections = detector.detect(frame)

            # Alert logic
            if detections and not mine_alert_active:
                alert_mine_detected(on=True)
                mine_alert_active = True
                for x1, y1, x2, y2, conf, cls_id in detections:
                    bbox_size = max(x2 - x1, y2 - y1)
                    dist = estimate_distance(bbox_size, frame.shape[1])
                    entry = logger.log(
                        bbox=(x1, y1, x2 - x1, y2 - y1),
                        confidence=conf,
                        distance_cm=dist,
                    )
                    ros_pub.publish_detection(x1, y1, x2, y2, dist)
                    print(f"  Mine: bbox=({x1},{y1},{x2},{y2}), dist={dist}")
            elif not detections and mine_alert_active:
                alert_mine_detected(on=False)
                ros_pub.publish_no_detection()
                mine_alert_active = False

            # Draw + display
            elapsed = time.time() - t_start
            if frame_count % 2 == 0 and elapsed > 0:
                fps = 1.0 / elapsed
            frame = draw_detections(frame, detections, fps)
            cv2.imshow("Minesweeper Detection", frame)

            # Key controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                # Calibration mode: hold mine at 30cm, press 'c'
                if detections:
                    x1, y1, x2, y2, conf, cls_id = detections[0]
                    bbox_size = max(x2 - x1, y2 - y1)
                    calibrate_focal_length(bbox_size, known_distance_cm=30)
                else:
                    print("No mine detected — hold mine at 30cm and press 'c'")
            elif key == ord("s"):
                logger.save()

    except KeyboardInterrupt:
        pass
    finally:
        logger.save()
        ros_pub.shutdown()
        cap.release()
        cv2.destroyAllWindows()
        # GPIO.cleanup()
        print("Done.")


if __name__ == "__main__":
    main()
