import socket
import cv2
import numpy as np
from ultralytics import YOLO
import time

# =========================
# SETTINGS
# =========================
CONF_THRESHOLD = 0.6
last_inference = 0
inference_delay = 0.1

# =========================
# LOAD MODEL
# =========================
model = YOLO("F:/Python/ML/best.pt")

# =========================
# CONNECT TO UNITY
# =========================
UNITY_IP = "127.0.0.1"
PORT = 5005

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

print("🔌 Connecting to Unity...")
client.connect((UNITY_IP, PORT))
print("✅ Connected to Unity")

# =========================
# WEBCAM
# =========================
cap = cv2.VideoCapture(0)

cv2.namedWindow("Webcam YOLO", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Webcam YOLO", 800, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    pothole_detected = 0

    if time.time() - last_inference > inference_delay:
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        last_inference = time.time()
    else:
        results = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if y1 < h * 0.3:
                continue

            pothole_detected = 1

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            cv2.putText(frame, "POTHOLE",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2)

    # Send result to Unity
    client.sendall(b"1" if pothole_detected else b"0")

    cv2.imshow("Webcam YOLO", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
client.close()
cv2.destroyAllWindows()