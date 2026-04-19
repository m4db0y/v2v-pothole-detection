import socket
import struct
import cv2
import numpy as np
from ultralytics import YOLO
import time

# =========================
# MODE SWITCH (1 = Webcam, 0 = Unity)
# =========================
USE_WEBCAM = 0

# =========================
# LOAD MODEL
# =========================
model = YOLO("F:/Downloads/best.pt")

# =========================
# SETTINGS
# =========================
CONF_THRESHOLD = 0.85
last_inference = 0
inference_delay = 0.2  # ~5 FPS

# =========================
# WEBCAM MODE
# =========================
if USE_WEBCAM == 1:

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Webcam not detected")
        exit()

    print("📷 Running WEBCAM mode")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        pothole_detected = 0

        # YOLO
        if time.time() - last_inference > inference_delay:
            results = model(frame, conf=CONF_THRESHOLD, verbose=False)
            last_inference = time.time()
        else:
            results = []

        for r in results:
            for box in r.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                width = x2 - x1
                height = y2 - y1
                area = width * height
                aspect_ratio = width / (height + 1e-5)

                if aspect_ratio > 3 or aspect_ratio < 0.3:
                    continue
                if y1 < h * 0.4:
                    continue
                if area < 800:
                    continue

                pothole_detected = 1

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Display
        if pothole_detected == 1:
            cv2.putText(frame, "POTHOLE DETECTED", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "NO POTHOLE", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        cv2.imshow("Webcam YOLO", frame)

        print("Detection:", pothole_detected)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# =========================
# UNITY MODE
# =========================
else:

    HOST = '0.0.0.0'
    PORT = 5005

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server_socket.bind((HOST, PORT))
    server_socket.listen(1)

    print("🎮 Waiting for Unity connection...")
    conn, addr = server_socket.accept()
    print("Connected from:", addr)

    data = b""

    while True:

        # Receive size
        while len(data) < 4:
            packet = conn.recv(4096)
            if not packet:
                break
            data += packet

        if len(data) < 4:
            break

        packed_size = data[:4]
        data = data[4:]
        msg_size = struct.unpack('I', packed_size)[0]

        # Receive frame
        while len(data) < msg_size:
            data += conn.recv(4096)

        frame_data = data[:msg_size]
        data = data[msg_size:]

        # Decode frame
        np_data = np.frombuffer(frame_data, dtype=np.uint8)
        frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        h, w, _ = frame.shape
        pothole_detected = 0

        # YOLO
        if time.time() - last_inference > inference_delay:
            results = model(frame, conf=CONF_THRESHOLD, verbose=False)
            last_inference = time.time()
        else:
            results = []

        for r in results:
            for box in r.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                width = x2 - x1
                height = y2 - y1
                area = width * height
                aspect_ratio = width / (height + 1e-5)

                if aspect_ratio > 3 or aspect_ratio < 0.3:
                    continue
                if y1 < h * 0.4:
                    continue
                if area < 800:
                    continue

                pothole_detected = 1

                # Draw box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Display in VS Code window
        if pothole_detected == 1:
            cv2.putText(frame, "POTHOLE DETECTED", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "NO POTHOLE", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        cv2.imshow("Unity YOLO Detection", frame)

        # Send result to Unity
        try:
            conn.sendall(b"1" if pothole_detected == 1 else b"0")
        except:
            print("❌ Connection lost")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()