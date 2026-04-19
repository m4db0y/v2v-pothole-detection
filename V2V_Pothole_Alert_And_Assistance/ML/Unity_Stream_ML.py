import socket
import struct
import cv2
import numpy as np
from ultralytics import YOLO
import time

# =========================
# SETTINGS
# =========================
HOST = '0.0.0.0'
PORT = 5005
CONF_THRESHOLD = 0.6
last_inference = 0
inference_delay = 0.1

# =========================
# LOAD MODEL
# =========================
model = YOLO("F:/Python/ML/best.pt")

# =========================
# SERVER (WAIT FOR UNITY)
# =========================
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print("🎮 Waiting for Unity connection...")
conn, addr = server_socket.accept()
print("✅ Connected from:", addr)

data = b""
window_created = False

# =========================
# MAIN LOOP
# =========================
while True:

    # Receive frame size
    while len(data) < 4:
        packet = conn.recv(4096)
        if not packet:
            break
        data += packet

    if len(data) < 4:
        continue

    packed_size = data[:4]
    data = data[4:]
    msg_size = struct.unpack('I', packed_size)[0]

    # Receive full frame
    while len(data) < msg_size:
        packet = conn.recv(4096)
        if not packet:
            break
        data += packet

    if len(data) < msg_size:
        continue

    frame_data = data[:msg_size]
    data = data[msg_size:]

    # Decode
    np_data = np.frombuffer(frame_data, dtype=np.uint8)
    frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

    if frame is None:
        continue

    # Create window after first frame
    if not window_created:
        cv2.namedWindow("Unity YOLO", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Unity YOLO", 800, 600)
        window_created = True

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

    # Show window
    cv2.imshow("Unity YOLO", frame)

    # Send result back to Unity
    try:
        conn.sendall(b"1" if pothole_detected else b"0")
    except:
        print("❌ Connection lost")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
conn.close()
server_socket.close()
cv2.destroyAllWindows()