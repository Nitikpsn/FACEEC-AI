import cv2
import face_recognition
import os
import numpy as np
from threading import Thread

# ---------------- OPTIMIZED SETTINGS ----------------
KNOWN_FACES_DIR = "known_faces"
FRAME_RESIZE_SCALE = 0.4
PROCESS_EVERY_N_FRAMES = 20
CONF_THRESHOLD = 0.1
HAAR_MIN_SIZE = (40, 40)
BOX_THICKNESS = 2
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 480

# ---------------- LOAD KNOWN FACES ----------------
known_face_encodings = []
known_face_names = []

print("Loading known faces...")
for file in os.listdir(KNOWN_FACES_DIR):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(KNOWN_FACES_DIR, file)
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(rgb_img, model="small")
        if len(enc) > 0:
            known_face_encodings.append(enc[0])
            known_face_names.append(os.path.splitext(file)[0])
            print("Loaded:", file)
print("Total known faces:", len(known_face_names))

# ---------------- HAAR CASCADE ----------------
haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
if haar.empty():
    print("Error loading Haar cascade!")
    exit()

# ---------------- THREADING FOR CAMERA ----------------
class VideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, DISPLAY_WIDTH)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, DISPLAY_HEIGHT)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def release(self):
        self.stopped = True
        self.stream.release()

# Initialize camera
print("Starting camera...")
cam = None
for i in range(4):
    temp_cam = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if temp_cam.isOpened():
        temp_cam.release()
        cam = VideoStream(src=i)
        print(f"Camera detected at index {i}")
        break

if cam is None:
    print("No camera detected!")
    exit()

frame_count = 0
face_locations = []
face_names = []
face_confidences = []

# ---------------- CREATE FULLSCREEN WINDOW ----------------
window_name = "Classroom CCTV - Press 'F' for Fullscreen, 'Q' to Quit"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("\nControls:")
print("F = Toggle Fullscreen")
print("Q = Quit")
print("-" * 40)

# ---------------- MAIN LOOP ----------------
while True:
    frame = cam.read()
    if frame is None:
        continue

    frame_count += 1

    # Only process every N frames for face recognition
    if frame_count % PROCESS_EVERY_N_FRAMES == 0:
        small_frame = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_SCALE, fy=FRAME_RESIZE_SCALE)
        gray_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        detections = haar.detectMultiScale(
            gray_small,
            scaleFactor=1.15,
            minNeighbors=3,
            minSize=HAAR_MIN_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        face_locations = []
        face_names = []
        face_confidences = []

        # Process up to 10 faces for classroom
        for (x, y, w, h) in detections[:10]:
            top = int(y / FRAME_RESIZE_SCALE)
            left = int(x / FRAME_RESIZE_SCALE)
            bottom = int((y + h) / FRAME_RESIZE_SCALE)
            right = int((x + w) / FRAME_RESIZE_SCALE)

            # Crop with bounds checking
            top = max(0, top)
            left = max(0, left)
            bottom = min(frame.shape[0], bottom)
            right = min(frame.shape[1], right)

            face_crop = frame[top:bottom, left:right]
            if face_crop.size == 0:
                continue

            rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            enc = face_recognition.face_encodings(rgb_face, model="small")

            name = "Unknown"
            conf = 0.0

            if len(enc) > 0 and len(known_face_encodings) > 0:
                distances = face_recognition.face_distance(known_face_encodings, enc[0])
                best_idx = np.argmin(distances)
                conf = max(0, min(1, 1 - distances[best_idx]))
                if conf >= CONF_THRESHOLD:
                    name = known_face_names[best_idx]

            face_locations.append((top, right, bottom, left))
            face_names.append(name)
            face_confidences.append(conf)

    # ---------------- DRAW BOXES (EVERY FRAME) ----------------
    for (top, right, bottom, left), name, conf in zip(face_locations, face_names, face_confidences):
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        # Face rectangle
        cv2.rectangle(frame, (left, top), (right, bottom), color, BOX_THICKNESS)
        # Name box
        cv2.rectangle(frame, (left, bottom), (right, bottom + 30), color, -1)
        label = f"{name} {int(conf * 100)}%"
        cv2.putText(frame, label, (left + 5, bottom + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow(window_name, frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("f"):
        # Toggle fullscreen
        prop = cv2.getWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN)
        if prop == cv2.WINDOW_FULLSCREEN:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        else:
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cam.release()
cv2.destroyAllWindows()
