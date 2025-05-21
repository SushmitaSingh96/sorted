import cv2
import time
import os
import subprocess
from datetime import datetime
import sys

TEMP_FOLDER = "temp_images"
os.makedirs(TEMP_FOLDER, exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(TEMP_FOLDER, f"microscope_{timestamp}.jpg")

        cv2.imwrite(image_path, frame)
        print(f"Saved {image_path}")

        subprocess.Popen([sys.executable, "inference.py", image_path])

        time.sleep(5)

except KeyboardInterrupt:
    print("Capture stopped by user")

finally:
    cap.release()
    cv2.destroyAllWindows()
