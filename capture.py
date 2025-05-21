import cv2
import subprocess
import sys
import os
from datetime import datetime

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
TEMP_FOLDER = "temp_images"
os.makedirs(TEMP_FOLDER, exist_ok=True)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Microscope Feed", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(TEMP_FOLDER, f"microscope_{timestamp}.jpg")
        cv2.imwrite(image_path, frame)
        print(f"Image saved as {image_path}!")

        # Run inference.py immediately on the saved image
        subprocess.Popen([sys.executable, "inference.py", image_path])

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
