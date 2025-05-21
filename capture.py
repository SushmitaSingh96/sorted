import cv2

# OpenCV will treat the video capture device as a webcam
cap = cv2.VideoCapture(1)  # 0 is typically the first video device

# Set resolution (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Display the frame (optional)
    cv2.imshow("Microscope Feed", frame)

    # Press 's' to save a frame
    key = cv2.waitKey(1)
    if key == ord('s'):
        cv2.imwrite("captured_image.jpg", frame)
        print("Image saved!")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
