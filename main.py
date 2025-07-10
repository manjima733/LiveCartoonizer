import cv2
import numpy as np

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def cartoonify_frame(frame):
    # Resize for consistency
    frame = cv2.resize(frame, (640, 480))

    # Convert to grayscale and blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)

    # Edge detection (bold cartoon outlines)
    edges = cv2.adaptiveThreshold(
        gray_blur, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9,
        C=2
    )

    # Color smoothing with bilateral filter
    color = frame
    for _ in range(2):
        color = cv2.bilateralFilter(color, d=9, sigmaColor=250, sigmaSpace=250)

    # Combine color and edges
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    # Detect face(s)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Apply soft blur to cartoon face region (not replace it)
    for (x, y, w, h) in faces:
        roi = cartoon[y:y+h, x:x+w]
        softened_face = cv2.bilateralFilter(roi, d=9, sigmaColor=75, sigmaSpace=75)
        cartoon[y:y+h, x:x+w] = softened_face  # blend smoothly

    return cartoon

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print(" Cannot open webcam.")
    exit()

print("📸 Press 's' to save a frame. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Failed to grab frame.")
        break

    cartoon_frame = cartoonify_frame(frame)
    cv2.imshow("Live Cartoon with Smooth Face", cartoon_frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        cv2.imwrite("final_cartoon_output.jpg", cartoon_frame)
        print(" Saved as 'final_cartoon_output.jpg'")
    elif key == ord('q'):
        print(" Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
