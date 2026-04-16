import cv2
import os

# Get current script directory (poseestimation)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Create images folder inside poseestimation
images_path = os.path.join(BASE_DIR, "images")
os.makedirs(images_path, exist_ok=True)

# Try external webcam (Logitech usually index 1 or 2)
cap = cv2.VideoCapture(1)  

count = 0

print("Press SPACE to capture image")
print("Press ESC to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)

    if key % 256 == 27:  # ESC
        print("Closing...")
        break

    elif key % 256 == 32:  # SPACE
        img_name = os.path.join(images_path, f"img_{count}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"Saved {img_name}")
        count += 1

cap.release()
cv2.destroyAllWindows()