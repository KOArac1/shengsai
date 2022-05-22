# Importing the HandDetector class from the HandTrackingModule.py file.
import cv2
from cvzone.HandTrackingModule import HandDetector

# Creating a video capture object and a hand detector object.
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2, detectionCon=0.8)

while 1:
    # Reading the video frame by frame.
    flag, img = cap.read()

    if not flag:
        break

    hands, img = detector.findHands(img)

    # Showing the image in a window.
    cv2.imshow("Windows", img)

    if cv2.waitKey(1) == ord('q'):
        break

# Releasing the camera and destroying the window.
cap.release()
cv2.destroyAllWindows()
