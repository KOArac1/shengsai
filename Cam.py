# Importing the OpenCV library.
import cv2

cap = cv2.VideoCapture(0)
# Capturing the video from the camera.

while 1:
    # Reading the image from the camera.
    flag, img = cap.read()

    # Checking if the camera is working.
    if not flag:
        break

    # Showing the image.
    cv2.imshow("Video", img)

    if cv2.waitKey(1) == ord('q'):
        break

# Releasing the camera and destroying the window.
cap.release()
cv2.destroyAllWindows()
