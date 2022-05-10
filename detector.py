import cv2

detector = cv2.face.LBPHFaceRecognizer_create()
detector.read('./data/train.yml')
det = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
image = cv2.imread('./picture/85.jpg')
face = det.detectMultiScale(image, 1.05, 10)

for x, y, w, h in face:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow('Image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
