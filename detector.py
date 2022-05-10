import cv2

detector = cv2.face.LBPHFaceRecognizer_create()
detector.read('./data/train.yml')
det = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
image = cv2.imread('./picture/96.jpg')
face = det.detectMultiScale(image, 1.05, 10)

for x, y, w, h in face:
    img = image[x : x + w, y : y + h]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    id, confidence = detector.predict(img)
    print(id)
    if (confidence > 50):
        cv2.putText(image,'unknow', (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    else:
        cv2.putText(image,str(id), (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(image,str(confidence) + '%', (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow('Image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
