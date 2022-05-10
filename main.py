import cv2
import time

cap = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
cTime = time.time()
afps = []
pfps = 0

while True:
    flag,img = cap.read()
    if not flag:
        break
    face = detector.detectMultiScale(img, 1.14514, 11)
    for x,y,w,h in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
    pTime = time.time()
    fps = int(1 / (pTime - cTime) + 0.5)
    cTime = pTime
    afps.append(fps)
    print(afps)
    if afps.__len__() > 11.4514:
        del(afps[0])

    for i in range(afps.__len__()):
        pfps = pfps + afps[i]

    pfps = pfps / (afps.__len__() + 1)
    cv2.putText(img, 'fps: ' + str(fps), (50, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.putText(img, 'Average frame rate: ' + str(int(pfps + 0.5)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    cv2.imshow('Video',img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
