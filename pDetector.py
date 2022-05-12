import math
import time

import cv2
from cvzone import HandTrackingModule

cap = cv2.VideoCapture(0)
detector = HandTrackingModule.HandDetector(detectionCon=0.75, maxHands=1)
cTime = time.time()
nowCenter = lastCenter = [0, 0]
Lcj = cj = 0
fps = 0
lastLmList = LmList = []

def Parkinson(image, lastLmList, lmList):
    if not lastLmList == []:
        length, info, img = detector.findDistance(lmList[8][0:2], lastLmList[8][0:2], img)
        if length > 25:
            cv2.putText(image, "Parkinson!", (lmList[8][0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        else:
            cv2.putText(image, "Normal!", (lmList[8][0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        
        pTime = time.time()
        Rfps = 1 / (pTime - cTime)
        cTime = pTime
        
        lastLmList = LmList

        return image, Rfps


while True:
    flag, img = cap.read()
    if not flag:
        print("Read Error!")
        break

    hands, img = detector.findHands(img)
    if len(hands) == 1:
        hand = hands[0]
        lmList = hand["lmList"]
        center = hand["center"]
        nowCenter = center[0 : 2]
        if not lastCenter == [0, 0]:
            cxj = abs(nowCenter[0] - lastCenter[0])
            cyj = abs(nowCenter[1] - lastCenter[1])
            cj = cxj ** 2 + cyj ** 2
            cj = math.sqrt(cj)
            if not Lcj == 0:
                if abs(cj - Lcj) < 25:
                    # img, fps = Parkinson(img, lastLmList, lmList)
                    if not lastLmList == []:
                        length, info, img = detector.findDistance(lmList[8][0:2], lastLmList[8][0:2], img)
                        if length > 25:
                            cv2.putText(img, "Parkinson!", (lmList[8][0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                        else:
                            cv2.putText(img, "Normal!", (lmList[8][0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        
                        pTime = time.time()
                        fps = 1 / (pTime - cTime)
                        cTime = pTime
        
                    lastLmList = LmList

                Lcj = cj
            else:
                Lcj = cj

            lastCenter = nowCenter
        else:
            lastCenter = nowCenter

    if not fps == 0:
        cv2.putText(img, str(fps), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        
    cv2.imshow("Parkinson", img)
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
