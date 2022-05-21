# **省赛项目**
- main.py
    - 实现人脸检测
- train.py
    - 训练人脸模型
- detector.py
    - 使用训练的模型来进行人脸识别
- 更新! pDetector.py
    - 尝试检测帕金森
- haarcascade_frontalface_alt2.xml
    - OpenCV 中的人脸检测文件
- data
    - 储存训练后的模型
- picture
    - 99 张人脸图片

# main.py
## 源码:
``` python
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
```
### 这里使用了python中的OpenCV (cv2) 包,以实现人脸检测的效果.
---
## 详解:
### 首先是这一段
``` python
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
```
### 光这一部分就占了全部代码的二分之一, *cTime* 在前面就已经被定义了,见下:
``` python
cTime = time.time()
afps = []
pfps = 0
```
### 这一段可以看到,在主体程序之前,也就是程序开始处,而后面的 *pTime* 是在每轮循环末尾,再来进行
``` python
pTime = time.time()
```
### 然后,再计算 ***fps(Frames Per Second,每秒传输帧数)*** :
``` python
fps = int(1 / (pTime - cTime) + 0.5)
```
### *pTime* - *cTime* 计算程序每轮循环所耗时间(单位秒),用一除以即可算出 ***fps*** .
### *afps* 是一个列表,记录了最近11次的 ***fps*** ; *pfps* 是平均 ***fps*** 值,用 *afps* 各项之和除以 *afps* 的长度来计算.
``` python
for i in range(afps.__len__()):
    pfps = pfps + afps[i]

pfps = pfps / (afps.__len__() + 1)
```
---

# train.py

## 源码:
``` python
import os
import cv2
import numpy
from PIL import Image


def getImage(path):
    faceS = []
    ids = []
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceD = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
    # print(imagePaths)
    for imagePath in imagePaths:
        PILImg = numpy.array(Image.open(imagePath).convert('L'))
        NPImg = numpy.array(PILImg, 'uint8')
        det_faces = faceD.detectMultiScale(NPImg)
        face = int(os.path.split(imagePath)[1].split('.')[0])
        for x, y, w, h in det_faces:
            try:
                faceS.append(cv2.resize(NPImg[x : x + w, y : y + h], (224, 224)))
                # faceS.append(NPImg[x : x + w, y : y + h])
                ids.append(face)
            except:
                print("Error:", imagePath)
            # if not x >= 0 or not y >= 0 or not w >= 0 or not h >= 0:
            #     print(x, y, w, h)
            # print(x, y, w, h)
        # print(face)
    return faceS, ids


if __name__ == '__main__':
    path = "./picture"
    faces, ids = getImage(path)
    r = cv2.face.LBPHFaceRecognizer_create()
    r.train(faces, numpy.array(ids))
    r.write('./data/train.yml')
```
### 这里首先使用了 OpenCV 中的人脸检测模型来对每张图片中的人脸进行划分,并使用 Numpy 包来对图片(列表) 进行分割.
``` python
faceD = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
```
``` python
det_faces = faceD.detectMultiScale(NPImg)
```
``` python
PILImg = numpy.array(Image.open(imagePath).convert('L'))
NPImg = numpy.array(PILImg, 'uint8')
```
``` python
faceS.append(cv2.resize(NPImg[x : x + w, y : y + h], (224, 224)))
# faceS.append(NPImg[x : x + w, y : y + h])  Error!
ids.append(face)
```
### 源码中使用了 **列表推导式** 来对指定路径中的每张图片进行遍历,获取每张图片的路径.
``` python
imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
```
### 然后使用 for 循环来对每个路径进行遍历.
``` python
for imagePath in imagePaths:
```
### 在检测到人脸之后,又使用了 for 循环来对图片中的各个像素点来进行遍历 **(图片是个列表)** .
``` python
for x, y, w, h in det_faces:
            try:
                faceS.append(cv2.resize(NPImg[x : x + w, y : y + h], (224, 224)))
                # faceS.append(NPImg[x : x + w, y : y + h])
                ids.append(face)
            except:
                print("Error:", imagePath)
```
### 在前面,还对 *face* (编号) 进行了定义.
``` python
face = int(os.path.split(imagePath)[1].split('.')[0])
```
### 在这个循环中,使用了 **try** ,因为有时程序会检测不到人脸,导致后面的 **train()** 出现报错.
### 然后就是训练了!
``` python
if __name__ == '__main__':
    path = "./picture"
    faces, ids = getImage(path)
    r = cv2.face.LBPHFaceRecognizer_create()
    r.train(faces, numpy.array(ids))
    r.write('./data/train.yml')
```
### 其中, *path* 是图片路径, **getImage()** 是上面的函数,用来获取每张图片中的人脸和编号, **train()** 就是 OpenCV 中的训练函数, **write()** 用来保存模型.
---
# detector.py
## 源码:
``` python
import cv2

detector = cv2.face.LBPHFaceRecognizer_create()
detector.read('./data/train.yml')
det = cv2.CascadeClassifier('./haarcascade_frontalface_alt2.xml')
image = cv2.imread('./picture/99.jpg')
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
```
### 再读取了模型之后,就可以进行人脸识别了.
``` python
detector = cv2.face.LBPHFaceRecognizer_create()
detector.read('./data/train.yml')
```
---
# pDetector.py
### 老规矩，看源码：
``` python
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
```
### 这个程序异常复杂。
### 这里只做简单介绍 ~~（其实就是懒）~~。
### 首先是这里：
``` python
cxj = abs(nowCenter[0] - lastCenter[0])
cyj = abs(nowCenter[1] - lastCenter[1])
cj = cxj ** 2 + cyj ** 2
cj = math.sqrt(cj)
```
### 这里的 *cxj* 是上一帧食指与这一帧食指的坐标之间的 x 距离。
### 同理， *cyj* 是上一帧食指与这一帧食指的坐标之间的 y 距离。
### 然后就是勾股定理了， ~~两直角边的平方之和的算数平方根即为第三边~~。
### **P.S：**
### 原理是帕金森患者手抖，于是这个程序的原理是在手的中心点没有太大变化的前提下，若手指坐标大规模变化，即判定为帕金森。
