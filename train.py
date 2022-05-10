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
