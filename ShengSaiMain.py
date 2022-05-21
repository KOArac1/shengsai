import speech_recognition as sr
from win32com import client
import threading
# import pyttsx3
import xpinyin
import random
import cv2
from cvzone.HandTrackingModule import HandDetector

MAX_WRONG_TIME = 10

randOver = False
get = [0, None]
wrong = 0
run = True
runBad = False
loopStartOrStop = False
getNum = 0
score = 0
# engine = pyttsx3.init()
engine = client.Dispatch('sapi.spvoice')
right = True
num = num1 = num2 = 0
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2, detectionCon=0.9)
fingers1 = fingers2 = []

engine.Rate = 3


def sayBad():
    print("Start saying bad...")

    global right
    global runBad
    runBad = False
    right = True

    engine.Speak("答错了QAQ...")

    print("Stop saying bad...")


def say(say):
    print("Saying Q...")

    global run
    run = False

    engine.Speak(say)
    # engine.runAndWait()

    print("Stop saying Q...")


def sayGood():
    global right
    global loopStartOrStop
    global score

    loopStartOrStop = True
    print("Loop Start!")

    engine.Speak("Good!你真棒~~~")
    # engine.runAndWait()
    right = True
    score = score + 1
    print("Loop End!")

    loopStartOrStop = False


def main(img_, fingers_, fingers__=[]):
    global getNum
    global right
    global score
    global num

    getNum = -1
    # print("Detector: {}".format(getNum == num))

    fingers = fingers_
    if not fingers__ == []:
        fingers.extend(fingers__)

    for i in fingers:
        if getNum < 0:
            getNum = 0

        if i == 1:
            getNum = getNum + 1

    if getNum == num:
        cv2.putText(img, 'Good!', (200, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
        # print("Detector: {}".format(getNum == num))
        if loopStartOrStop == False:
            print("Start!!!")
            t = threading.Thread(target=sayGood)
            t.start()
        # right = True
    else:
        get[0] = get[0] + 1

        if not get[1] == getNum:
            get[0] = 0
            get[1] = getNum

    if get[0] > MAX_WRONG_TIME:
        if runBad:
            s = threading.Thread(target=sayBad)
            s.start()

    cv2.putText(img_, str(getNum), (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(img_, str(fingers), (100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(img_, str(right), (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return img_, fingers, getNum


while 1:
    # print(getNum)

    if right == True:
        run = True
        wrong = 0
        while 1:
            num1 = random.randint(0, 10)
            num2 = random.randint(0, 10)
            num = num1 + num2
            if num <= 10 and num > 0:
                right = False
                break
            else:
                right = True

        randOver = True

    flag, img = cap.read()

    if not flag:
        break

    hands, img = detector.findHands(img)

    if randOver:
        if hands:
            hand1 = hands[0]
            fingers1 = detector.fingersUp(hand1)
        else:
            # print("Please move your hand in front of the camera until your hand appears on the screen.")
            pass

        if len(hands) == 2:
            hand2 = hands[1]
            fingers2 = detector.fingersUp(hand2)
            img, fingers, fingersUp = main(img, fingers1, fingers2)
        elif len(hands) == 1:
            img, fingers, fingersUp = main(img, fingers1)
        else:
            cv2.putText(img, "? ? ?", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.putText(img, str(num1) + "+" + str(num2) + "=?",
                    (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    if run == True:
        if loopStartOrStop == False:
            s = threading.Thread(target=say, args=(
                str(num1) + "+" + str(num2) + "=几?",))
            s.start()

    cv2.putText(img, "Score: {}".format(score), (500, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv2.imshow("Game", img)

    if cv2.waitKey(1) == ord('q'):
        f = open("./save/game.sav", "w")
        f.write(str(score))
        f.close()
        break

del engine
cap.release()
cv2.destroyAllWindows()
