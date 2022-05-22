# import speech_recognition as sr
from win32com import client
import threading
# import xpinyin
import random
import cv2
# Importing a module from a package.
from cvzone.HandTrackingModule import HandDetector

# It's setting the maximum number of times you can get the answer wrong before the program tells you
# that you got it wrong.
MAX_WRONG_TIME = 50

# It's setting the global variables to their default values.
best = True
randOver = True
get = [0, None]
run = True
runBad = False
loopStartOrStop = False
getNum = 0
score = 0
engine = client.Dispatch('sapi.spvoice')
right = True
num = num1 = num2 = 0
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2, detectionCon=0.9)
fingers1 = fingers2 = []
randTime = 0
exitOrNot = False
runBadOver = True
lastNum = None

engine.Rate = 3  # 设置语言语速


def setupGet(_=0, __=None):  # 初始化get列表
    """
    It sets the first two elements of the global variable `get` to 0 and None
    """
    global get

    get[0] = _
    get[1] = __


def rand():    # 随机题目
    global right
    global run
    global wrong
    global randOver
    global num1
    global num2
    global num
    global get
    global randTime
    global exitOrNot
    global lastNum

    randOver = False

    if randTime < 10:
        if right == True:
            setupGet()
            run = True
            wrong = 0
            # An infinite loop.
            while 1:
                lastNum = num
                num1 = random.randint(0, 10)
                num2 = random.randint(0, 10)
                num = num1 + num2
                if num <= 10 and num > 0 and num != lastNum:
                    right = False
                    randTime += 1
                    break
                else:
                    right = True

        randOver = True
    else:
        exitOrNot = True


def sayBad():  # 播报答错了
    """
    > It's a function that says "答错了" and then sets the global variable `runBad` to `False`
    """
    print("Start saying bad...")

    global get
    global score
    global right
    global runBad
    global runBadOver

    runBadOver = False
    runBad = False
    setupGet()

    engine.Speak("答错了...")
    right = True
    # score -= 1
    # 扣分，但会读成“杠x分”
    print("Stop saying bad...")

    runBadOver = True
    rand()


def say(say):  # 播报题目
    print("Saying Q...")

    global run

    run = False

    engine.Speak(say)

    print("Stop saying Q...")


def sayGood():  # 播报奖励
    global right
    global loopStartOrStop
    global score

    loopStartOrStop = True
    print("Loop Start!")

    engine.Speak("Good!你真棒~~~")
    right = True
    score = score + 1
    print("Loop End!")

    loopStartOrStop = False
    rand()


def main(img_, fingers_, fingers__=None):    # 检测主程序
    if fingers__ is None:
        fingers__ = []
    global getNum
    global right
    global score
    global num
    global runBad
    global loopStartOrStop
    global run

    getNum = -1
    # print("Detector: {}".format(getNum == num))

    fingers = fingers_
    if fingers__ != []:
        fingers.extend(fingers__)

    for i in fingers:
        getNum = max(getNum, 0)
        if i == 1:
            getNum = getNum + 1

    if runBadOver and not loopStartOrStop and randOver:
        if getNum == num:
            cv2.putText(img, 'Good!', (200, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            print("Start!!!")
            t = threading.Thread(target=sayGood)
            t.start()  # 多线程!播报鼓励内容.
        else:
            get[0] += 1

            if get[1] != getNum:
                get[0] = 0
                get[1] = getNum

    # print(get[0])
    # print(get[0] > MAX_WRONG_TIME)
    # print(runBad)
    # 修BUG用的，现在似乎没什么用...

    # It's checking if the number of times you got the answer wrong is equal to the maximum number of
    # times you can get the answer wrong before the program tells you that you got it wrong. If it is,
    # it sets the global variable `runBad` to `True`.
    if get[0] == MAX_WRONG_TIME:
        runBad = True

    if not loopStartOrStop and randOver and runBad and runBadOver and get != [0, None]:
        s = threading.Thread(target=sayBad)
        s.start()

    cv2.putText(img_, str(getNum), (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(img_, str(fingers), (100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(img_, str(get[1]), (100, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 显示一些数值

    return img_, fingers, getNum  # 返回数值


rand()

while True:
    print(get)

    # It's checking if the user has answered 10 questions. If they have, it breaks out of the loop.
    if exitOrNot:
        break

    flag, img = cap.read()  # 读取摄像头每帧图片

    # It's checking if the video is being read correctly. If it isn't, it prints "视频读取失败。。。" and then
    # breaks out of the loop.
    if not flag:
        print("视频读取失败。。。")
        break

    hands, img = detector.findHands(img)

    if randOver:
        if hands:
            hand1 = hands[0]
            fingers1 = detector.fingersUp(hand1)
        else:
            # print("Please move your hand in front of the camera until your hand appears on the screen.")
            setupGet()

        if len(hands) == 2:
            hand2 = hands[1]
            fingers2 = detector.fingersUp(hand2)
            img, fingers, fingersUp = main(img, fingers1, fingers2)
        elif len(hands) == 1:
            img, fingers, fingersUp = main(img, fingers1)
        else:
            cv2.putText(img, "? ? ?", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        cv2.putText(img, f"({randTime}) {str(num1)}+{str(num2)}=几?", (200, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    if run and not loopStartOrStop:
        s = threading.Thread(target=say, args=(
            f"{str(num1)}+{str(num2)}=几?", ))
        s.start()

    # It's drawing the score on the screen.
    cv2.putText(img, f"Score: {score}", (500, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 设置图片为灰度图
    cv2.imshow("Game", img)  # 显示图片

    if cv2.waitKey(1) in [ord('c'), ord('C')]:
        with open("./save/game.sav", "w+") as f:
            f.write('')
        print("Cleaned!!!")

    # It's checking if the user pressed the `q` or `Q` key. If they did, it says "本次分数为: {score} 分!"
    # and then
    #         it opens the file `./save/game.sav` and checks if the current score is the highest
    # score. If it is, it
    #         sets the global variable `best` to `True`. It then writes the current score to the file.
    if cv2.waitKey(1) in [ord('q'), ord('Q')]:
        break

engine.Speak(f"本次分数为: {score} 分!")
with open("./save/game.sav", "r+", encoding="utf-8") as f:
    ch = f.readlines()
    print(ch)
    for chr in ch:
        chr.strip()
        print(chr)
        if int(chr) >= score:
            best = False
            break
        """
        with open('ww.txt',encoding='utf-8') as file:
            content=file.read()
            print(content.rstrip())     ##rstrip()删除字符串末尾的空行
            ###逐行读取数据
        for line in content:
            print(line)
        """

    f.write(str(score) + "\n")

# It's checking if the current score is the highest score. If it is, it says "本次为历史最高分！"
if best or score == 10:
    engine.Speak("本次为历史最高分！")

# It's releasing the memory that the program is using.
del engine
cap.release()
cv2.destroyAllWindows()

# It's exiting the program.
exit(0)
