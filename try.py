import threading
import time
import pyttsx3

engine = pyttsx3.init()

def thread():
    print("This is a thread!")
    engine.say("This is a thread!")
    engine.runAndWait()

# for i in range(10):
    # t = threading.Thread(target=thread)
    # t.start()
    # time.sleep(0.5)

t = threading.Thread(target=thread)
t.start()