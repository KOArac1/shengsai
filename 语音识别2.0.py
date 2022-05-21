import speech_recognition as sr
import pyttsx3

engine = pyttsx3.init()
recognizer = sr.Recognizer()

while True:
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.2)
        print("Say something please !")
        audio = recognizer.listen(source)

        try:
            text = recognizer.recognize_sphinx(audio, language="zh-CN")  # 还可以选择不同的数据源，从而用来识别不同的语言
            print("You said : ", text)
            engine.say(text)
            engine.runAndWait()

        except:
            print("Sorry I can't hear you!")
            engine.say("Sorry I can't hear you!")
            engine.runAndWait()
