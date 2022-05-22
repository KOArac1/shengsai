from xpinyin import Pinyin
import speech_recognition as sr

p = Pinyin()
r = sr.Recognizer()

while True:
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        print("Please speak something...")
        audio = r.listen(source)

        try:
            text = r.recognize_sphinx(audio, language="zh-CN")
            print("\nYou said: ", text)
            text = p.get_pinyin(text, '')
            print("You said: ", text)
            if ("ihao" in text) or ("i hao" in text):
                print("你好！")
        except:
            print("Sorry,I can here you!")
