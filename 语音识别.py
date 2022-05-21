import speech_recognition as sr

audio_file = 'F:\\shengsai\\data\\outfile.wav'
r = sr.Recognizer()

with sr.AudioFile(audio_file) as source:
    audio = r.record(source)

print(r.recognize_sphinx(audio, language="zh-CN"))