from comtypes.gen import SpeechLib
from inspect import EndOfBlock
from comtypes.client import CreateObject

engine = CreateObject("sapi.spvoice")
stream = CreateObject("sapi.spfilestream")
infile = 'F:\\shengsai\\test\\infile.txt'
outfile = 'F:\\shengsai\\data\\outfile.wav'
stream.Open(outfile, SpeechLib.SSFMCreateForWrite)
engine.AudioOutputStream = stream
f = open(infile, 'r', encoding='utf-8')
theText = f.read()
f.close()
print(theText)
engine.speak(theText)
stream.close()
