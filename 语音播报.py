from win32com import client
# import time

filename = 'F:\\shengsai\\test\\test.txt'

engine = client.Dispatch('sapi.spvoice')

file = open(filename, 'r', encoding='utf-8')
result = file.read()
print('Start!')

# time.sleep(1)

# print(result)

# engine.Speak(result)

while 1:
    engine.Speak("Fuck you.")

print("End!")

file.close()

del engine
