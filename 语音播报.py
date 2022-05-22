from win32com import client

# Creating an instance of the SAPI.SpVoice COM object.
engine = client.Dispatch('sapi.spvoice')

# 说出括号内的文字。
engine.Speak("语音播报,运行成功!")

del engine
# Deleting the COM object.
