import pyautogui as pag
import time

time.sleep(5)

# Aキーを押す
for i in range(10):
    pag.press('a')
    time.sleep(1)

for i in range(10):
    pag.press('d')
    time.sleep(1)