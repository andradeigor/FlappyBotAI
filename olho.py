import mss
from PIL import Image
#import pyautogui as pg
import numpy as np
import cv2
from calc import calculo
while True:
        
    with mss.mss() as sct:
        monitor = {"top": 80, "left": 10, "width": 280, "height": 480}
        screen = np.array(sct.grab(monitor))
        screen,game = calculo(screen)
        cv2.imshow("Eye",screen)
        cv2.waitKey(1)
