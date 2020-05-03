import numpy as np
import cv2
teste = cv2.imread("Temp/example.jpg",1)
game = True
def Game(screen):
    temp_game = cv2.imread('Temp/gameover.png',0)
    x, y = temp_game.shape[::-1]
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    res_bot = cv2.matchTemplate(screen_gray,temp_game,cv2.TM_CCOEFF_NORMED)
    corte = 0.5
    local_bot = np.where(res_bot >= corte)
    print(local_bot)
    for i in zip(*local_bot[::-1]):
        cv2.rectangle(screen, i, (i[0] + x, i[1] + y), (0,0,255),1)
        game = False
        return screen_gray
    return screen_gray

while True:
    screen= Game(teste)
    #print(game)
    cv2.imshow("alo",screen)
    cv2.waitKey(1)