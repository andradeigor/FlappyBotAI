import numpy as np
import cv2
from pynput.keyboard import Key, Controller
import os
import neat
import mss
import pygame
from PIL import Image
import pickle
font  = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
fontColor = (255,255,255)
lineType = 2


keyboard = Controller()

"""


ANALISE DE IMAGENS



"""

def pipes(screen):#encontra, pela técnica de template matching os canos.
    temp_bot = cv2.imread('template/cano_bot.jpg',0)
    temp_top = cv2.imread('template/cano_top.jpg',0)
    x, y = temp_bot.shape[::-1]
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY )
    res_bot = cv2.matchTemplate(screen_gray,temp_bot,cv2.TM_CCOEFF_NORMED)
    corte = 0.8
    local_bot = np.where(res_bot >= corte)
    for i in zip(*local_bot[::-1]):
        cv2.rectangle(screen, i, (i[0] + x, i[1] + y), (0,0,255),1)
    x, y = temp_top.shape[::-1]
    res_top = cv2.matchTemplate(screen_gray,temp_top,cv2.TM_CCOEFF_NORMED)
    local_top = np.where(res_top >= corte)
    for i in zip(*local_top[::-1]):
        cv2.rectangle(screen, i, (i[0] + x, i[1] + y), (0,0,255),1)
    local_top = local_top[::-1]
    return screen, local_top

def bird(screen):#encontra, por meio da cor, o pássaro e desenha um quadrado em volta dele.
    screen = cv2.cvtColor(screen, cv2.COLOR_RGBA2RGB)
    lower_bird = np.array([46,178,243])#gap de valores a serem procurados, esse valor é BGR e depende da tela/sistema, nao 'sei ao certo. 
    upper_bird = np.array([55,188,253])#Quando vim para o linux em um notebook precisei alterar o valor.
    mask = cv2.inRange(screen, lower_bird, upper_bird)
    local = np.where(mask)
    local = local[::-1]#local é uma array 2d, sendo a primeira os conjuntos dos x, e a segunda os dos y.
    if len(local[0]) >0:
        cv2.rectangle(screen, (local[0][0]-10,local[1][0]-10), (local[0][0] + 24, local[1][0] + 24), (255,0,0),1)
    return screen, local

def process(screen,pipes,bird):
    try:
        cv2.line(screen,(pipes[0][0]+30,pipes[1][0]+43),(bird[0][0],bird[1][0]),(0,0,255),1)
        cv2.line(screen,(pipes[0][1]+30,pipes[1][1]+143),(bird[0][0],bird[1][0]),(0,0,255),1)
        cv2.putText(screen,str((pipes[0][0]+30-bird[0][0])), (((bird[0][0] + int(((pipes[0][0]+30)-bird[0][0])/2))),(pipes[1][0]+100) ), font, fontScale,(0,0,0),lineType)
    except:
        pass
    try:
        cv2.line(screen,(bird[0][0],bird[1][0]),(bird[0][0],390),(0,0,255),1)
        bird_y = int((390-bird[1][0])/2)
        cv2.putText(screen,str(int((390-bird[1][0])/2)), ((bird[0][0]),(390-bird_y)), font, fontScale,(0,0,0),lineType)
    except:
        pass
    return screen

def Game(screen):
    game = True
    temp_game = cv2.imread('template/gameover.png',0)
    x, y = temp_game.shape[::-1]
    screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
    res_bot = cv2.matchTemplate(screen_gray,temp_game,cv2.TM_CCOEFF_NORMED)
    corte = 0.56
    local_bot = np.where(res_bot >= corte)
    for i in zip(*local_bot[::-1]):
        cv2.rectangle(screen, i, (i[0] + x, i[1] + y), (0,0,255),1)
        game = False
        return screen,game
    return screen,game
"""


FIM DA ANALISE DE IMAGENS



"""


"""


MOVIMENTOS



"""

def jump():
    keyboard.press(Key.space)
    keyboard.release(Key.space)

def reset():
    keyboard.press(Key.enter)
    keyboard.release(Key.enter)

"""


FIM DOS MOVIMENTOS



"""


"""


REDE NEURAL



"""

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
    neat.DefaultSpeciesSet,neat.DefaultStagnation,config_path)

    p = neat.Population(config)
    
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    
    winner = p.run(main)

    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

def main(genomes,config):
    for _,g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g,config)
        g.fitness = 0
        clock = pygame.time.Clock()
        run = True
        while run:
            clock.tick(30)
            with mss.mss() as sct:
                monitor = {"top": 80, "left": 10, "width": 280, "height": 480}
                screen = np.array(sct.grab(monitor))
                screen,cord = pipes(screen)#processamento de imagem
                screen,cord_bird= bird(screen)
                screen = process(screen,cord,cord_bird)
                screen,game = Game(screen)
                cv2.imshow("Eye",screen)
                cv2.waitKey(1)
                try:
                    cord_top = cord[1][0] + 43
                except:
                    cord_top = 200
                try:
                    cord_bot = cord[1][1]+143
                except:
                    cord_bot = 300            
                try:
                    cord_bird = cord_bird[1][0]
                except:
                    cord_bird = 0            
                if cord_bird > cord_top and cord_bird < cord_bot:
                    g.fitness += 1
                if game ==False:
                    print("resetando")
                    reset()
                    run = False
                g.fitness += 0.1
                output = net.activate((cord_bird, cord_top, cord_bot))
                if output[0]> 0.5:
                    jump()

                

if __name__ == "__main__":     
        local_dir = os.path.dirname(__file__)
        config_path = os.path.join(local_dir, "config_neat.txt")
        run(config_path)















"""


FIM DA REDE NEURAL



"""
