import pygame
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import itertools
from collections import defaultdict

'''
    map:
    xxxxxxxxxxxx
    x...bpbg.twx
    x....bgdg.tx
    x...bpbg...x
    x....b.b...x
    x...b.bpb..x
    x..bpb.bpb.x
    x.bpb.bpb..x
    x..b...b...x
    x.bpb.bpb..x
    xs.bpb.b...x
    xxxxxxxxxxxx

    x = wall
    s = start
    d = destination
    b = breeze
    p = pit
    g = glitter
    t = stench
    w = wumpus

'''
white = (255, 255, 255)
black = (0, 0, 0)

class Environment(object):

    def __init__(self, width=720, height=540):
        '''
            Initialize pygame, window, background
            observation: (stench, breeze, glitter, bump, scream)
            action: (left, right, up, down, shoot)
        '''
        pygame.init()
        logo = pygame.image.load('wumpus.jpg')
        pygame.display.set_icon(logo)
        pygame.display.set_caption("Wumpus World")
        self.map = list("...bpbg.tw....bgdg.t...bpbg.......b.b......b.bpb....bpb.bpb..bpb.bpb....b...b....bpb.bpb..s.bpb.b...")
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.background = pygame.Surface(self.screen.get_size()).convert()
        self.background.fill(white)
        self.spritey = 461
        self.spritex = 11
        self.wumpus = 1 # live wumpus
        self.agent_pos = 90
        self.done = 0
        self.states = {0:[0,0,0,0,0],1:[0,0,0,0,0],2:[0,0,0,0,0],3:[0,1,0,0,0],4:'p',5:[0,1,0,0,0],6:[0,0,1,0,0],7:[0,0,0,0,0],8:[1,0,0,0,0],9:'w',
                          10:[0,0,0,0,0],11:[0,0,0,0,0],12:[0,0,0,0,0],13:[0,0,0,0,0],14:[0,1,0,0,0],15:[0,0,1,0,0],16:'d',17:[0,0,1,0,0],18:[0,0,0,0,0],19:[1,0,0,0,0],
                          20:[0,0,0,0,0],21:[0,0,0,0,0],22:[0,0,0,0,0],23:[0,1,0,0,0],24:'p',25:[0,1,0,0,0],26:[0,0,1,0,0],27:[0,0,0,0,0],28:[0,0,0,0,0],29:[0,0,0,0,0],
                          30:[0,0,0,0,0],31:[0,0,0,0,0],32:[0,0,0,0,0],33:[0,0,0,0,0],34:[0,1,0,0,0],35:[0,0,0,0,0],36:[0,1,0,0,0],37:[0,0,0,0,0],38:[0,0,0,0,0],39:[0,0,0,0,0],
                          40:[0,0,0,0,0],41:[0,0,0,0,0],42:[0,0,0,0,0],43:[0,1,0,0,0],44:[0,0,0,0,0],45:[0,1,0,0,0],46:'p',47:[0,1,0,0,0],48:[0,0,0,0,0],49:[0,0,0,0,0],
                          50:[0,0,0,0,0],51:[0,0,0,0,0],52:[0,1,0,0,0],53:'p',54:[0,1,0,0,0],55:[0,0,0,0,0],56:[0,1,0,0,0],57:'p',58:[0,1,0,0,0],59:[0,0,0,0,0],
                          60:[0,0,0,0,0],61:[0,1,0,0,0],62:'p',63:[0,1,0,0,0],64:[0,0,0,0,0],65:[0,1,0,0,0],66:'p',67:[0,1,0,0,0],68:[0,0,0,0,0],69:[0,0,0,0,0],
                          70:[0,0,0,0,0],71:[0,0,0,0,0],72:[0,1,0,0,0],73:[0,0,0,0,0],74:[0,0,0,0,0],75:[0,0,0,0,0],76:[0,1,0,0,0],77:[0,0,0,0,0],78:[0,0,0,0,0],79:[0,0,0,0,0],
                          80:[0,0,0,0,0],81:[0,1,0,0,0],82:'p',83:[0,1,0,0,0],84:[0,0,0,0,0],85:[0,1,0,0,0],86:'p',87:[0,1,0,0,0],88:[0,0,0,0,0],89:[0,0,0,0,0],
                          90:[0,0,0,0,0],91:[0,0,0,0,0],92:[0,1,0,0,0],93:'p',94:[0,1,0,0,0],95:[0,0,0,0,0],96:[0,1,0,0,0],97:[0,0,0,0,0],98:[0,0,0,0,0],99:[0,0,0,0,0]}

    def draw_grid(self):

        # Horizontal lines of the grid
        pygame.draw.line(self.background, black, (10, 10), (510, 10))
        pygame.draw.line(self.background, black, (10, 60), (510, 60))
        pygame.draw.line(self.background, black, (10, 110), (510, 110))
        pygame.draw.line(self.background, black, (10, 160), (510, 160))
        pygame.draw.line(self.background, black, (10, 210), (510, 210))
        pygame.draw.line(self.background, black, (10, 260), (510, 260))
        pygame.draw.line(self.background, black, (10, 310), (510, 310))
        pygame.draw.line(self.background, black, (10, 360), (510, 360))
        pygame.draw.line(self.background, black, (10, 410), (510, 410))
        pygame.draw.line(self.background, black, (10, 460), (510, 460))
        pygame.draw.line(self.background, black, (10, 510), (510, 510))

        # Vertical lines of the grid
        pygame.draw.line(self.background, black, (10, 10), (10, 510))
        pygame.draw.line(self.background, black, (60, 10), (60, 510))
        pygame.draw.line(self.background, black, (110, 10), (110, 510))
        pygame.draw.line(self.background, black, (160, 10), (160, 510))
        pygame.draw.line(self.background, black, (210, 10), (210, 510))
        pygame.draw.line(self.background, black, (260, 10), (260, 510))
        pygame.draw.line(self.background, black, (310, 10), (310, 510))
        pygame.draw.line(self.background, black, (360, 10), (360, 510))
        pygame.draw.line(self.background, black, (410, 10), (410, 510))
        pygame.draw.line(self.background, black, (460, 10), (460, 510))
        pygame.draw.line(self.background, black, (510, 10), (510, 510))

    def draw_actor(self, x, y):
        img = pygame.image.load("player.gif")
        img = pygame.transform.scale(img, (48, 48))
        self.background.blit(img, [x, y])

    def draw_pit(self, x, y):
        img = pygame.image.load("pit.gif")
        img = pygame.transform.scale(img, (48, 48))
        self.background.blit(img, [x, y])

    def draw_breeze(self, x, y):
        img = pygame.image.load("breeze2.png")
        img = pygame.transform.scale(img, (48, 48))
        self.background.blit(img, [x, y])

    def draw_glitter(self, x, y):
        img = pygame.image.load("glitter.jpg")
        img = pygame.transform.scale(img, (48, 48))
        self.background.blit(img, [x, y])

    def draw_dest(self, x, y):
        img = pygame.image.load("gold.jpg")
        img = pygame.transform.scale(img, (48, 48))
        self.background.blit(img, [x, y])

    def draw_stench(self, x, y):
        img = pygame.image.load("stench.jpg")
        img = pygame.transform.scale(img, (48, 48))
        self.background.blit(img, [x, y])

    def draw_wumpus(self, x, y):
        img = pygame.image.load("wumpus.jpg")
        img = pygame.transform.scale(img, (48, 48))
        self.background.blit(img, [x, y])

    def draw_blank(self, x, y):
        pygame.draw.rect(self.background,white,(x,y,48,48))

    def draw_map(self):
        x = 11
        y = 11
        count = 0
        for i in self.map:
            if i == 'd':
                self.draw_dest(x, y)
            elif i == 'b':
                self.draw_breeze(x, y)
            elif i == 'p':
                self.draw_pit(x, y)
            elif i == 'g':
                self.draw_glitter(x, y)
            elif i == 't':
                self.draw_stench(x, y)
            elif i == 'w':
                self.draw_wumpus(x, y)
            else:
                self.draw_blank(x,y)
            count += 1
            x += 50
            if count == 10:
                x = 11
                y += 50
                count = 0

    def run(self):
        '''
            Main loop
        '''
        # variable to control main loop
        running = True

        # main loop
        while running:
            # event handling from event queue
            self.draw_grid()
            self.draw_map()
            self.draw_actor(self.spritex, self.spritey)
            pygame.display.flip()
            self.screen.blit(self.background, (0, 0))
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    # user presses escape key
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_LEFT:
                        if not self.spritex <=11:
                            self.spritex -= 50
                            self.agent_pos -= 1
                            print(self.states[self.agent_pos])
                        else:
                            print([0,0,0,1,0])
                    elif event.key == pygame.K_RIGHT:
                        if not self.spritex >=460:
                            self.spritex += 50
                            self.agent_pos += 1
                            print(self.states[self.agent_pos])
                        else:
                            print([0,0,0,1,0])
                    elif event.key == pygame.K_UP:
                        if not self.spritey <=11:
                            self.spritey -= 50
                            self.agent_pos -= 10
                            print(self.states[self.agent_pos])
                        else:
                            print([0,0,0,1,0])
                    elif event.key == pygame.K_DOWN:
                        if not self.spritey >= 460:
                            self.spritey += 50
                            self.agent_pos += 10
                            print(self.states[self.agent_pos])
                        else:
                            print([0,0,0,1,0])

            pygame.display.update()

    def step(self, action):
        done = 0
        
        if action == 0:
            if int(str(self.agent_pos)[-1]) > 0:
                self.agent_pos -= 1
            else:
                done = self.reset()
                return [0,0,0,1,0], -1, done
            observation = self.states[self.agent_pos]
            if observation == 'w' and self.wumpus == 1:
                done = self.reset()
                return observation, -1000, done
            elif observation == 'p':
                done = self.reset()
                return observation, -1000, done
            elif observation == 'd':
                done = self.reset()
                return observation, 1000, done
            else:
                return observation, -1, done
            
        elif action == 1:
            if int(str(self.agent_pos)[-1]) < 9:
                self.agent_pos += 1
            else:
                done = self.reset()
                return [0,0,0,1,0], -1, done
            observation = self.states[self.agent_pos]
            if observation == 'w' and self.wumpus == 1:
                done = self.reset()
                return observation, -1000, done
            elif observation == 'p':
                done = self.reset()
                return observation, -1000, done
            elif observation == 'd':
                done = self.reset()
                return observation, 1000, done
            else:
                return observation, -1, done
            
        elif action == 2:
            if self.agent_pos >= 10:
                self.agent_pos -= 10
            else:
                done = self.reset()
                return [0,0,0,1,0], -1, done
            observation = self.states[self.agent_pos]
            if observation == 'w' and self.wumpus == 1:
                done = self.reset()
                return observation, -1000, done
            elif observation == 'p':
                done = self.reset()
                return observation, -1000, done
            elif observation == 'd':
                self.reset()
                return observation, 1000, done
            else:
                return observation, -1, done
            
        elif action == 3:
            if self.agent_pos <= 89:
                self.agent_pos += 10
            else:
                done = self.reset()
                return [0,0,0,1,0], -1, done
            observation = self.states[self.agent_pos]
            if observation == 'w' and self.wumpus == 1:
                done = self.reset()
                return observation, -1000, done
            elif observation == 'p':
                done = self.reset()
                return observation, -1000, done
            elif observation == 'd':
                done = self.reset()
                return observation, 1000, done
            else:
                return observation, -1, done
            
        elif action == 4:
            if self.wumpus == 1 and (self.states[self.agent_pos+1]=='w' or self.states[self.agent_pos-10]=='w'):
                return [1,0,0,0,1], -10, done
            else:
                return self.states[self.agent_pos], -10, done

    def reset(self):
        done = 1
        self.agent_pos = 90
        self.spritey = 461
        self.spritex = 11
        self.wumpus = 1
        return done
