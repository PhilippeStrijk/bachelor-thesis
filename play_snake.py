# import the modules needed
import math
import random
import pygame
import time
import sys
from pygame.locals import *
import itertools


def handle_key_press():
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == KEYDOWN:
            if event.key == K_RIGHT or event.key == ord('d'):
                change_to = 'RIGHT'
                return change_to
            if event.key == K_LEFT or event.key == ord('a'):
                change_to = 'LEFT'
                return change_to
            if event.key == K_UP or event.key == ord('w'):
                change_to = 'UP'
                return change_to
            if event.key == K_DOWN or event.key == ord('s'):
                change_to = 'DOWN'
                return change_to
            if event.key == K_ESCAPE:
                pygame.event.post(pygame.event.Event(QUIT))


class Game:
    # intialize the pygame
    pygame.init()

    def move(self, action):
        if action == 0:
            self.change_to == 'RIGHT'
        if action == 1:
            self.change_to == 'LEFT'
        if action == 2:
            self.change_to == 'UP'
        if action == 3:
            self.change_to == 'DOWN'
        fps_clock = pygame.time.Clock()
        fps_clock.tick(self.fps)

        return self.reward

    def __init__(self):
        # set up the window
        self.window_width = 495
        self.window_height = 700
        self.window = pygame.display.set_mode(
            (self.window_width, self.window_height))
        pygame.display.set_caption('Snake Game')

        # colors
        self.black = pygame.Color(0, 0, 0)
        self.white = pygame.Color(255, 255, 255)
        self.red = pygame.Color(255, 0, 0)
        self.green = pygame.Color(0, 255, 0)
        self.blue = pygame.Color(0, 0, 255)

        # game FPS
        self.fps = 24

        # obstacles
        x1_position_obstacle = [100, 100]
        x2_position_obstacle = [200, 100]
        x3_position_obstacle = [300, 100]
        x4_position_obstacle = [400, 100]

        self.obstacles = [x1_position_obstacle,
                          x2_position_obstacle,
                          x3_position_obstacle,
                          x4_position_obstacle]

        self.obstacles_position = [x1_position_obstacle[0] / 400, x1_position_obstacle[1] / 400,
                                   x2_position_obstacle[0] /
                                   400, x2_position_obstacle[1] / 400,
                                   x3_position_obstacle[0] /
                                   400, x3_position_obstacle[1] / 400,
                                   x4_position_obstacle[0] / 400, x4_position_obstacle[1] / 400]

        self.width_obstacle = 30
        self.height_obstacle = 50

        # game variables
        self.score = 0
        self.reward = 0

        self.game_over = False

        self.snake_pos = [100, 50]
        self.snake_body = [[100, 50], [90, 50], [80, 50]]

        self.food_pos = [random.randrange(
            1, self.window_width//10)*10, random.randrange(1, self.window_height//10)*10]
        self.food_spawn = True

        self.direction = 'RIGHT'
        self.change_to = self.direction

        self.danger = False

        # proximity to walls: up, down, left, right
        self.proximity = [self.snake_pos[1], self.window_height - self.snake_pos[1] -
                          10, self.snake_pos[0], self.window_width - self.snake_pos[0] - 10]

        self.game_over = False

    # reset the game
    def reset(self):
        self.score = 0
        self.reward = 0
        self.game_over = False
        self.snake_pos = [100, 50]
        self.snake_body = [[100, 50], [90, 50], [80, 50]]
        self.food_pos = [random.randrange(
            1, self.window_width//10)*10, random.randrange(1, self.window_height//10)*10]
        self.food_spawn = True
        self.direction = 'RIGHT'
        self.change_to = self.direction
        self.danger = False
        self.proximity = [self.snake_pos[1] / 400, (self.snake_pos[0]) / 400]
        self.game_over = False

    # get the state of the game
    def get_state(self):
        if self.direction == 'RIGHT':
            direction_arr = [1, 0, 0, 0]
        if self.direction == 'LEFT':
            direction_arr = [0, 1, 0, 0]
        if self.direction == 'UP':
            direction_arr = [0, 0, 1, 0]
        if self.direction == 'DOWN':
            direction_arr = [0, 0, 0, 1]
        if self.game_over == True:
            game_over = 0
        if self.game_over == False:
            game_over = 1

        def divide_by_400(x): return [i/400 for i in x]

        snake_body = [[x/400, y/400] for x, y in self.snake_body]
        self.snake_head = snake_body[0]
        snake_tail = snake_body[-1]

        self.rel_food_pos = divide_by_400(self.food_pos)

        self.food_proximity = math.dist(self.rel_food_pos, self.snake_head)


        state = [self.score,
                 self.reward,
                 direction_arr,
                 self.proximity,
                 self.snake_head,
                 snake_tail,
                 self.food_proximity,
                 self.obstacles_position,
                 self.width_obstacle / 400,
                 self.height_obstacle / 400,
                 game_over]

        state = [x for x in state if not isinstance(x, int)]
        state = [x for x in state if not isinstance(x, float)]

        state = list(itertools.chain.from_iterable(state))

        return state

    def main(self):
        def game_over():
            self.reward -= 10
            self.game_over = True
            pygame.mixer.music.stop()
            game_over_font = pygame.font.SysFont('monaco', 72)
            game_over_screen = game_over_font.render(
                'Game Over', True, self.red)
            game_over_rect = game_over_screen.get_rect()
            game_over_rect.midtop = (self.window_width/2, self.window_height/4)
            self.window.blit(game_over_screen, game_over_rect)
            pygame.display.update()
            pygame.time.wait(500)
            pygame.quit()
            sys.exit()

        def show_score(choice, color, font, size):
            # game over function
            score_font = pygame.font.SysFont(font, size)
            score_surface = score_font.render(
                'Score : ' + str(self.score), True, color)
            score_rect = score_surface.get_rect()
            if choice == 1:
                score_rect.midtop = (self.window_width/10, 15)
            else:
                score_rect.midtop = (self.window_width/2,
                                     self.window_height/1.25)
            self.window.blit(score_surface, score_rect)

        def draw_snake():
            for pos in self.snake_body:
                pygame.draw.rect(self.window, self.green,
                                 pygame.Rect(pos[0], pos[1], 10, 10))
            pygame.draw.rect(self.window, self.red, pygame.Rect(
                self.food_pos[0], self.food_pos[1], 10, 10))

        # food spawn
        def spawn_food():
            if self.food_spawn == False:
                self.food_pos = [random.randrange(
                    1, self.window_width//10)*10, random.randrange(1, self.window_height//10)*10]
            for obs in self.obstacles:
                if (self.food_pos[0] >= obs[0] and (self.food_pos[0] <= obs[0]+self.width_obstacle-10) and self.food_pos[1] >= obs[1] and self.food_pos[1] <= obs[1]+self.height_obstacle-10):
                    self.food_pos[0] += 30
                    self.food_pos[1] += 50
            self.food_spawn = True

        def validate_direction():
            if self.change_to == 'RIGHT' and not self.direction == 'LEFT':
                self.direction = 'RIGHT'
            if self.change_to == 'LEFT' and not self.direction == 'RIGHT':
                self.direction = 'LEFT'
            if self.change_to == 'UP' and not self.direction == 'DOWN':
                self.direction = 'UP'
            if self.change_to == 'DOWN' and not self.direction == 'UP':
                self.direction = 'DOWN'

        def update_snake_position():
            if self.direction == 'RIGHT':
                self.snake_pos[0] += 10
            if self.direction == 'LEFT':
                self.snake_pos[0] -= 10
            if self.direction == 'UP':
                self.snake_pos[1] -= 10
            if self.direction == 'DOWN':
                self.snake_pos[1] += 10

        # main logic of the game
        while self.game_over == False:
            self.get_state()
            print("Snake Head: ", self.snake_head)
            print("Food Position: ", self.rel_food_pos)
            print("Food Proximity: ", self.food_proximity)

            self.proximity = [self.snake_pos[1] /
                              400, (self.snake_pos[0]) / 400]

            self.change_to = handle_key_press()

            # validation of direction
            validate_direction()
            # update snake position [x,y]
            update_snake_position()

            # snake body mechanism
            self.snake_body.insert(0, list(self.snake_pos))
            if self.snake_pos[0] == self.food_pos[0] and self.snake_pos[1] == self.food_pos[1]:
                self.reward += 10
                self.score += 1
                self.food_spawn = False
                self.width_obstacle = random.randrange(20, 60, 10)
                self.height_obstacle = random.randrange(20, 60, 10)
                x1_position_obstacle = [random.randrange(
                    1, self.window_width//10)*10, random.randrange(1, self.window_height//10)*10]
                x2_position_obstacle = [random.randrange(
                    1, self.window_width//10)*10, random.randrange(1, self.window_height//10)*10]
                x3_position_obstacle = [random.randrange(
                    1, self.window_width//10)*10, random.randrange(1, self.window_height//10)*10]
                x4_position_obstacle = [random.randrange(
                    1, self.window_width//10)*10, random.randrange(1, self.window_height//10)*10]

                self.obstacles_position = [x1_position_obstacle[0] / 400, x1_position_obstacle[1] / 400,
                                           x2_position_obstacle[0] /
                                           400, x2_position_obstacle[1] / 400,
                                           x3_position_obstacle[0] /
                                           400, x3_position_obstacle[1] / 400,
                                           x4_position_obstacle[0] / 400, x4_position_obstacle[1] / 400]

                self.obstacles = [x1_position_obstacle,
                                  x2_position_obstacle,
                                  x3_position_obstacle,
                                  x4_position_obstacle]
            else:
                self.snake_body.pop()

            spawn_food()
            # background
            self.window.fill(self.black)

            # draw snake
            draw_snake()

            # Game Over conditions
            if self.snake_pos[0] < 0 or self.snake_pos[0] > self.window_width-10:
                game_over()
            if self.snake_pos[1] < 0 or self.snake_pos[1] > self.window_height-10:
                game_over()
            for block in self.snake_body[1:]:
                if self.snake_pos[0] == block[0] and self.snake_pos[1] == block[1]:
                    game_over()
            for obs in self.obstacles:
                if (self.snake_pos[0] >= obs[0] and (self.snake_pos[0] <= obs[0]+self.width_obstacle-10) and self.snake_pos[1] >= obs[1] and self.snake_pos[1] <= obs[1]+self.height_obstacle-10):
                    game_over()

            # draw obstacles
            for pos in self.obstacles:
                if (self.snake_pos[0] >= pos[0] and (self.snake_pos[0] <= pos[0]+self.width_obstacle-10) and self.snake_pos[1] >= pos[1] and self.snake_pos[1] <= pos[1]+self.height_obstacle-10):
                    pygame.draw.rect(self.window, self.blue, pygame.Rect(
                        pos[0]+30, pos[1]+50, self.width_obstacle, self.height_obstacle))
                else:
                    pygame.draw.rect(self.window, self.blue, pygame.Rect(
                        pos[0], pos[1], self.width_obstacle, self.height_obstacle))

            pygame.draw.rect(self.window, self.red, pygame.Rect(
                self.food_pos[0], self.food_pos[1], 10, 10))

            show_score(1, self.white, 'monaco', 20)
            pygame.display.update()
            fps_clock = pygame.time.Clock()
            fps_clock.tick(self.fps)

MY = Game()
MY.main()