from dqn_keras import Agent
import numpy as np
from snake2 import Game
from utils import plotLearning
import time


class SnakeEnvironment:
    def __init__(self):
        # initialize your Snake game
        self.game = Game()
        self.state = None
    def reset(self):
        # reset the game and return the initial observation
        self.game.reset()
        self.state = self.game.get_state()
        return self.state

    def step(self, action):
        # perform the given action in the game and return the next observation, reward, done flag, and info

        # Implement closeness to food
        #self.game.reward = 1/self.game.food_proximity
        reward = self.game.reward
        done = self.game.game_over

        if done == True:
            reward -= 10

        if self.game.direction == "RIGHT":
            if action == 0:
                action = 2
            elif action == 1:
                action = 0
            elif action == 2:
                action = 3
        elif self.game.direction == "LEFT":
            if action == 0:
                action = 3
            elif action == 1:
                action = 1
            elif action == 2:
                action = 2
        elif self.game.direction == "UP":
            if action == 0:
                action = 1
            elif action == 1:
                action = 2
            elif action == 2:
                action = 0
        elif self.game.direction == "DOWN":
            if action == 0:
                action = 0
            elif action == 1:
                action = 3
            elif action == 2:
                action = 1

        self.game.next_state(self.game.get_state, action)

        self.state = self.game.get_state()
        
        return self.state, reward, done, {}

    def render(self):
        # display the current state of the game
        self.game.display()


if __name__ == '__main__':
    env = SnakeEnvironment()
    n_games = 2000
    temperature = 0
    model_name = "final_try.h5"
    state_size = 18
    num_actions = 3


    agent = Agent(gamma=0.6, epsilon=1, lr=0.002,
                  input_dims=state_size, n_actions=3, mem_size=1000000,
                  batch_size=6, epsilon_end=0.01, temperature=temperature)

    scores = []
    eps_history = []

    for i in range(n_games):
        print("Game: ", i)
        done = False
        score = 0
        observation = env.reset()
        
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            print(observation)
            score += reward
            print("Score: ", score)
            agent.remember(observation, action, reward,
                           observation_, int(done))
            observation = observation_.copy()
            agent.learn()

        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[max(0, i-100):(i+1)])
        print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % avg_score)
        if i % 10 == 0 and i > 0:
            agent.save_model()

    x = [i+1 for i in range(n_games)]
    plotLearning(x, scores, eps_history, filename='snake_final_try.png')
