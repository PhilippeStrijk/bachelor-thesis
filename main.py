from dqn_keras import Agent
import numpy as np
from snake_no_obstacles import Game
import time
import csv
import atexit
import os

csv_file_path = './data/resit_3/resit_3.csv'
data = ['Game', 'Frame', 'Score', 'Epsilon Value', 'Random Generated Number', 'Action', 'Observation', 'Next Observation', 'Reward', 'Done']

if os.path.exists(csv_file_path):
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        data = [row for row in csv_reader]


def save_to_csv(row):
    with open(csv_file_path, 'a', newline='') as csv_file:
        fieldnames = row.keys()
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        
        if csv_file.tell() == 0:  # Check if the file is empty
            csv_writer.writeheader()  # Write header only if the file is empty
        csv_writer.writerow(row)  # Append the data row




atexit.register(save_to_csv)



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
            reward -= 5

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

        self.state = self.game.get_state()
        self.game.next_state(self.state, action)
        return self.state, reward, done, {}, self.game.frame, self.game.score

    def render(self):
        # display the current state of the game
        self.game.display()


if __name__ == '__main__':

    env = SnakeEnvironment()
    n_games = 1000
    temperature = 0
    model_name = "resit_3.h5"
    state_size = 11
    num_actions = 3


    agent = Agent(gamma=0.6, epsilon=1, lr=0.002,
                  input_dims=state_size, n_actions=3, mem_size=45000,
                  batch_size=24, epsilon_dec=0.99985, epsilon_end=0.01, temperature=temperature)

    scores = []
    eps_history = []

    for i in range(n_games):
        print("Game: ", i)
        done = False
        score = 0
        observation = env.reset()
        # observation[0] = Omhoog JA / NEE
        # observation[1] = Rechts JA / NEE

        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info, frame, score = env.step(action)

            print("Frame: ", frame)
            print("OBSERVATION ----->>> " + str(observation_))
            print("RIGHT ----->>> " + str(observation_[0]))
            print("LEFT ----->>> " + str(observation_[1]))
            print("UP ----->>> " + str(observation_[2]))
            print("DOWN ----->>> " + str(observation_[3]))
            print("SNAKE HEAD POSITION ----->>> " + str(observation_[4]) + " : " + str(observation_[5]))
            print("SNAKE TAIL POSITION ----->>> " + str(observation_[6]) + " : " + str(observation_[7]))
            print("FOOD POSITION ----->>> " + str(observation_[8])+ " : " + str(observation_[9]))
            print("GAME OVER ----->>> " + str(observation_[10]))
            print("SCORE ----->>> " + str(score))
            
            # Save all necessary data into a .csv for later analysis
            row = {'Game': i, 'Frame': frame, 'Score': score, 'Epsilon Value': agent.epsilon, 'Random Generated Number': agent.rand, 'Action': action, 'Observation': observation, 'Next Observation':observation_, 'Reward': reward, 'Done': done}
            save_to_csv(row)
            print("Data added:", row)
            agent.remember(observation, action, reward,
                           observation_, int(done), game = i, frame = frame)


            observation = observation_.copy()
            agent.learn()

        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[max(0, i-100):(i+1)])
        print('episode ', i, 'score %.2f' % score,
              'average score %.2f' % avg_score)
        if i % 10 == 0 and i > 0:
            agent.save_model()