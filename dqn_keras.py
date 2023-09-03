from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np
import itertools
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random



# Memory of the agent
class ReplayBuffer(object):
    def __init__(self, max_size, input_shape, n_actions, discrete=True):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.discrete = discrete

        # Initialize the memory arrays with zeros
        self.state_memory = np.zeros((self.mem_size, input_shape))
        self.new_state_memory = np.zeros((self.mem_size, input_shape))
        dtype = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
        
        # --------- PER MEMORY -------------
        ## Useful for keeping track of the game, frame and index in memory
        self.frame_memory = np.zeros((self.mem_size, 2))

        ## Useful for keeping track of the unique rewards and their associated game-frame identifiers 
        self.unique_rewards_dict = {}

        ## Useful for keeping track of the chances of being selected for training
        self.importance_memory = np.zeros(self.mem_size)
        # ----------------------------------


    # ------ PER REWARD SCALING AND IMPORTANC CALCULATION ------
    def min_max_scale_rewards(self, data):
        # Create an instance of MinMaxScaler with feature_range=(-1, 1)
        scaler = MinMaxScaler(feature_range=(-1, 1))

        # Add -10 and +10 entries to the data
        #data_with_bounds = np.concatenate((data, [-1, 1]))

        # Reshape the array into a 2D shape (required by MinMaxScaler)
        data_reshaped = [[value] for value in data]

        # Fit the scaler on the data and transform the values
        scaled_data = scaler.fit_transform(data_reshaped)

        # Flatten the scaled data array back to 1D
        scaled_data_flat = [value[0] for value in scaled_data]

        return scaled_data_flat

    def calculate_importance(self, rewards):
        rewards = np.array(rewards)  # Convert the rewards list to a NumPy array
        importances = np.power((np.abs(2 / (1 + np.exp(-rewards * np.log(99))) - 1) + 0.02), 1 / 2)
        importances /= np.sum(importances)
        return importances   
    # -----------------------------------------------------------
    
    def store_transition(self, state, action, reward, state_, done, game, frame):
        index = self.mem_cntr % self.mem_size

        # --------- PER ------------
        # Append the game and frame to the 2 dimensional array named frame_memory:
        self.frame_memory[index] = [game, frame]
        # --------------------------
        self.state_memory[index] = state
        self.new_state_memory[index] = state_

        # store one hot encoding of actions, if appropriate
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done


        # --------- PER ------------
        if reward not in self.unique_rewards_dict.keys():
            self.unique_rewards_dict[reward] = []
            self.unique_rewards = list(self.unique_rewards_dict.keys())
            print("Unique Rewards: " + str(self.unique_rewards))
            flatten_rewards = self.min_max_scale_rewards(self.unique_rewards)
            self.importances = self.calculate_importance(flatten_rewards)

        self.unique_rewards_dict[reward].append((game, frame))
        # --------------------------
        self.mem_cntr += 1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        selected_rewards = []
        selected_game_frames = []
        batch = []
        # For loop for the batch:
        for i in range(batch_size):
            # Prioritized selection of the reward:
            selected_reward = np.random.choice(self.unique_rewards, replace=True, p=self.importances)

            # Random frame with associated reward:
            selected_game_frame = random.choice(self.unique_rewards_dict[selected_reward])

            # Find the index of the selected game-frame in the frame_memory array
            index = np.where((self.frame_memory == selected_game_frame).all(axis=1))[0][0]

            selected_game_frames.append(selected_game_frame)
            selected_rewards.append(selected_reward)
            batch.append(index)
        print("SELECTED REWARDS: " + str(selected_rewards))
        print("SELECTED GAME FRAMES: " + str(selected_game_frames))
        print("BATCH LOOKS LIKE THIS: " + str(batch))
        
        
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

# Build Network
def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims, fc3_dims):
    model = Sequential([
                Dense(fc1_dims, input_shape=(input_dims,)),
                Activation('relu'),
                Dense(fc2_dims),
                Activation('relu'),
                Dense(fc3_dims),
                Activation('relu'),
                Dense(n_actions)])

    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

    return model

class Agent(object):
        # decay for eps = 0 at 500 games : 0.990823
        # decay for eps = 0.17 at 100 games : 0.99965
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, temperature,
                 input_dims, epsilon_dec=0.99965,  epsilon_end=0.01,
                 mem_size=1000000, fname='dqn_model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions,
                                   discrete=True)
        #self.memory = PrioritizedReplayBuffer(mem_size, input_dims, n_actions, discrete=True)
        self.q_eval = build_dqn(lr, n_actions, input_dims, 50, 50, 50)
        self.temperature = temperature
        self.rand = 0

    def remember(self, state, action, reward, new_state, done, game, frame):
        self.memory.store_transition(state, action, reward, new_state, done, game, frame)

    def choose_action(self, state):
        state = np.array(state)
        state = state[np.newaxis, :]
        self.rand = np.random.random()
        print("Random number: " + str(self.rand))
        print("Epsilon: " + str(self.epsilon))
        if self.rand < self.epsilon:
            print("------------------------Random Action Chosen------------------------------")
            action = np.random.choice(self.action_space)
            print("Action Chosen By Agent: " + str(action))
            print("---------------------------------------------------------------------------")
        else:
            print("------------------------Action Chosen By Agent-----------------------------")
            actions = self.q_eval.predict(state)
            
            action = np.argmax(actions)
            print("All Possible Actions: " + str(actions))
            print("Action Chosen By Agent: " + str(action)) 
            print("---------------------------------------------------------------------------")
        return action

    def learn(self):
        # dont learn if there is not enough memory
        if self.memory.mem_cntr < self.batch_size:
            return
        # do learn if there is enough memory
        if self.memory.mem_cntr > self.batch_size:
            state, action, reward, new_state, done = \
                                          self.memory.sample_buffer(self.batch_size)

            action_values = np.array(self.action_space, dtype=np.int8)
            action_indices = np.dot(action, action_values)

            q_eval = self.q_eval.predict(state)

            q_next = self.q_eval.predict(new_state)

            q_target = q_eval.copy()

            batch_index = np.arange(self.batch_size, dtype=np.int32)

            q_target[batch_index, action_indices] = reward + \
                                  self.gamma*np.max(q_next, axis=1)*done

            _ = self.q_eval.fit(state, q_target, verbose=0)

            self.epsilon = self.epsilon*self.epsilon_dec if self.epsilon > \
                           self.epsilon_min else self.epsilon_min

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)