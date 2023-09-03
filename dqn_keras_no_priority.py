from keras.layers import Dense, Activation
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import numpy as np
import itertools
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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
        
    def store_transition(self, state, action, reward, state_, done):
        # keep track of the index of the memory, reset if it exceeds the memory size
        index = self.mem_cntr % self.mem_size

        # state you're passing in
        self.state_memory[index] = state

        # new state after making an action
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

        self.mem_cntr += 1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        # Repeats wouldn't make a lot of sense...
        batch = np.random.choice(max_mem, size=batch_size, replace=False)
        print("Batch: " + str(batch))
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
        # decay for eps = 0.01 at 125 games : 0.99965
        # decay for eps = 0.1 at x games : 0.999965
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, temperature, 
                 mem_size, input_dims, epsilon_dec,  epsilon_end=0.01, fname='dqn_model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions,
                                   discrete=True)
        self.q_eval = build_dqn(lr, n_actions, input_dims, 50, 50, 50)
        self.temperature = temperature
        self.rand = 0

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

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