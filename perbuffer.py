import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Read the CSV file
csv_file = './data/resit_2/resit_2.csv'  # Replace with your file path
data = pd.read_csv(csv_file)

# Create a dictionary to store unique rewards and their associated game-frame identifiers
unique_rewards_dict = {}

def calculate_importance(rewards):
    rewards = np.array(rewards)  # Convert the rewards list to a NumPy array
    importances = np.power((np.abs(2 / (1 + np.exp(-rewards * np.log(99))) - 1) + 0.02), 1 / 2)
    importances /= np.sum(importances)

    return importances

def min_max_scale_rewards(data):
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

# Iterate over the CSV rows
for index, row in data.iterrows():
    reward = row['Reward']
    game_frame = (row['Game'], row['Frame'])

    if reward not in unique_rewards_dict.keys():
        unique_rewards_dict[reward] = []
        unique_rewards = list(unique_rewards_dict.keys())
        flatten_rewards = min_max_scale_rewards(unique_rewards)
        importances = calculate_importance(flatten_rewards)  # Call the function to recalculate importance
        # map game and frame to index in memory.
        # -> keep an array named frame_memory: 
    unique_rewards_dict[reward].append(game_frame)

# Print the unique rewards dictionary
print("Unique Rewards Dictionary:")
for reward, game_frames in unique_rewards_dict.items():
    print(f"Reward: {reward}")
    for game_frame in game_frames:
        print(f"  Game: {game_frame[0]}, Frame: {game_frame[1]}")


print("\nUnique Rewards:", unique_rewards)
print("Flatten Rewards:", flatten_rewards)
print("Normalized importances:", importances)



selected_reward = np.random.choice(unique_rewards, replace=True, p=importances)

# Randomly select a game-frame identifier associated with the selected reward
selected_game_frame = random.choice(unique_rewards_dict[selected_reward])

    # to do: find the selectefd game frame in the data 


print("\nSelected Reward:", selected_reward)
print("Selected Game-Frame:", selected_game_frame)



