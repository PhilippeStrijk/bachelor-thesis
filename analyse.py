import pandas as pd
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
import numpy as np
from scipy.stats import linregress
import pandas as pd
import matplotlib.pyplot as plt
from tkinter import *

# Lees het CSV-bestand in een pandas DataFrame
data = pd.read_csv('./data/resit_3/resit_3.csv')

# Haal de positiegegevens uit de observation kolom
data['x_position'] = data['Observation'].apply(lambda obs: float(obs.split(',')[4]))
data['y_position'] = data['Observation'].apply(lambda obs: float(obs.split(',')[5]))

print(data.head())
def update_heatmap(*args):
    epsilon_value = epsilon_slider.get()
    games_value = games_slider.get()
    
    filtered_data = data[(data['Epsilon Value'] >= epsilon_value) & (data['Game'] >= games_value)]
    print(filtered_data.head())
    if filtered_data.empty:
        print("No data found for the specified epsilon and number of games.")
        return
    
    # Extract x and y coordinates from the position column
    # Ensure that the columns contain numerical values
    filtered_data['x_position'] = pd.to_numeric(filtered_data['x_position'])
    filtered_data['y_position'] = pd.to_numeric(filtered_data['y_position'])

    
    # Create a 2D histogram
    hist, xedges, yedges = np.histogram2d(filtered_data['x_position'], filtered_data['y_position'], bins=(50, 50))
    
    plt.imshow(hist.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='YlOrRd', aspect='auto')
    plt.colorbar()
    plt.title(f"Heatmap for Epsilon >= {epsilon_value} and Number of Games >= {games_value}")
    plt.xlabel("X-coordinate of Position")
    plt.ylabel("Y-coordinate of Position")
    plt.show()




def update_plot(*args):
    games_data = data.groupby('Game')['Frame'].max()
    epsilon_values = data.groupby('Game')['Epsilon Value'].mean()
    
    fig, ax1 = plt.subplots()
    
    # Plot epsilon values on the left y-axis
    color = 'tab:blue'
    ax1.set_xlabel('Games')
    ax1.set_ylabel('Epsilon Value', color=color)
    ax1.plot(games_data.index, epsilon_values.values, color=color, marker='o', label='Epsilon Value')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    
    # Plot frames and linear regression on the right y-axis
    color = 'green'
    ax2.plot(games_data.index, games_data.values, color=color, marker='o', label='Frames')
    
    slope, intercept, _, _, _ = linregress(games_data.index, games_data.values)
    regression_line = slope * games_data.index + intercept
    
    ax2.plot(games_data.index, regression_line, color='red', label='Linear Regression')
    
    ax2.set_ylabel('Frames', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    
    fig.tight_layout()
    plt.title("Epsilon Value and Frames in function of Games with linear regression")
    plt.xlabel("Games")
    plt.legend()
    plt.grid(True)
    plt.show()


def update_combined_plot(*args):
    games_data = data.groupby('Game')['Frame'].max()
    top_scores = data.groupby('Game')['Score'].max()

    epsilon_values = []
    for game, max_frame in games_data.items():
        epsilon_values.append(data[(data['Game'] == game) & (data['Frame'] == max_frame)]['Epsilon Value'].iloc[0])

    fig, ax1 = plt.subplots()

    # Plot epsilon values for the maximum frame per game
    color = 'tab:blue'
    ax1.set_xlabel('Number of Games')
    ax1.set_ylabel('Epsilon Value', color=color)
    ax1.plot(games_data.index, epsilon_values, color=color, marker='o', label='Epsilon Value')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 1)

    # Create a twin Axes sharing the xaxis
    ax2 = ax1.twinx()

    # Plot top scores per game
    color = 'green'
    ax2.set_ylabel('Top Score per Game', color=color)
    ax2.plot(top_scores.index, top_scores.values, 'go', label='Top Score')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title("Epsilon Value and Top Score per Game")
    plt.xlabel("Number of Games")
    plt.grid(True)
    plt.show()

# Calculate and print the mean number of frames per game using the maximum frame value per game
mean_frames_per_game = data.groupby('Game')['Frame'].max().mean()
print(f"Mean number of frames per game: {mean_frames_per_game:.2f}")


# Count frames with reward 0 and non-zero rewards
frames_with_zero_reward = (data['Reward'] == 0).sum()
frames_with_nonzero_reward = len(data) - frames_with_zero_reward

# Calculate the percentage of non-zero rewards
percentage_nonzero_reward = (frames_with_nonzero_reward / len(data)) * 100
#total number of frames
print(f"Total number of frames: {len(data)}")
print(f"Frames with reward 0: {frames_with_zero_reward}")
print(f"Frames with non-zero reward: {frames_with_nonzero_reward}")
print(f"Percentage of non-zero rewards: {percentage_nonzero_reward:.2f}%")

""" def find_n_below_threshold(x, value=1, n=1):
    if value < 0.5:
        return n
    else:
        return find_n_below_threshold(x, value * x, n + 1)

x = 0.9999965  # Initial value of x
result_n = find_n_below_threshold(x)
print(f"The smallest n for which f(n) < 0.5 is: {result_n}") """


# GUI setup
root = Tk()
root.title("Data Analyse")

epsilon_label = Label(root, text="Selecteer Epsilon:")
epsilon_label.pack()
epsilon_slider = Scale(root, from_=0, to=1, resolution=0.01, orient=HORIZONTAL, length=200)
epsilon_slider.pack()
epsilon_slider.set(0.5)

games_label = Label(root, text="Selecteer Aantal Games:")
games_label.pack()
games_slider = Scale(root, from_=0, to=100, orient=HORIZONTAL, length=200)
games_slider.pack()
games_slider.set(50)

heatmap_button = Button(root, text="Generate Heatmap", command=update_heatmap)
heatmap_button.pack()

plot_button = Button(root, text="Generate Longevity Plot", command=update_plot)
plot_button.pack()

combined_plot_button = Button(root, text="Generate Epsilon-Score Plot", command=update_combined_plot)
combined_plot_button.pack()

root.mainloop()
