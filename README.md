# Bachelor Thesis HoGent Strijk Philippe
# Reinforcement Learning in Snake. A DQN Agent with Prioritized Experience Replay.

This bachelor thesis was written as part of my studies for a "Bachelor in Applied Computer Science" with a specialization in "Data Engineering". I look forward to continuing my academic journey next year at VUB and pursuing a Master of Science in Applied Computer Science: Artificial Intelligence.

## Introduction

This repository contains the code and research findings for training a Reinforcement Learning (RL) agent to play a custom Snake game. The study aims to evaluate the agent's performance with a focus on prioritized experience replay (PER).

## Acknowledgments

I would like to express my gratitude to my supervisor for their valuable guidance and oversight throughout the completion of this thesis. I also extend my appreciation to my co-supervisor and dear friend, Manu, for their mentorship and assistance whenever I faced challenges along the way. Additionally, I would like to thank Merel for helping me bring out the best in myself. Finally, I am deeply grateful to my parents for their unwavering support and belief in me throughout my academic journey.

## Preliminary Baseline

Before delving into the exploration of customized replay buffers or even providing an overview of the methodology, it's crucial to establish baseline measurements that provide context for the subsequent discussions in this chapter.

## Key Parameters

In addition, the evaluation includes a selection of key parameters, each carefully chosen to tailor the learning process to the constraints of the study.

### State

The input parameters for this measurement are set at 11, with the parameters being:
- 4 Direction Parameters (One-Hot Encoded): [right, left, up, down]
- 2 Position Parameters for the head: (x, y)
- 2 Position Parameters for the tail: (x, y)
- 2 Position Parameters for the food: (x, y)
- 1 Parameter for Game Over: 1 or 0

## Baseline Measurements

After 155 gaming sessions and observing a total of 24,130 frames, a series of measurements were taken. These measurements were chosen with care to provide insight into the performance and behavior of the baseline approach.

### Summary Metrics for Baseline Analysis

The average number of frames provides valuable insight into determining an appropriate epsilon-decay strategy for a given number of games and provides a fairly accurate estimate for different configurations.

## Epsilon-Longevity

This graph shows the number of frames the snake survived in the game in combination with the decaying exploration rate.

## Epsilon-Score

The following figure shows the epsilon decay per game on the left Y-axis, contrasted with the maximum score achieved per game on the right Y-axis.

## Baseline Measurement Analysis

In examining the preliminary results, valuable insight into the stochastic nature of the agent's play and how it affects its performance is gained.

### Increasing Training and Exploration

One strategy to address the agent's stochastic play is to focus on increasing the number of games played.

### Adjusting the Learning Rate

Given the small number of non-zero reward states, the network has difficulty finding a consistent direction for updating weights.

### Increasing the Batch Size

Increasing the batch size amplifies the statistical probability of including a rare event in the batch.

## Limitations

While the study's scope is constrained, several avenues for exploration and optimization remain open.

## Methodology Overview

The following part of this chapter will refrain from presenting new measurements or results. Instead, the results section discusses the improved baseline and PER measurements.

### Primary Goal

The primary goal is to assess the impact of the custom PER buffer implementation.

## Performance

Performance is evaluated by analyzing the metrics and graphs presented in the preceding section.

### The enhanced baseline and PER buffer measurement have the following adjusted hyperparameters:
- Total amount of frames trained: 43,349
- Learning Rate: 0.001
- Epsilon Decay: 0.99985
- Batch Size: 24

### Key Methodological Details

The Snake Game
In order to facilitate learning from the game, an architecture is established where the agent can interact with the snake class to perform the following operations:
- Retrieve the current state
- Determine the next state based on a given state and action
All state parameters are normalized within the range of 0 and 1.

### DQN Agent

This DQN Agent was inspired by data scientist, and expert instructor Phil Tabor. The DQN Agent comprises the following components:
- Neural network
- Action selection mechanism
- Learning step

### Neural Network

For the neural network, a Keras sequential model is employed. It consists of an input layer with a size corresponding to the state dimensions, followed by three dense layers with 50 neurons each, and ReLU activation functions. The output layer maps the network to the three available action choices.

### Action Selection

The agent employs an epsilon-greedy policy to determine the chosen action.

### Epsilon Action & Decay

To illustrate the concept of reducing the randomness in the agent's actions over time, let's consider an example using an epsilon (￿).

### Q Value Action

If the randomly generated number between 0 and 1 is greater than epsilon, the neural network is utilized to make a decision through the predict function.

### Prioritized Experience Replay

The Prioritized Experience Replay mechanism is constructed by collecting distinct rewards encountered during the training process.

### Memory

The replay buffer organizes the memory into separate components for efficient implementation.

### Importance Function & Calculation

In the learning step, it is important to prioritize states that yield positive or negative rewards, giving them a higher probability of being sampled.

### Hyperparameter Selection

In this section, we discuss and establish the hyperparameters that are relevant to our model.

### Controller

The controller is responsible for coordinating and integrating the various components of the Snake game environment and the Deep Q-Network (DQN) agent.

### Game Analysis

Game analysis is conducted by recording all the traversed states in a .csv file.

## Running & Evaluating Results

### Running the Experiment

Due to limited resources, the experiment will be conducted on a personal laptop with the following specifications:
- Microprocessor: AMD Ryzen 7 5800H with Radeon Graphics
- System memory: 16 GB (8GB Samsung 3200MHz in each memory slot)
- Video: AMD Radeon(TM) Graphics and Radeon RX 5500M

For future researchers, it is advisable to consider running the model without the pygame library.

### Evaluating Results

The evaluation of this setup involves assessing its average reward, exploration-exploitation balance, longevity, and visual inspection.

## Implications

In comparison to previous research conducted by DeepMind (Mnih et al., 2015) in the field of Reinforcement Learning in gaming, it is worth noting the disparity in the number of frames used for training.

## Contributions to Knowledge

The application of a sigmoidal curve as an importance function in a prioritized experience replay buffer is a novel and relatively unexplored approach.

## Methodological Reflections & Recommendations for Future Work

While the study's scope is constrained, several avenues for exploration and optimization remain open.

## Conclusion

The study concludes by summarizing the research's objectives and outcomes, emphasizing the potential for further advancements in reinforcement learning techniques.

