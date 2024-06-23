from absl import app
import numpy as np
import gym
import random
import matplotlib.pyplot as plt


def main(argv):
    observations = np.loadtxt('observations.txt')
    actions = np.loadtxt('actions.txt')
    weights = np.loadtxt('weight.txt')

    # Assuming observations, actions, and weights are aligned row-wise
    # and the environment has a fixed number of states and actions
    num_states = int(np.max(observations)) + 1  # Number of states
    num_actions = int(np.max(actions)) + 1  # Number of actions

    # Initialize the Q-table with zeros
    Q_table = np.zeros((num_states, num_actions))

    # Populate the Q-table with weights
    for state, action, weight in zip(observations, actions, weights):
        Q_table[int(state), int(action)] = weight

    print(Q_table)

if __name__ == '__main__':
  app.run(main)