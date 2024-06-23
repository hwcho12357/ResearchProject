from absl import app
import numpy as np
import gym
import random
import matplotlib.pyplot as plt

def run_q_learning(env, qtable, total_steps=25000, learning_rate=0.8, gamma=0.95, 
                   epsilon=1.0, max_epsilon=1.0, min_epsilon=0.01, decay_rate=0.0001):
    max_steps_per_episode = 99
    steps = 0
    rewards = []
    cumulative_rewards = []
    cumulative_reward = 0
    avg_rewards = []

    while steps < total_steps:
        state = env.reset()
        episode_rewards = 0
        for _ in range(max_steps_per_episode):
            steps += 1
            exp_exp_tradeoff = random.uniform(0, 1)
            if exp_exp_tradeoff > epsilon:
                action = np.argmax(qtable[state, :])
            else:
                action = env.action_space.sample()

            new_state, reward, done, _ = env.step(action)
            qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
            
            episode_rewards += reward
            cumulative_reward += reward
            cumulative_rewards.append(cumulative_reward)
            avg_rewards.append(cumulative_reward / steps)  # Compute average reward per step
            
            state = new_state
            if done or steps >= total_steps:
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * steps)
        rewards.append(episode_rewards)
    
    return rewards, cumulative_rewards, avg_rewards, qtable

def main(argv):
    env = gym.make("FrozenLake-v1", is_slippery=False)
    action_size = env.action_space.n
    state_size = env.observation_space.n

    # Q-table initialized with zeros
    qtable_zeros = np.zeros((state_size, action_size))
    
    # Q-table initialized with DICE values (Replace with actual DICE values)
    qtable_dice = np.array([
    [0.50368656, 0.28417100, 0.24744200, 0.30284700],
    [0.00666858, 0.00369054, 0.00309231, 0.00149879],
    [0.02507504, 0.00371495, 0.00309155, 0.01011084],
    [0.05221552, 0.01277991, 0.00067256, 0.00430612],
    [0.26031873, 0.00403238, 0.00468556, 0.00260281],
    [0.00233800, 0.00251356, 0.00273770, 0.00179974],
    [0.02266099, 0.00031998, 0.00077225, 0.00000000],
    [0.00000000, 0.00000000, 0.00000000, 0.01678710],
    [0.00223199, 0.00417178, 0.00007266, 0.13045901],
    [0.01144835, 0.11829945, 0.00535556, 0.00362227],
    [0.00075249, 0.01784247, 0.00250205, 0.00025089],
    [0.01815625, 0.00211173, 0.00307075, 0.00129442],
    [0.00279297, 0.00271804, 0.00299118, 0.00082534],
    [0.00014303, 0.00039642, 0.02397827, 0.00021320],
    [0.00124678, 0.01325846, 0.00001431, 0.00456260],
    [0.00000000, 0.00000000, 0.00000000, 0.00000000],
])


    # Run Q-learning with zero-initialized Q-table
    rewards_zero, cumulative_rewards_zero, avg_rewards_zero, qtable_zeros = run_q_learning(env, qtable_zeros)
    
    # Run Q-learning with DICE-initialized Q-table
    rewards_dice, cumulative_rewards_dice, avg_rewards_dice, qtable_dice = run_q_learning(env, qtable_dice)

    # Plot cumulative rewards for comparison
    plt.figure(figsize=(12, 8))
    plt.plot(avg_rewards_zero, label='Zero-initialized Q-values')
    plt.plot(avg_rewards_dice, label='DICE-initialized Q-values')
    plt.title('Average Rewards Over Time for Q-Learning on FrozenLake')
    plt.xlabel('Steps')
    plt.ylabel('Average Rewards')
    plt.legend()
    plt.grid(True)
    plt.savefig('QLearning_AverageRewards_Steps3.png')
    plt.show()

if __name__ == '__main__':
  app.run(main)