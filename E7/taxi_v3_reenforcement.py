#%%
# Import libraries
import gym
import numpy as np
import matplotlib.pyplot as plt

# Create Taxi environment
env = gym.make("Taxi-v3")
 
# Q-table represents the rewards (Q-values) the agent can expect performing a certain action in a certain state
state_space = env.observation_space.n # total number of states
action_space = env.action_space.n # total number of actions
qtable = np.zeros((state_space, action_space)) # initialize Q-table with zeros


# Variables for training/testing
test_episodes = 10000   # number of episodes for testing
train_episodes = 40000  # number of episodes for training
episodes = train_episodes + test_episodes   # total number of episodes
max_steps = 100     # maximum number of steps per episode

# Q-learning algorithm hyperparameters to tune
# Q-learning algorithm hyperparameters to tune
alpha = 0.3  # learning rate: you may change it to see the difference
gamma = 0.75  # discount factor: you may change it to see the difference


# Exploration-exploitation trade-off
epsilon = 1.0           # probability the agent will explore (initial value is 1.0)
epsilon_min = 0.001     # minimum value of epsilon 
epsilon_decay = 0.999 # decay multiplied with epsilon after each episode

# TODO:
# Implement Q-learning algorithm to train the agent to be a better taxi driver. 
# Plot the reward with respect to each episode (during the training and testing phases).
# Plot the number of steps taken with each episode (during the training and testing phases).

#Lists for later plots
steps_train = []
steps_test = []
rewards_train = []
rewards_test = []


#training
for _ in range(train_episodes):
    
    #Flag for termination of episode
    terminate = False

    #initialize stepcounter and reward sum
    steps = 0
    ges_reward = 0
    
    #Initialize random state and reset
    state = env.reset()

    for _ in range(max_steps):
        steps += 1
        if np.random.rand() < epsilon:
            #choose randomly
            action = env.action_space.sample()
        else:
            #choose action with highest q-value
            action = np.argmax(qtable[state[0],:])
        
        
        #Take action
        next_state, reward, terminate, trunc ,info = env.step(action)

        ges_reward += reward

        #Check if terminal in terminal state
        if terminate:
            break

        #Update Q-table using given formula
        qtable[state[0],action] = (1-alpha) * qtable[state[0],action] + alpha * (reward + gamma * np.max(qtable[next_state,:]))

        #Update state
        state = (next_state, info)

        
    steps_train.append(steps)
    rewards_train.append(ges_reward)

    
    #Update epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
    else:
        epsilon = epsilon_min







# Plotting rewards per episode
plt.figure(figsize=(10, 5))
plt.plot(range(train_episodes), rewards_train, label='Training')
#plt.plot(range(train_episodes, episodes), rewards_test, label='Testing')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards per Episode')
plt.legend()


# Plotting steps per episode
plt.figure(figsize=(10, 5))
plt.plot(range(train_episodes), steps_train, label='Training')
#plt.plot(range(train_episodes, episodes), steps_test, label='Testing')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode')
plt.legend()
plt.show()
