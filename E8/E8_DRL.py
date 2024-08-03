#%%
# Import libraries
import gym
import numpy as np
import random

from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam 
import matplotlib.pyplot as plt
 
class DQNAgent():
    def __init__(self, env_id, epsilon_decay=0.9999, epsilon=1.0, epsilon_min=0.001, 
            gamma=0.95, alpha=0.01, alpha_decay=0.01, batch_size=16):

        self.memory = deque(maxlen=50000) # memory length
        self.env = gym.make(env_id) # create environment with env_id

        self.state_size = self.env.observation_space.shape[0] 
        self.action_size = self.env.action_space.n # total number of actions

        self.epsilon = epsilon # probability the agent will explore (initial value is 1.0)
        self.epsilon_decay = epsilon_decay # decay multiplied with epsilon after each episode
        self.epsilon_min = epsilon_min # minimum value of epsilon 
        self.gamma = gamma # discount factor
        self.alpha = alpha # learning rate
        self.alpha_decay = alpha_decay # learning rate decay factor
        self.batch_size = batch_size # number of samples used for training

        self.model = self._build_model()

    # Creating deep neural network model to output Q-values 
    def _build_model(self):
        # Sequential API used to create the neural network
        model = Sequential() # Sequential() creates the foundation of layers
        # Densely connected layers used due to simple working environment
        model.add(Dense(24, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(24, activation='tanh'))
        model.add(Dense(self.action_size, activation='linear')) # get Q-value for each action
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.alpha, decay=self.alpha_decay))
        return model
    


    def act(self, max_steps, train): 
        # TODO: 
        # Take the corresponding action (random or the best action) based on epsilon-greedy exploration strategy
        ges_reward = 0
        state, info = self.env.reset()
        for i in range(max_steps):
            if np.random.rand() < self.epsilon and train:
                action = self.env.action_space.sample()

            else:
                action_values = self.model.predict(np.reshape(state, (1,self.state_size)))
                action = np.argmax(action_values[0])
  

            next_state, reward, terminate, trunc , info = self.env.step(action)

            ges_reward += reward


            if train:

                self.remember(state,action,reward, next_state, terminate)

                self.replay()

            state = next_state

            if terminate:
                break

        return ges_reward, i


    def remember(self, state, action, reward, next_state, done): 
        # TODO: 
        # Store the experience in the knowledge base (memory) for the purpose of training DNN using experience replay
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        # Randomly sample experience from the memory
        x_batch, y_batch = [], []
        # Take a batch if possible
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
        # Train DNN through experience replay process (iterate through all the samples)
        for state, action, reward, next_state, done in minibatch:
            y_target = self.model.predict(np.reshape(state, (1,self.state_size))) # predict the action for the state "state"
            if done: # If the episode ended, take the reward as the Q-value
                y_target[0][action] = reward
            else: # If the episode continues, take the sum of the reward and discounted future reward as the Q-value
                y_target[0][action] = reward + self.gamma * np.max(self.model.predict(np.reshape(next_state, (1,self.state_size)))[0])
            x_batch.append(np.reshape(state, (1,self.state_size))[0])
            y_batch.append(y_target[0])
        # Train the model on the pairs of states (input to the model) and Q-values (output to the model)
        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch), verbose=0)
        # Decrease epsilon --> the agent will explore less and less
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ID of the working environment
env_id = "CartPole-v1"

# Variables for training/testing
test_episodes = 20   # number of episodes for testing
train_episodes = 80  # number of episodes for training
episodes = train_episodes + test_episodes   # total number of episodes
max_steps = 100     # maximum number of steps per episode


# TODO:
# Implement DQN algorithm to train the agent to balance a pole for as long as possible
# Plot the reward with respect to each episode (during the training and testing phases).
# Plot the number of steps taken with each episode (during the training and testing phases).

agent = DQNAgent(env_id)

reward_list = []
steps_list = []

for i in range(train_episodes):
    print(f'Episode {i}/{train_episodes}')
    reward, steps = agent.act(max_steps, True)
    reward_list.append(reward)
    steps_list.append(steps)

for i in range(test_episodes):
    print(f'Episode{i}/{test_episodes}')
    reward, steps = agent.act(max_steps, False)
    reward_list.append(reward)
    steps_list.append(steps)


# Plotting rewards per episode
plt.figure(figsize=(10, 5))
plt.plot(range(episodes), reward_list, label='Rewards')
plt.axvline(x=train_episodes, color='r', linestyle='--', label='Test-Boundary')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Rewards per Episode')
plt.legend()
plt.savefig('rewards_per_episode.png')



# Plotting steps per episode
plt.figure(figsize=(10, 5))
plt.plot(range(episodes), steps_list, label='Training')
plt.axvline(x=train_episodes, color='r', linestyle='--', label='Test-Boundary')
plt.xlabel('Episode')
plt.ylabel('Steps')
plt.title('Steps per Episode')
plt.legend()
plt.savefig('steps_per_episode.png')

plt.show()

