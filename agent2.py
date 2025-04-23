#%%
import gymnasium as gym
import numpy as np
from blackjack import BlackjackEnv
import matplotlib.pyplot as plt
from BlackjackAgent import BlackjackSARSA
#%%
# Create the Blackjack environment
env = BlackjackEnv(natural=False, sab=False)

# Set an action map
action_map = {0: "Stick", 1: "Hit"}

n_episodes = 1000000
start_epsilon = 1.0

agent = BlackjackSARSA(
    learning_rate=0.01,
    initial_epsilon=1.0,
    epsilon_decay= start_epsilon / (n_episodes / 2),
    final_epsilon=0.1,
    discount_factor=0.95
)

bigArray = []

#%%
# Episode loop for training
for episode in range(n_episodes):
    # Get the initial state by resetting the environment
    state, info = env.reset()
    terminated = [False, False]

    array1 = [(state[:3])]
    array2 = [(state[3:])]
    
    # Main loop for one had of blackjack
    while not terminated[0] and not terminated[1]:
        # Obtain  random action
        #action1 = int(env.action_space.sample())
        #action2 = int(env.action_space.sample())

        action1 = int(BlackjackSARSA.get_action(self = agent, env = env,  obs = (state[:3])))
        action2 = int(BlackjackSARSA.get_action(self = agent, env = env, obs = (state[3:])))
        
        count1 = 0
        count2 = 0

        # Update the state of the environment using env.step()
        next_state, reward, terminated, truncated, info = env.step((action1, action2))
        if count1 < 1:
            array1.append(action1)
            array1.append(reward[0])
            array1.append(next_state[:3])
            if reward[0] != 0:
                array1.append(2.0)
                count1+=1
        if count2 < 1:
            array2.append(action2)
            array2.append(reward[1])
            array2.append(next_state[3:])
            if reward[1] != 0:
                array2.append(2.0)
                count2+=1

        state = next_state
    bigArray.append(array1)
    bigArray.append(array2)
    BlackjackSARSA.decay_epsilon(agent)

for i, row in enumerate(bigArray):
    j = 0
    while j < len(row):
        if j+4 < len(row):
            BlackjackSARSA.update(agent, obs=row[j], action=row[j+1],reward=row[j+2],next_obs=row[j+3], next_act = row[j+4])
            j += 3
        else:
            break
value_grid, policy_grid = BlackjackSARSA.create_grids(agent, usable_ace=False)
fig1 = BlackjackSARSA.create_plots(value_grid, policy_grid, title="Without usable ace")
plt.show()

env.close()
