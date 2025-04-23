#%%
import time
import gymnasium as gym
import numpy as np
from blackjack import BlackjackEnv
import matplotlib.pyplot as plt
from BlackjackAgent import BlackjackAgent
#%%
# Create the Blackjack environment
env = BlackjackEnv(natural=False, sab=False)

# Set an action map
action_map = {0: "Stick", 1: "Hit"}

n_episodes = 200000
start_epsilon = 1.0

# SARSA agent
agent1 = BlackjackAgent(
    learning_rate=0.01,
    initial_epsilon=1.0,
    epsilon_decay= start_epsilon / (n_episodes / 2),
    final_epsilon=0.1,
    discount_factor=0.95
)

# Q-learning agent
agent2 = BlackjackAgent(
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
    action1 = int(BlackjackAgent.get_action(self = agent1, env = env,  obs = (state[:3])))
    count1 = 0
    count2 = 0
    # Main loop for one had of blackjack
    while not terminated[0] and not terminated[1]:
        action2 = int(BlackjackAgent.get_action(self = agent2, env = env, obs = (state[3:])))

        next_state, reward, terminated, truncated, info = env.step((action1, action2))
        next_act = int(BlackjackAgent.get_action(self = agent1, env = env,  obs = (next_state[:3])))
        if reward[0] != 0:
            next_act = 2.0
            count1 += 1
        if reward[1] != 0:
            count2 += 1
        if count1 <= 1:
            BlackjackAgent.update(agent1, obs=state[:3], action=action1, reward=reward[0], next_obs=next_state[:3], next_act = next_act, type = True)
        if count2 <= 1:
            BlackjackAgent.update(agent2, obs=state[3:], action=action2, reward=reward[1], next_obs=next_state[3:], next_act = next_act, type = False)
        state = next_state
        action1 = next_act
        
    BlackjackAgent.decay_epsilon(agent1)
    BlackjackAgent.decay_epsilon(agent2)


player1 = 0.0
player2 = 0.0
total = 0.0

for i in range(10000):
    obs, _ = env.reset()
    done = [False, False]
    total += 1
    while not all(done):
        actions = [int(BlackjackAgent.get_action(self = agent1, env = env,  obs = (obs[:3]))), int(BlackjackAgent.get_action(self = agent2, env = env,  obs = (obs[3:])))]
        obs, rewards, done, info, _ = env.step(actions)
        player1 += max(rewards[0],0)
        player2 += max(rewards[1],0)

print("In a sample of 10000 games of Blackjack, these were the winning percentages for each algorithm")
print("SARSA: ",player1/total*100, "%")
print("Q-learning: ",player2/total*100, "%\n")
input0 = int(input("Enter 1 if you would like to see a sample game and 2 if you would like to see a graph of optimal policy/value of each state "))

if input0 == 1:
    env = BlackjackEnv(render_mode="human")
    obs, _ = env.reset()
    done = [False, False]
    while not all(done):
        actions = [int(BlackjackAgent.get_action(self = agent1, env = env,  obs = (obs[:3]))), int(BlackjackAgent.get_action(self = agent2, env = env,  obs = (obs[3:])))]
        time.sleep(4)
        obs, rewards, done, info, _ = env.step(actions)
    time.sleep(10)
else:
    ace = False
    input2 = int(input("Enter 1 if you want to see the policy/value for SARSA and 2 if you want to see Q-learning "))
    input3 = int(input("Enter 1 if you want to see the policy/value with a usable ace and 2 for without "))
    title = ""
    if input3 == 1:
        ace = True
    if input2 == 1:
        value_grid, policy_grid = BlackjackAgent.create_grids(agent1, usable_ace=ace)
        title = "SARSA "
    else:
        value_grid, policy_grid = BlackjackAgent.create_grids(agent2, usable_ace=ace)
        title = "Q-learning "
    if input3 == 1:
        title += "with usable ace"
    else:
        title += "without usable ace"
    fig1 = BlackjackAgent.create_plots(value_grid, policy_grid, title=title)
    plt.show()
env.close()
