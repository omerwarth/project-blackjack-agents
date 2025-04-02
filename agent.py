#%%
import gymnasium as gym
from blackjack import BlackjackEnv
import matplotlib.pyplot as plt
import time
#%%
# Create the Blackjack environment
env = BlackjackEnv(natural=False, sab=False)
action_map = {0: "Stick", 1: "Hit"}

#%%
for episode in range(1):
    run = True
    state, info = env.reset()
    print(state)
    
    while run:        
        action1 = env.action_space.sample()
        action2 = env.action_space.sample()
        
        print(f'Player 1 will: {action_map[action1]} and player2 will: {action_map[action2]}')
        
        next_state, reward, terminated, truncated, info = env.step((action1, action2))
        
        print(f'The dealer has: {next_state[1]}, and the two hands are now: {next_state[0]} and {next_state[3]}')
        
        time.sleep(1)
        
        if terminated:
            run = False
        
        state = next_state
            
        
# The game is finished, so close the window
env.close()
        
#%%
