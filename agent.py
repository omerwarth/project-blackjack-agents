#%%
import gymnasium as gym
from blackjack import BlackjackEnv
import matplotlib.pyplot as plt
import time
#%%
# Create the Blackjack environment
env = BlackjackEnv(natural=False, sab=False)

# Set an action map
action_map = {0: "Stick", 1: "Hit"}

#%%
# Episode loop for training
for episode in range(1):
    run = True
    
    # Get the initial state by resetting the environment
    state, info = env.reset()
    print(state)
    
    # Main loop for one had of blackjack
    while run:
        # Obtain two random actions
        action1 = env.action_space.sample()
        action2 = env.action_space.sample()
        
        print(f'Player 1 will: {action_map[action1]} and player2 will: {action_map[action2]}')
        
        # Update the state of the environment using env.step()
        next_state, reward, terminated, truncated, info = env.step((action1, action2))
        
        print(f'The dealer has: {next_state[1]}, and the two hands are now: {next_state[0]} and {next_state[3]}')
        
        time.sleep(1)
        
        if terminated:
            run = False
        
        # Update the current state
        state = next_state
            
        
# The game is finished, so close the window
env.close()
