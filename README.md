# Blackjack RL Agents

This project implements and compares two reinforcement learning algorithms—SARSA and Q-learning—on a custom-modified Blackjack environment based on OpenAI Gym's Blackjack. The environment supports two simultaneous players, and the agents learn optimal strategies through self-play. The project visualizes results and allows interactive exploration of learned policies.

## Features

- **Custom Blackjack Environment:**  
  Modified from OpenAI Gym to support two players per game, with graphical rendering using Pygame.
- **SARSA & Q-Learning Agents:**  
  Both agents use tabular methods to learn optimal policies.
- **Training & Evaluation:**  
  Agents are trained over thousands of episodes, and their performance is evaluated and compared.
- **Visualization:**  
  - Win rates for each agent.
  - 3D surface plots of state values.
  - Policy heatmaps.
  - Option to watch a sample game with graphical interface.

## Project Structure

```
agent.py                  # Main script: trains agents, evaluates, and visualizes results
blackjack.py              # Modified Blackjack environment (two-player support, rendering)
BlackjackAgent.py         # Agent class (SARSA & Q-learning logic, plotting utilities)
record_episode_statistics.py # (Unused) Wrapper for episode statistics
font/                     # Font for rendering
images/                   # Card images for rendering
```

## Requirements

- Python 3.8+
- `gymnasium`
- `pygame`
- `numpy`
- `matplotlib`
- `seaborn`

Install dependencies with:

```sh
pip install gymnasium pygame numpy matplotlib seaborn
```

## Usage

Run the main script to train both agents and display results:

```sh
python agent.py
```

You will be prompted to:
- View a sample game (with graphical rendering), or
- View policy/value plots for either agent, with or without a usable ace.

## Example Output

- **Win Rates:**  
  After training, the script prints the win percentages for both SARSA and Q-learning agents over 10,000 games.

- **Visualization:**  
  - 3D value function plots
  - Policy heatmaps (Stick/Hit decisions)
  - Interactive sample game

## Custom Environment

The Blackjack environment (blackjack.py) is extended to allow two players to play against the dealer simultaneously, with full graphical rendering using card images and custom fonts.

## Agent Implementation

The `BlackjackAgent` class implements both SARSA and Q-learning update rules, epsilon-greedy action selection, and plotting utilities for policy/value visualization.

## Credits

- Blackjack environment adapted from [OpenAI Gym](https://github.com/openai/gym).
- Card images and font assets are included for rendering.

---

